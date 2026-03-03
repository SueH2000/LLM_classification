#!/usr/bin/env python3
"""Reproducible paper classification with lightweight RAG prompting.

This script classifies papers into:
- Primary: generated own dataset
- Reuse: reused public/external dataset
- Unclear: insufficient evidence

It adds a lightweight RAG step by retrieving the most similar labeled examples
from a local CSV and injecting them into the LLM prompt.

Design goals for beginners:
1) Keep each step explicit and independently testable.
2) Save enough metadata (`run_config.json`) so a run can be reproduced later.
3) Prefer deterministic defaults (`temperature=0`, fixed seed) for easier debugging.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ACC_PATTERNS = {
    "GEO_GSE": re.compile(r"\bGSE\d+\b", re.IGNORECASE),
    "GEO_GSM": re.compile(r"\bGSM\d+\b", re.IGNORECASE),
    "SRA_SRP": re.compile(r"\bSRP\d+\b", re.IGNORECASE),
    "SRA_SRR": re.compile(r"\bSRR\d+\b", re.IGNORECASE),
    "ENA_PRJ": re.compile(r"\bPRJ[EDNA][A-Z0-9]+\b", re.IGNORECASE),
    "ArrayExpress": re.compile(r"\bE-\w{2,3}-\d+\b", re.IGNORECASE),
}
ACC_ANY = re.compile(r"(?i)\b(GSE\d+|GSM\d+|SRP\d+|SRR\d+|E-\w{2,3}-\d+|PRJ[EDNA][A-Z0-9]+)\b")
GEO_WORDS = re.compile(r"(?i)\b(GEO|Gene Expression Omnibus|SRA|ArrayExpress|ENA|NCBI)\b")
PROV_WORDS = re.compile(
    r"(?i)\b(downloaded|retrieved|obtained|reanaly[sz]ed|re-analysed|publicly available|"
    r"deposited|submitted|accession|available at|data availability|data are available)\b"
)
PRIMARY_STRONG = re.compile(
    r"(?i)\b(we (collected|recruited|enrolled|sequenced|generated|performed rna-?seq|acquired)|"
    r"library preparation|sample collection|patients were recruited|ethics approval|informed consent|our cohort)\b"
)
REUSE_STRONG = re.compile(
    r"(?i)\b(downloaded|retrieved|obtained from|publicly available|reanaly[sz]ed|secondary analysis)\b"
)
DEPOSIT = re.compile(r"(?i)\b(deposited|submitted)\b")
WE_OUR = re.compile(r"(?i)\b(we|our)\b")


@dataclass
class Config:
    """Single source of truth for run-time configuration.

    Why a dataclass helps reproducibility:
    - Keeps all parameters in one object instead of hidden globals.
    - Can be serialized directly to JSON so a run is auditable.
    - Makes experiment comparison easier because every run has the same schema.
    """
    jsonl_path: Path
    out_dir: Path
    labeled_csv_path: Optional[Path]
    ollama_url: str = "http://localhost:11434/api/generate"
    ollama_model: str = "llama3"
    llm_mode: str = "unclear_only"
    seed: int = 42
    win_before: int = 350
    win_after: int = 900
    max_evidence_chars: int = 2200
    temperature: float = 0.0
    num_predict: int = 220
    timeout_connect: int = 10
    timeout_read: int = 120
    rag_top_k: int = 3
    rag_max_examples: int = 30
    rag_per_label_cap: int = 2
    rag_min_similarity: float = 0.05
    rag_candidate_pool: int = 12
    phrase_hints_path: Optional[Path] = None
    phrase_hints_per_class: int = 8


def set_global_seed(seed: int) -> None:
    """Set all common random seeds used in this script.

    Note: this controls python/numpy randomness. LLM endpoints may still be
    partially non-deterministic depending on back-end implementation.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def extract_text_fields(rec: Dict[str, Any]) -> str:
    """Concatenate known text fields from one JSON record.

    We intentionally keep the field list short and explicit to avoid silently
    pulling unexpected noisy fields from heterogeneous JSON sources.
    """
    parts: List[str] = []
    for k in ["title", "abstract", "full_text", "body", "text"]:
        v = rec.get(k)
        if isinstance(v, str) and v.strip():
            parts.append(v.strip())
    return "\n\n".join(parts)


def extract_accessions(text: str) -> Dict[str, List[str]]:
    t = text or ""
    acc: Dict[str, List[str]] = {}
    for k, pat in ACC_PATTERNS.items():
        acc[k] = sorted(set(m.group(0).upper() for m in pat.finditer(t)))
    return acc


def format_accession_string(acc: Dict[str, List[str]], per_key_cap: int = 10) -> str:
    """Convert accession dictionary to a compact prompt-friendly string."""
    parts = []
    for k, v in acc.items():
        if v:
            parts.append(f"{k}: {', '.join(v[:per_key_cap])}")
    return "; ".join(parts) if parts else "None"


def evidence_windows(
    text: str,
    win_before: int,
    win_after: int,
    max_chars: int,
    max_hits: int = 40,
) -> Tuple[List[str], str]:
    """Build an evidence bundle for classification.

    Strategy:
    1) Prefer windows around accession IDs (usually strongest provenance clues).
    2) If no accession windows found, fall back to provenance keyword windows.
    3) De-duplicate near-identical chunks to reduce prompt noise.

    Returns:
    - list of chunks (for debugging/inspection)
    - merged evidence string truncated to `max_chars` for stable token budget
    """
    flat = re.sub(r"\s+", " ", (text or "")).strip()
    if not flat:
        return [], ""

    hits: List[str] = []
    for m in ACC_ANY.finditer(flat):
        s = max(0, m.start() - win_before)
        e = min(len(flat), m.end() + win_after)
        chunk = flat[s:e]
        if GEO_WORDS.search(chunk) or PROV_WORDS.search(chunk):
            hits.append(chunk)
        if len(hits) >= max_hits:
            break

    if not hits:
        for m in PROV_WORDS.finditer(flat):
            s = max(0, m.start() - win_before)
            e = min(len(flat), m.end() + win_after)
            hits.append(flat[s:e])
            if len(hits) >= max_hits:
                break

    dedup: List[str] = []
    seen = set()
    for h in hits:
        key = h[:140].lower()
        if key in seen:
            continue
        seen.add(key)
        dedup.append(h)

    dedup.sort(key=lambda c: PROV_WORDS.search(c) is not None, reverse=True)
    evidence = re.sub(r"\s+", " ", " ... ".join(dedup)).strip()
    return dedup, evidence[:max_chars]


def heuristic_triage(evidence: str, accessions: Dict[str, List[str]]) -> Tuple[str, float]:
    """Cheap first-pass classifier before optional LLM call.

    This stage improves efficiency: obvious cases are solved quickly,
    while uncertain ones can be escalated to LLM (`llm_mode=unclear_only`).
    """
    ev = evidence or ""
    if REUSE_STRONG.search(ev):
        return "Reuse", 0.85
    if PRIMARY_STRONG.search(ev):
        return "Primary", 0.80
    if DEPOSIT.search(ev) and WE_OUR.search(ev) and any(accessions.values()):
        return "Primary", 0.65
    return "Unclear", 0.50


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    """Yield records from JSONL and skip malformed lines safely.

    Skipping bad lines is often better than hard-failing long batch jobs.
    Warning messages retain enough context (line number + parse error).
    """
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[warn] JSON decode failed at line {i}: {exc}")


class ExampleRetriever:
    """Lightweight retriever over labeled examples (TF-IDF + cosine similarity).

    This is intentionally simple and local:
    - no vector database required
    - no embedding API dependency
    - very fast to iterate for small/medium labeled sets
    """

    def __init__(self, csv_path: Optional[Path], max_examples: int = 30, seed: int = 42):
        self.df: Optional[pd.DataFrame] = None
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.matrix = None
        if csv_path is None or not csv_path.exists():
            return

        df = pd.read_csv(csv_path, encoding="latin1")
        label_col = "ground_truth" if "ground_truth" in df.columns else "human_label"
        if label_col not in df.columns:
            return
        work = df[[c for c in ["paper_id", "title", "abstract", "full_text", label_col] if c in df.columns]].copy()
        work = work.rename(columns={label_col: "label"})
        work["label"] = work["label"].astype(str).str.strip().str.title()
        work = work[work["label"].isin(["Primary", "Reuse"])].copy()
        if work.empty:
            return

        work["text_for_retrieval"] = (
            work.get("title", "").fillna("") + " " + work.get("abstract", "").fillna("") + " " + work.get("full_text", "").fillna("")
        ).str.replace(r"\s+", " ", regex=True)

        # Keep labels balanced so retrieval does not overfit to whichever class
        # appears first or appears more often in the labeled CSV.
        per_label_budget = max(1, max_examples // 2)
        sampled_parts: List[pd.DataFrame] = []
        for label in ["Primary", "Reuse"]:
            part = work[work["label"] == label]
            if part.empty:
                continue
            sampled_parts.append(part.sample(n=min(per_label_budget, len(part)), random_state=seed))

        if sampled_parts:
            work = pd.concat(sampled_parts, ignore_index=True)
        if len(work) > max_examples:
            work = work.sample(n=max_examples, random_state=seed)
        work = work.reset_index(drop=True)
        self.vectorizer = TfidfVectorizer(max_features=6000, ngram_range=(1, 2), stop_words="english")
        self.matrix = self.vectorizer.fit_transform(work["text_for_retrieval"])
        self.df = work

    def retrieve(
        self,
        query_text: str,
        top_k: int = 3,
        min_similarity: float = 0.05,
        candidate_pool: int = 12,
        per_label_cap: int = 2,
    ) -> List[Dict[str, str]]:
        """Return top-k nearest labeled examples for prompt augmentation."""
        if self.df is None or self.vectorizer is None or self.matrix is None:
            return []
        qv = self.vectorizer.transform([query_text])
        sims = cosine_similarity(qv, self.matrix).flatten()
        idxs = np.argsort(-sims)[: max(top_k, candidate_pool)]
        rows = []
        per_label_count: Dict[str, int] = {"Primary": 0, "Reuse": 0}
        for i in idxs:
            sim = float(sims[i])
            if sim < min_similarity:
                continue
            row = self.df.iloc[int(i)]
            label = str(row["label"])
            if per_label_count.get(label, 0) >= per_label_cap:
                continue

            rows.append(
                {
                    "paper_id": str(row.get("paper_id", "")),
                    "label": label,
                    "score": f"{sim:.3f}",
                    "snippet": str(row.get("text_for_retrieval", ""))[:280],
                }
            )
            per_label_count[label] = per_label_count.get(label, 0) + 1
            if len(rows) >= top_k:
                break
        return rows




def load_phrase_hints(path: Optional[Path], per_class: int = 8) -> Dict[str, List[str]]:
    """Load mined phrases (from evidence_modeling.py) for prompt guidance.

    Expected JSON format:
      {"Primary": [{"phrase": "...", "score": ...}, ...], "Reuse": [...]}
    """
    if path is None or not path.exists():
        return {"Primary": [], "Reuse": []}

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"Primary": [], "Reuse": []}

    out: Dict[str, List[str]] = {"Primary": [], "Reuse": []}
    for cls in ["Primary", "Reuse"]:
        items = data.get(cls, [])
        phrases: List[str] = []
        for it in items:
            if isinstance(it, dict):
                ph = str(it.get("phrase", "")).strip()
            else:
                ph = str(it).strip()
            if ph:
                phrases.append(ph)
        out[cls] = phrases[: max(0, int(per_class))]
    return out


def build_prompt(accessions: str, evidence: str, rag_examples: List[Dict[str, str]], phrase_hints: Dict[str, List[str]]) -> str:
    """Assemble final LLM prompt with retrieved examples and target evidence.

    We request strict JSON output to simplify downstream parsing and logging.
    """
    if rag_examples:
        examples_text = "\n".join(
            f"- label={ex['label']} sim={ex['score']} snippet={ex['snippet']}"
            for ex in rag_examples
        )
    else:
        examples_text = "(none)"

    primary_hints = ", ".join(phrase_hints.get("Primary", [])) or "(none)"
    reuse_hints = ", ".join(phrase_hints.get("Reuse", [])) or "(none)"

    return f"""You are classifying whether a paper GENERATED its dataset (Primary) or REUSED a public/external dataset (Reuse).

Definitions:
- Primary: authors generated the dataset used for analysis (even if they later deposited it).
- Reuse: authors downloaded/obtained public or external datasets.
- Unclear: evidence is insufficient.

Retrieved labeled examples (for guidance only):
{examples_text}

Mined phrase hints (soft cues, not strict rules):
- Primary-like phrases: {primary_hints}
- Reuse-like phrases: {reuse_hints}

Target paper accessions:
{accessions}

Target paper evidence:
{evidence}

Output JSON only:
{{"label":"Primary|Reuse|Unclear","confidence":0.0-1.0,"rationale":"short evidence-based reason"}}"""


def call_ollama(prompt: str, cfg: Config) -> Dict[str, Any]:
    """Call Ollama endpoint and parse a resilient JSON result.

    Parser behavior is intentionally defensive: if output is malformed, return
    a conservative `Unclear` fallback instead of raising and aborting the run.
    """
    payload = {
        "model": cfg.ollama_model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": cfg.temperature,
            "num_predict": cfg.num_predict,
        },
    }
    r = requests.post(
        cfg.ollama_url,
        json=payload,
        timeout=(cfg.timeout_connect, cfg.timeout_read),
    )
    r.raise_for_status()
    raw = r.json().get("response", "").strip()

    match = re.search(r"\{.*\}", raw, flags=re.S)
    if not match:
        return {"label": "Unclear", "confidence": 0.5, "rationale": "No JSON found", "raw": raw}
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"label": "Unclear", "confidence": 0.5, "rationale": "Invalid JSON", "raw": raw}

    label = str(obj.get("label", "Unclear")).title()
    if label not in {"Primary", "Reuse", "Unclear"}:
        label = "Unclear"
    try:
        conf = float(obj.get("confidence", 0.5))
    except Exception:
        conf = 0.5
    conf = max(0.0, min(1.0, conf))
    return {
        "label": label,
        "confidence": conf,
        "rationale": str(obj.get("rationale", "")).strip(),
        "raw": raw,
    }


def run(cfg: Config) -> Path:
    """Execute full pipeline and save artifacts.

    Artifacts:
    - predictions_rag.csv : row-level predictions and rationale
    - run_config.json     : exact settings used in this run
    """
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    set_global_seed(cfg.seed)

    retriever = ExampleRetriever(cfg.labeled_csv_path, max_examples=cfg.rag_max_examples, seed=cfg.seed)
    phrase_hints = load_phrase_hints(cfg.phrase_hints_path, per_class=cfg.phrase_hints_per_class)

    rows: List[Dict[str, Any]] = []
    for rec in iter_jsonl(cfg.jsonl_path):
        paper_id = rec.get("paper_id") or rec.get("pmcid") or rec.get("doi") or rec.get("id") or ""
        text = extract_text_fields(rec)
        acc = extract_accessions(text)
        acc_str = format_accession_string(acc)
        _, evidence = evidence_windows(
            text,
            win_before=cfg.win_before,
            win_after=cfg.win_after,
            max_chars=cfg.max_evidence_chars,
        )

        h_label, h_conf = heuristic_triage(evidence, acc)
        rag_examples = retriever.retrieve(
            evidence or text,
            top_k=cfg.rag_top_k,
            min_similarity=cfg.rag_min_similarity,
            candidate_pool=cfg.rag_candidate_pool,
            per_label_cap=cfg.rag_per_label_cap,
        )

        do_llm = cfg.llm_mode == "all" or (cfg.llm_mode == "unclear_only" and h_label == "Unclear")
        llm_label, llm_conf, llm_reason = h_label, h_conf, "heuristic-only"

        if do_llm:
            prompt = build_prompt(acc_str, evidence, rag_examples, phrase_hints)
            try:
                out = call_ollama(prompt, cfg)
                llm_label, llm_conf, llm_reason = out["label"], out["confidence"], out["rationale"]
            except Exception as exc:
                llm_label, llm_conf, llm_reason = "Unclear", 0.5, f"LLM error: {exc}"

        rows.append(
            {
                "paper_id": paper_id,
                "heuristic_label": h_label,
                "heuristic_conf": h_conf,
                "llm_label": llm_label,
                "llm_conf": llm_conf,
                "llm_reason": llm_reason,
                "accessions": acc_str,
                "rag_examples": json.dumps(rag_examples, ensure_ascii=False),
            }
        )

    preds = pd.DataFrame(rows)
    out_csv = cfg.out_dir / "predictions_rag.csv"
    preds.to_csv(out_csv, index=False)

    with (cfg.out_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2, default=str)

    return out_csv




def run_notebook(
    jsonl_path: str,
    labeled_csv_path: Optional[str] = None,
    out_dir: str = "outputs_rag",
    llm_mode: str = "unclear_only",
    ollama_model: str = "llama3",
    ollama_url: str = "http://localhost:11434/api/generate",
    seed: int = 42,
    rag_top_k: int = 3,
    rag_max_examples: int = 30,
    rag_per_label_cap: int = 2,
    rag_min_similarity: float = 0.05,
    rag_candidate_pool: int = 12,
    phrase_hints_path: Optional[str] = None,
    phrase_hints_per_class: int = 8,
) -> pd.DataFrame:
    """Notebook-friendly wrapper around the CLI pipeline.

    Why this helper exists (beginner explanation):
    - In Jupyter, passing Python variables is easier than building shell commands.
    - This wrapper builds the same Config object used by CLI, so behavior stays aligned.
    - It returns a DataFrame directly so you can inspect results immediately.

    Example (inside notebook):
        preds = run_notebook(
            jsonl_path="pmc_gse_articles_clean.jsonl",
            labeled_csv_path="manual_ground_truth_with_GSE_links_REFRESHED.csv",
            llm_mode="unclear_only",
            ollama_model="llama3.1:8b",
        )
        preds.head()
    """
    cfg = Config(
        jsonl_path=Path(jsonl_path),
        out_dir=Path(out_dir),
        labeled_csv_path=Path(labeled_csv_path) if labeled_csv_path else None,
        ollama_url=ollama_url,
        ollama_model=ollama_model,
        llm_mode=llm_mode,
        seed=seed,
        rag_top_k=rag_top_k,
        rag_max_examples=rag_max_examples,
        rag_per_label_cap=rag_per_label_cap,
        rag_min_similarity=rag_min_similarity,
        rag_candidate_pool=rag_candidate_pool,
        phrase_hints_path=Path(phrase_hints_path) if phrase_hints_path else None,
        phrase_hints_per_class=phrase_hints_per_class,
    )

    out_csv = run(cfg)
    return pd.read_csv(out_csv)


def parse_args() -> Config:
    """Parse CLI arguments and map them into a Config object."""
    p = argparse.ArgumentParser(description="Reproducible classification with lightweight RAG")
    p.add_argument("--jsonl-path", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("outputs_rag"))
    p.add_argument("--labeled-csv-path", type=Path, default=None)
    p.add_argument("--ollama-url", default="http://localhost:11434/api/generate")
    p.add_argument("--ollama-model", default="llama3")
    p.add_argument("--llm-mode", choices=["off", "unclear_only", "all"], default="unclear_only")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rag-top-k", type=int, default=3)
    p.add_argument("--rag-max-examples", type=int, default=30)
    p.add_argument("--rag-per-label-cap", type=int, default=2)
    p.add_argument("--rag-min-similarity", type=float, default=0.05)
    p.add_argument("--rag-candidate-pool", type=int, default=12)
    p.add_argument("--phrase-hints-path", type=Path, default=None, help="Optional mined_phrases.json from evidence_modeling.py")
    p.add_argument("--phrase-hints-per-class", type=int, default=8)
    args = p.parse_args()

    return Config(
        jsonl_path=args.jsonl_path,
        out_dir=args.out_dir,
        labeled_csv_path=args.labeled_csv_path,
        ollama_url=args.ollama_url,
        ollama_model=args.ollama_model,
        llm_mode=args.llm_mode,
        seed=args.seed,
        rag_top_k=args.rag_top_k,
        rag_max_examples=args.rag_max_examples,
        rag_per_label_cap=args.rag_per_label_cap,
        rag_min_similarity=args.rag_min_similarity,
        rag_candidate_pool=args.rag_candidate_pool,
        phrase_hints_path=args.phrase_hints_path,
        phrase_hints_per_class=args.phrase_hints_per_class,
    )


if __name__ == "__main__":
    config = parse_args()
    output = run(config)
    print(f"Saved predictions: {output}")
