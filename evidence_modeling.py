#!/usr/bin/env python3
"""Systematic evidence modeling for Primary vs Reuse classification.

This script provides a reproducible upgrade path:
1) Auto phrase mining from labeled data (top-k phrases per class)
2) Convert mined phrases into regex pattern templates
3) Train a sentence-level linear model on evidence text
4) Compare all methods on the same train/test split

Important data assumption:
- You can provide article content in JSONL and labels in CSV.
- They are matched by `paper_id` (recommended setup).
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

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
class ModelingConfig:
    labeled_csv_path: Path
    out_dir: Path
    jsonl_path: Optional[Path] = None
    test_size: float = 0.2
    random_state: int = 42
    top_k_phrases: int = 40
    ngram_min: int = 1
    ngram_max: int = 3
    min_df: int = 2
    max_features: int = 20000


def normalize_label(x: Any) -> str:
    if pd.isna(x):
        return "Unclear"
    s = str(x).strip().lower()
    if s in {"primary", "p", "generated", "own", "new"} or "primary" in s:
        return "Primary"
    if s in {"reuse", "re-used", "reused", "secondary", "public", "old"} or "reuse" in s:
        return "Reuse"
    return "Unclear"


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[warn] JSON decode failed at line {i}: {exc}")


def extract_text_fields(rec: Dict[str, Any]) -> str:
    parts: List[str] = []
    for k in ["title", "abstract", "full_text", "body", "text"]:
        v = rec.get(k)
        if isinstance(v, str) and v.strip():
            parts.append(v.strip())
    return "\n\n".join(parts)


def evidence_windows(
    text: str,
    win_before: int = 350,
    win_after: int = 900,
    max_chars: int = 2200,
    max_hits: int = 40,
) -> str:
    flat = re.sub(r"\s+", " ", text or "").strip()
    if not flat:
        return ""

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

    dedup, seen = [], set()
    for h in hits:
        key = h[:140].lower()
        if key in seen:
            continue
        seen.add(key)
        dedup.append(h)

    evidence = re.sub(r"\s+", " ", " ... ".join(dedup)).strip()
    return evidence[:max_chars]


def classic_heuristic(text: str) -> str:
    if REUSE_STRONG.search(text):
        return "Reuse"
    if PRIMARY_STRONG.search(text):
        return "Primary"
    if DEPOSIT.search(text) and WE_OUR.search(text):
        return "Primary"
    return "Unclear"


def _safe_pattern_from_phrase(phrase: str) -> str:
    tokens = [re.escape(t) for t in phrase.lower().split() if t.strip()]
    if not tokens:
        return ""
    return r"\b" + r"\s+".join(tokens) + r"\b"


def mine_top_phrases(
    texts: pd.Series,
    labels: pd.Series,
    top_k: int,
    ngram_range: Tuple[int, int],
    min_df: int,
    max_features: int,
) -> Dict[str, List[Dict[str, float]]]:
    vec = CountVectorizer(
        ngram_range=ngram_range,
        lowercase=True,
        stop_words="english",
        min_df=min_df,
        max_features=max_features,
    )
    X = vec.fit_transform(texts.fillna(""))
    vocab = np.array(vec.get_feature_names_out())

    results: Dict[str, List[Dict[str, float]]] = {}
    eps = 1.0

    for cls in ["Primary", "Reuse"]:
        mask_cls = labels == cls
        mask_other = labels != cls
        c_cls = np.asarray(X[mask_cls].sum(axis=0)).ravel() + eps
        c_oth = np.asarray(X[mask_other].sum(axis=0)).ravel() + eps
        p_cls = c_cls / c_cls.sum()
        p_oth = c_oth / c_oth.sum()
        log_odds = np.log(p_cls / p_oth)

        idx = np.argsort(-log_odds)[:top_k]
        results[cls] = [{"phrase": str(vocab[i]), "score": float(log_odds[i])} for i in idx]

    return results


def build_template_rules(mined: Dict[str, List[Dict[str, float]]]) -> Dict[str, List[re.Pattern[str]]]:
    rules: Dict[str, List[re.Pattern[str]]] = {"Primary": [], "Reuse": []}
    for cls in ["Primary", "Reuse"]:
        for item in mined.get(cls, []):
            patt = _safe_pattern_from_phrase(item["phrase"])
            if patt:
                rules[cls].append(re.compile(patt, re.IGNORECASE))
    return rules


def template_rule_predict(text: str, rules: Dict[str, List[re.Pattern[str]]]) -> str:
    t = text or ""
    p_score = sum(1 for r in rules["Primary"] if r.search(t))
    r_score = sum(1 for r in rules["Reuse"] if r.search(t))
    if p_score > r_score:
        return "Primary"
    if r_score > p_score:
        return "Reuse"
    return "Unclear"


def evaluate_predictions(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, Any]:
    all_acc = float(accuracy_score(y_true, y_pred))
    mask = y_true.isin(["Primary", "Reuse"]) & y_pred.isin(["Primary", "Reuse"])
    binary_acc = float(accuracy_score(y_true[mask], y_pred[mask])) if int(mask.sum()) > 0 else np.nan
    return {
        "accuracy_all": all_acc,
        "accuracy_binary_on_covered": binary_acc,
        "coverage_binary_pred": float(y_pred.isin(["Primary", "Reuse"]).mean()),
        "unclear_rate": float((y_pred == "Unclear").mean()),
        "n": int(len(y_true)),
    }


def load_article_text_from_jsonl(jsonl_path: Path) -> pd.DataFrame:
    """Load article text from JSONL and return unique paper_id -> article_text mapping."""
    rows: List[Dict[str, str]] = []
    for rec in iter_jsonl(jsonl_path):
        paper_id = rec.get("paper_id") or rec.get("pmcid") or rec.get("doi") or rec.get("id")
        if not paper_id:
            continue
        rows.append({"paper_id": str(paper_id), "article_text": extract_text_fields(rec)})

    if not rows:
        return pd.DataFrame(columns=["paper_id", "article_text"])

    text_df = pd.DataFrame(rows).drop_duplicates(subset=["paper_id"], keep="first")
    return text_df


def build_labeled_dataset(cfg: ModelingConfig) -> pd.DataFrame:
    """Build training table with labels + evidence text.

    If JSONL is provided: merge CSV labels with JSONL articles on paper_id.
    Else: fall back to text columns inside CSV itself.
    """
    labels_df = pd.read_csv(cfg.labeled_csv_path, encoding="latin1")

    if "paper_id" not in labels_df.columns:
        raise ValueError("Labeled CSV must contain 'paper_id' for safe matching.")

    label_col = "ground_truth" if "ground_truth" in labels_df.columns else "human_label"
    if label_col not in labels_df.columns:
        raise ValueError("CSV must contain 'ground_truth' or 'human_label'.")

    labels_df = labels_df.copy()
    labels_df["paper_id"] = labels_df["paper_id"].astype(str)
    labels_df["label"] = labels_df[label_col].apply(normalize_label)
    labels_df = labels_df[labels_df["label"].isin(["Primary", "Reuse"])].copy()

    if cfg.jsonl_path is not None:
        text_df = load_article_text_from_jsonl(cfg.jsonl_path)
        text_df["paper_id"] = text_df["paper_id"].astype(str)
        merged = labels_df.merge(text_df, on="paper_id", how="inner")

        dropped = len(labels_df) - len(merged)
        if dropped > 0:
            print(f"[info] Dropped {dropped} labeled rows not found in JSONL by paper_id.")

        merged["evidence_text"] = merged["article_text"].apply(evidence_windows)
        return merged

    combined = (
        labels_df.get("title", "").fillna("").astype(str)
        + " "
        + labels_df.get("abstract", "").fillna("").astype(str)
        + " "
        + labels_df.get("full_text", "").fillna("").astype(str)
    )
    labels_df["evidence_text"] = combined.apply(evidence_windows)
    return labels_df


def train_and_compare(cfg: ModelingConfig) -> None:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    df = build_labeled_dataset(cfg)

    if df.empty:
        raise ValueError("No matched/usable rows found after label filtering and optional JSONL merge.")

    train_df, test_df = train_test_split(
        df,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=df["label"],
    )

    test_df = test_df.copy()
    test_df["pred_static_rules"] = test_df["evidence_text"].apply(classic_heuristic)

    mined = mine_top_phrases(
        texts=train_df["evidence_text"],
        labels=train_df["label"],
        top_k=cfg.top_k_phrases,
        ngram_range=(cfg.ngram_min, cfg.ngram_max),
        min_df=cfg.min_df,
        max_features=cfg.max_features,
    )
    template_rules = build_template_rules(mined)
    test_df["pred_mined_templates"] = test_df["evidence_text"].apply(lambda x: template_rule_predict(x, template_rules))

    vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", min_df=2, max_features=20000)
    X_train = vec.fit_transform(train_df["evidence_text"].fillna(""))
    X_test = vec.transform(test_df["evidence_text"].fillna(""))

    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=cfg.random_state)
    clf.fit(X_train, train_df["label"])
    test_df["pred_linear_model"] = clf.predict(X_test)

    rows = []
    for name, col in [
        ("static_rules", "pred_static_rules"),
        ("mined_template_rules", "pred_mined_templates"),
        ("linear_model", "pred_linear_model"),
    ]:
        m = evaluate_predictions(test_df["label"], test_df[col])
        m["model"] = name
        rows.append(m)

    metrics_df = pd.DataFrame(rows).set_index("model")
    metrics_path = cfg.out_dir / "model_comparison.csv"
    metrics_df.reset_index().to_csv(metrics_path, index=False)

    preds_path = cfg.out_dir / "test_predictions.csv"
    cols = ["paper_id", "label", "evidence_text", "pred_static_rules", "pred_mined_templates", "pred_linear_model"]
    existing_cols = [c for c in cols if c in test_df.columns]
    test_df[existing_cols].to_csv(preds_path, index=False)

    mined_path = cfg.out_dir / "mined_phrases.json"
    with mined_path.open("w", encoding="utf-8") as f:
        json.dump(mined, f, ensure_ascii=False, indent=2)

    report_path = cfg.out_dir / "linear_model_report.txt"
    rep = classification_report(test_df["label"], test_df["pred_linear_model"], digits=4)
    report_path.write_text(rep, encoding="utf-8")

    print(f"Saved: {metrics_path}")
    print(f"Saved: {preds_path}")
    print(f"Saved: {mined_path}")
    print(f"Saved: {report_path}")
    print("\n=== Model comparison ===")
    print(metrics_df)


def parse_args() -> ModelingConfig:
    p = argparse.ArgumentParser(description="Train and compare systematic evidence models")
    p.add_argument("--labeled-csv-path", type=Path, required=True)
    p.add_argument("--jsonl-path", type=Path, default=None, help="Optional article JSONL matched with CSV by paper_id")
    p.add_argument("--out-dir", type=Path, default=Path("outputs_evidence_modeling"))
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--top-k-phrases", type=int, default=40)
    p.add_argument("--ngram-min", type=int, default=1)
    p.add_argument("--ngram-max", type=int, default=3)
    p.add_argument("--min-df", type=int, default=2)
    p.add_argument("--max-features", type=int, default=20000)
    args = p.parse_args()

    return ModelingConfig(
        labeled_csv_path=args.labeled_csv_path,
        jsonl_path=args.jsonl_path,
        out_dir=args.out_dir,
        test_size=args.test_size,
        random_state=args.random_state,
        top_k_phrases=args.top_k_phrases,
        ngram_min=args.ngram_min,
        ngram_max=args.ngram_max,
        min_df=args.min_df,
        max_features=args.max_features,
    )


if __name__ == "__main__":
    cfg = parse_args()
    train_and_compare(cfg)
