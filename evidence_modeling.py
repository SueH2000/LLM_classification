#!/usr/bin/env python3
"""Systematic evidence modeling for Primary vs Reuse classification.

This script helps you move from manual keyword lists to a more data-driven setup:
1) Auto phrase mining from labeled CSV (top-k phrases per class)
2) Convert mined phrases into pattern templates (regex)
3) Train a sentence-level linear model on evidence text
4) Compare all versions on the same evaluation split

Beginner note:
- "Sentence-level" here means each row uses evidence-centric text snippets,
  rather than full noisy paper text.
- We keep the workflow explicit and reproducible with fixed random seeds.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# --- Keep core provenance cues aligned with your existing pipeline ---
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
    test_size: float = 0.2
    random_state: int = 42
    top_k_phrases: int = 40
    ngram_min: int = 1
    ngram_max: int = 3
    min_df: int = 2
    max_features: int = 20000


def normalize_label(x: Any) -> str:
    """Map noisy human labels into {Primary, Reuse, Unclear}."""
    if pd.isna(x):
        return "Unclear"
    s = str(x).strip().lower()
    if s in {"primary", "p", "generated", "own", "new"} or "primary" in s:
        return "Primary"
    if s in {"reuse", "re-used", "reused", "secondary", "public", "old"} or "reuse" in s:
        return "Reuse"
    return "Unclear"


def evidence_windows(
    text: str,
    win_before: int = 350,
    win_after: int = 900,
    max_chars: int = 2200,
    max_hits: int = 40,
) -> str:
    """Build evidence-centric text windows around provenance clues."""
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
    """Original-style heuristic baseline using static regex rules."""
    if REUSE_STRONG.search(text):
        return "Reuse"
    if PRIMARY_STRONG.search(text):
        return "Primary"
    if DEPOSIT.search(text) and WE_OUR.search(text):
        return "Primary"
    return "Unclear"


def _safe_pattern_from_phrase(phrase: str) -> str:
    """Convert phrase into a whitespace-tolerant regex snippet.

    Example:
    "publicly available" -> "publicly\\s+available"
    """
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
    """Mine discriminative n-grams with simple log-odds scoring.

    We estimate phrase strength by comparing class-relative frequencies.
    """
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
        results[cls] = [
            {"phrase": str(vocab[i]), "score": float(log_odds[i])}
            for i in idx
        ]

    return results


def build_template_rules(mined: Dict[str, List[Dict[str, float]]]) -> Dict[str, List[re.Pattern[str]]]:
    """Convert mined phrases into regex templates by class."""
    rules: Dict[str, List[re.Pattern[str]]] = {"Primary": [], "Reuse": []}
    for cls in ["Primary", "Reuse"]:
        for item in mined.get(cls, []):
            patt = _safe_pattern_from_phrase(item["phrase"])
            if patt:
                rules[cls].append(re.compile(patt, re.IGNORECASE))
    return rules


def template_rule_predict(text: str, rules: Dict[str, List[re.Pattern[str]]]) -> str:
    """Predict label by counting matched class templates on evidence text."""
    t = text or ""
    p_score = sum(1 for r in rules["Primary"] if r.search(t))
    r_score = sum(1 for r in rules["Reuse"] if r.search(t))

    if p_score > r_score:
        return "Primary"
    if r_score > p_score:
        return "Reuse"
    return "Unclear"


def evaluate_predictions(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, Any]:
    """Compute compact metrics on all rows and binary subset only."""
    all_acc = float(accuracy_score(y_true, y_pred))

    mask = y_true.isin(["Primary", "Reuse"]) & y_pred.isin(["Primary", "Reuse"])
    if int(mask.sum()) == 0:
        binary_acc = np.nan
    else:
        binary_acc = float(accuracy_score(y_true[mask], y_pred[mask]))

    return {
        "accuracy_all": all_acc,
        "accuracy_binary_on_covered": binary_acc,
        "coverage_binary_pred": float(y_pred.isin(["Primary", "Reuse"]).mean()),
        "unclear_rate": float((y_pred == "Unclear").mean()),
        "n": int(len(y_true)),
    }


def build_evidence_text(df: pd.DataFrame) -> pd.Series:
    """Create evidence text from title/abstract/full_text fields per row."""
    text = (
        df.get("title", "").fillna("").astype(str)
        + " "
        + df.get("abstract", "").fillna("").astype(str)
        + " "
        + df.get("full_text", "").fillna("").astype(str)
    )
    return text.apply(evidence_windows)


def train_and_compare(cfg: ModelingConfig) -> None:
    """Run the full experiment and write all comparison artifacts."""
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(cfg.labeled_csv_path, encoding="latin1")

    label_col = "ground_truth" if "ground_truth" in df.columns else "human_label"
    if label_col not in df.columns:
        raise ValueError("CSV must contain 'ground_truth' or 'human_label'.")

    df = df.copy()
    df["label"] = df[label_col].apply(normalize_label)
    df = df[df["label"].isin(["Primary", "Reuse"])].copy()
    if df.empty:
        raise ValueError("No valid Primary/Reuse rows after normalization.")

    df["evidence_text"] = build_evidence_text(df)

    train_df, test_df = train_test_split(
        df,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=df["label"],
    )

    # 1) Baseline static heuristic
    test_df = test_df.copy()
    test_df["pred_static_rules"] = test_df["evidence_text"].apply(classic_heuristic)

    # 2) Auto phrase mining + template rules (train split only)
    mined = mine_top_phrases(
        texts=train_df["evidence_text"],
        labels=train_df["label"],
        top_k=cfg.top_k_phrases,
        ngram_range=(cfg.ngram_min, cfg.ngram_max),
        min_df=cfg.min_df,
        max_features=cfg.max_features,
    )
    template_rules = build_template_rules(mined)
    test_df["pred_mined_templates"] = test_df["evidence_text"].apply(
        lambda x: template_rule_predict(x, template_rules)
    )

    # 3) Sentence-level linear model (TF-IDF + logistic regression)
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

    # Save artifacts
    metrics_path = cfg.out_dir / "model_comparison.csv"
    metrics_df.reset_index().to_csv(metrics_path, index=False)

    preds_path = cfg.out_dir / "test_predictions.csv"
    cols = ["label", "evidence_text", "pred_static_rules", "pred_mined_templates", "pred_linear_model"]
    if "paper_id" in test_df.columns:
        cols = ["paper_id"] + cols
    test_df[cols].to_csv(preds_path, index=False)

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
