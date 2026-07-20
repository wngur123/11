from __future__ import annotations

import argparse
import json
import random
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

import pandas as pd


PROJECT_LABELS = [
    "TRUE",
    "MOSTLY_TRUE",
    "MIXED_CONTEXT_DISTORTED",
    "FALSE",
    "UNVERIFIABLE",
]

REASON_CODES = [
    "INSUFFICIENT_EVIDENCE",
    "CONFLICTING_EVIDENCE",
    "TEMPORAL_MISMATCH",
    "NON_VERIFIABLE_CLAIM",
    "UNSUPPORTED_INPUT",
]

LABEL_TO_ID = {label: index for index, label in enumerate(PROJECT_LABELS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}

DEFAULT_MODEL_NAME = "skt/kobert-base-v1"
DEFAULT_MAX_LENGTH = 256
DEFAULT_SEED = 42


@dataclass
class ClaimCandidate:
    claim_id: str
    text: str
    sentence_index: int
    start_offset: int
    end_offset: int
    is_checkable: bool
    non_verifiable_reason: str | None = None


@dataclass
class EvidenceItem:
    evidence_id: str
    text: str
    source_name: str
    source_tier: str | None = None
    url: str | None = None
    published_at: str | None = None
    score: float | None = None
    retrieval_score: float | None = None
    source_tier_score: float | None = None
    temporal_score: float | None = None
    reliability_score: float | None = None


@dataclass
class ClaimVerdict:
    claim: str
    label: str
    confidence: float
    reason_code: str | None
    selected_evidence: list[EvidenceItem]


def set_seed(seed: int = DEFAULT_SEED) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_csv_utf8(csv_path: str | Path) -> pd.DataFrame:
    """Read Korean CSV files safely from Excel or normal UTF-8 editors."""
    csv_path = Path(csv_path)
    encodings = ["utf-8-sig", "utf-8", "cp949"]
    last_error: Exception | None = None

    for encoding in encodings:
        try:
            return pd.read_csv(csv_path, encoding=encoding)
        except UnicodeDecodeError as error:
            last_error = error

    raise UnicodeDecodeError(
        "utf-8",
        b"",
        0,
        1,
        f"Cannot read {csv_path}. Try saving the CSV as UTF-8. Last error: {last_error}",
    )


def clamp_score(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def normalize_retrieval_score(raw_score: float | None) -> float:
    if raw_score is None:
        return 0.0

    # FAISS inner product with normalized embeddings is usually cosine similarity.
    # If the value is already 0~1, keep it. If it is -1~1, map it to 0~1.
    score = float(raw_score)
    if score < 0:
        score = (score + 1.0) / 2.0
    return round(clamp_score(score), 4)


def score_source_tier(source_tier: str | None) -> float:
    if not source_tier:
        return 0.5

    normalized = source_tier.strip().lower()
    tier_scores = {
        "공공기관": 1.0,
        "정부기관": 1.0,
        "공식자료": 1.0,
        "팩트체크": 0.95,
        "국제기구": 0.9,
        "학술자료": 0.9,
        "주요언론": 0.8,
        "주요 언론": 0.8,
        "언론": 0.7,
        "보조분석": 0.65,
        "보조 분석": 0.65,
        "일반웹": 0.45,
        "일반 웹": 0.45,
    }

    return tier_scores.get(normalized, 0.5)


def score_temporal_freshness(published_at: str | None, now: datetime | None = None) -> float:
    if not published_at:
        return 0.5

    now = now or datetime.now(timezone.utc)
    date_text = str(published_at).strip()[:10]

    try:
        published = datetime.fromisoformat(date_text).replace(tzinfo=timezone.utc)
    except ValueError:
        return 0.5

    age_days = max((now - published).days, 0)
    if age_days <= 180:
        return 1.0
    if age_days <= 365:
        return 0.85
    if age_days <= 730:
        return 0.7
    return 0.5


def score_evidence_reliability(evidence: EvidenceItem) -> EvidenceItem:
    retrieval_score = normalize_retrieval_score(evidence.score)
    source_tier_score = score_source_tier(evidence.source_tier)
    temporal_score = score_temporal_freshness(evidence.published_at)

    reliability_score = (
        0.4 * retrieval_score
        + 0.4 * source_tier_score
        + 0.2 * temporal_score
    )

    evidence.retrieval_score = round(retrieval_score, 4)
    evidence.source_tier_score = round(source_tier_score, 4)
    evidence.temporal_score = round(temporal_score, 4)
    evidence.reliability_score = round(reliability_score, 4)
    return evidence


def calculate_final_confidence(model_confidence: float, evidence_items: list[EvidenceItem]) -> dict[str, float]:
    if not evidence_items:
        return {
            "model_confidence": round(model_confidence, 4),
            "evidence_reliability_avg": 0.0,
            "final_confidence": 0.0,
        }

    evidence_scores = [
        item.reliability_score if item.reliability_score is not None else score_evidence_reliability(item).reliability_score
        for item in evidence_items
    ]
    evidence_reliability_avg = sum(float(score) for score in evidence_scores) / len(evidence_scores)
    final_confidence = 0.5 * model_confidence + 0.5 * evidence_reliability_avg

    return {
        "model_confidence": round(model_confidence, 4),
        "evidence_reliability_avg": round(evidence_reliability_avg, 4),
        "final_confidence": round(final_confidence, 4),
    }


def split_sentences_with_kss(article_text: str) -> list[str]:
    try:
        import kss

        sentences = kss.split_sentences(article_text)
        return [sentence.strip() for sentence in sentences if sentence.strip()]
    except Exception:
        # Fallback for environments where kss is not installed.
        pattern = r"(?<=[.!?。！？])\s+|(?<=[다요죠함음됨임])\.\s*"
        return [sentence.strip() for sentence in re.split(pattern, article_text) if sentence.strip()]


def is_checkable_claim(sentence: str) -> tuple[bool, str | None]:
    sentence = sentence.strip()

    non_checkable_patterns = [
        "생각한다",
        "전망된다",
        "우려된다",
        "예상된다",
        "가능성이 있다",
        "필요하다",
        "바람직하다",
        "의견이다",
    ]

    factual_markers = [
        "발표",
        "밝혔다",
        "확인",
        "시행",
        "폐지",
        "인상",
        "인하",
        "증가",
        "감소",
        "도입",
        "개정",
        "통과",
        "조사",
        "집계",
        "기록",
        "변경",
        "조정",
        "지원",
    ]

    if len(sentence) < 12:
        return False, "NON_VERIFIABLE_CLAIM"

    if any(pattern in sentence for pattern in non_checkable_patterns):
        return False, "NON_VERIFIABLE_CLAIM"

    if any(marker in sentence for marker in factual_markers):
        return True, None

    if re.search(r"\d|%|원|명|건|년|월|일|억원|조원", sentence):
        return True, None

    return False, "NON_VERIFIABLE_CLAIM"


def extract_claim_candidates(article_text: str, max_claims: int = 5) -> list[ClaimCandidate]:
    sentences = split_sentences_with_kss(article_text)
    candidates: list[ClaimCandidate] = []
    cursor = 0

    for index, sentence in enumerate(sentences):
        start = article_text.find(sentence, cursor)
        if start == -1:
            start = cursor
        end = start + len(sentence)
        cursor = end

        is_checkable, reason = is_checkable_claim(sentence)
        candidates.append(
            ClaimCandidate(
                claim_id=f"claim-{index + 1}",
                text=sentence[:300],
                sentence_index=index,
                start_offset=start,
                end_offset=end,
                is_checkable=is_checkable,
                non_verifiable_reason=reason,
            )
        )

    checkable = [candidate for candidate in candidates if candidate.is_checkable]
    return checkable[:max_claims]


class KoBERTTokenizerForFactGround:
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, max_length: int = DEFAULT_MAX_LENGTH):
        from transformers import AutoTokenizer

        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=False,
        )

    def encode_pair(self, claim: str, evidence: str) -> dict[str, Any]:
        return self.tokenizer(
            evidence,
            claim,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def encode_batch(self, claims: list[str], evidences: list[str]) -> dict[str, Any]:
        return self.tokenizer(
            evidences,
            claims,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )


def load_training_csv(csv_path: str | Path, min_text_length: int = 2) -> pd.DataFrame:
    df = read_csv_utf8(csv_path)
    required = {"claim", "evidence", "label"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Training CSV is missing columns: {sorted(missing)}")

    df = df.dropna(subset=["claim", "evidence", "label"]).copy()
    df["claim"] = df["claim"].astype(str).str.strip()
    df["evidence"] = df["evidence"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip().str.upper()

    df = df[(df["claim"].str.len() >= min_text_length) & (df["evidence"].str.len() >= min_text_length)]
    df = df.drop_duplicates(subset=["claim", "evidence", "label"]).reset_index(drop=True)

    invalid_labels = sorted(set(df["label"]) - set(PROJECT_LABELS))
    if invalid_labels:
        raise ValueError(f"Unknown labels: {invalid_labels}. Use: {PROJECT_LABELS}")

    if len(df) < 5:
        raise ValueError("Training CSV is too small. Add more labeled claim-evidence rows.")

    df["labels"] = df["label"].map(LABEL_TO_ID).astype(int)
    return df


def print_label_distribution(df: pd.DataFrame) -> None:
    print("\n[Label distribution]")
    counts = df["label"].value_counts().reindex(PROJECT_LABELS, fill_value=0)
    for label, count in counts.items():
        print(f"- {label}: {count}")


def split_train_valid_test(
    df: pd.DataFrame,
    valid_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = DEFAULT_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from sklearn.model_selection import train_test_split

    min_count = df["labels"].value_counts().min()
    can_stratify = min_count >= 3 and len(df) >= 20
    stratify = df["labels"] if can_stratify else None

    temp_size = valid_size + test_size
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_size,
        random_state=seed,
        stratify=stratify,
    )

    temp_stratify = temp_df["labels"] if can_stratify and temp_df["labels"].value_counts().min() >= 2 else None
    valid_ratio_inside_temp = valid_size / temp_size
    valid_df, test_df = train_test_split(
        temp_df,
        test_size=1 - valid_ratio_inside_temp,
        random_state=seed,
        stratify=temp_stratify,
    )

    return (
        train_df.reset_index(drop=True),
        valid_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def build_hf_dataset(df: pd.DataFrame, tokenizer, max_length: int):
    from datasets import Dataset

    dataset = Dataset.from_pandas(df[["claim", "evidence", "labels"]], preserve_index=False)

    def tokenize(batch: dict[str, list[str]]) -> dict[str, Any]:
        return tokenizer(
            batch["evidence"],
            batch["claim"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    dataset = dataset.map(tokenize, batched=True)
    dataset = dataset.remove_columns(["claim", "evidence"])
    dataset.set_format("torch")
    return dataset


def make_compute_metrics():
    def compute_metrics(eval_prediction):
        import numpy as np
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_recall_fscore_support,
        )

        logits, labels = eval_prediction
        predictions = np.argmax(logits, axis=-1)
        precision, recall, macro_f1, _ = precision_recall_fscore_support(
            labels,
            predictions,
            average="macro",
            zero_division=0,
        )
        weighted_f1 = f1_score(labels, predictions, average="weighted", zero_division=0)
        return {
            "accuracy": accuracy_score(labels, predictions),
            "macro_precision": precision,
            "macro_recall": recall,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
        }

    return compute_metrics


def save_training_metadata(
    output_dir: str | Path,
    train_csv: str | Path,
    model_name: str,
    max_length: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    label_counts: dict[str, int],
    test_metrics: dict[str, float] | None,
) -> None:
    metadata = {
        "project": "FactGround",
        "task": "claim-evidence verdict classification",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "base_model": model_name,
        "labels": PROJECT_LABELS,
        "train_csv": str(train_csv),
        "max_length": max_length,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
        "label_counts": label_counts,
        "test_metrics": test_metrics,
    }
    output_path = Path(output_dir) / "training_metadata.json"
    output_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def train_kobert_verifier(
    train_csv: str | Path,
    output_dir: str | Path = "kobert_factground_verifier",
    model_name: str = DEFAULT_MODEL_NAME,
    max_length: int = DEFAULT_MAX_LENGTH,
    epochs: int = 5,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    seed: int = DEFAULT_SEED,
) -> None:
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    set_seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_training_csv(train_csv)
    print_label_distribution(df)

    train_df, valid_df, test_df = split_train_valid_test(df, seed=seed)
    print(f"\n[Split size] train={len(train_df)}, valid={len(valid_df)}, test={len(test_df)}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=False,
    )

    train_dataset = build_hf_dataset(train_df, tokenizer, max_length)
    valid_dataset = build_hf_dataset(valid_df, tokenizer, max_length)
    test_dataset = build_hf_dataset(test_df, tokenizer, max_length)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(PROJECT_LABELS),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
        trust_remote_code=True,
    )

    args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        seed=seed,
        report_to="none",
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=make_compute_metrics(),
    )

    trainer.train()

    print("\n[Test metrics]")
    test_metrics = trainer.evaluate(test_dataset, metric_key_prefix="test")
    for key, value in test_metrics.items():
        if isinstance(value, float):
            print(f"- {key}: {value:.4f}")
        else:
            print(f"- {key}: {value}")

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    label_counts = df["label"].value_counts().reindex(PROJECT_LABELS, fill_value=0).to_dict()
    save_training_metadata(
        output_dir=output_dir,
        train_csv=train_csv,
        model_name=model_name,
        max_length=max_length,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
        label_counts={str(k): int(v) for k, v in label_counts.items()},
        test_metrics={k: float(v) for k, v in test_metrics.items() if isinstance(v, (int, float))},
    )

    print(f"\nSaved model and metadata to: {output_dir}")


class FactGroundVerifier:
    def __init__(self, model_path: str | Path, max_length: int = DEFAULT_MAX_LENGTH):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()

    def predict(self, claim: str, evidence: str) -> dict[str, Any]:
        import torch

        encoded = self.tokenizer(
            evidence,
            claim,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        with torch.no_grad():
            outputs = self.model(**encoded)
            probabilities = torch.softmax(outputs.logits, dim=-1)[0]
            label_id = int(torch.argmax(probabilities).item())

        label = self.model.config.id2label[label_id]
        label_scores = {
            self.model.config.id2label[index]: round(float(score), 4)
            for index, score in enumerate(probabilities)
        }
        return {
            "label": label,
            "confidence": round(float(probabilities[label_id].item()), 4),
            "label_scores": label_scores,
        }


class SimpleRAGIndex:
    def __init__(
        self,
        embedding_model_name: str = "jhgan/ko-sroberta-multitask",
    ):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.index = None
        self.documents: list[EvidenceItem] = []

    def build(self, evidence_csv: str | Path) -> None:
        df = read_csv_utf8(evidence_csv)

        required = {"evidence_id", "text", "source_name"}
        missing = required - set(df.columns)

        if missing:
            raise ValueError(
                f"Evidence CSV is missing columns: {sorted(missing)}"
            )

        self.documents = [
            EvidenceItem(
                evidence_id=str(row.evidence_id),
                text=str(row.text)[:500],
                source_name=str(row.source_name),
                source_tier=(
                    None
                    if pd.isna(getattr(row, "source_tier", None))
                    else str(getattr(row, "source_tier", None))
                ),
                url=(
                    None
                    if pd.isna(getattr(row, "url", None))
                    else str(getattr(row, "url", None))
                ),
                published_at=(
                    None
                    if pd.isna(getattr(row, "published_at", None))
                    else str(getattr(row, "published_at", None))
                ),
            )
            for row in df.itertuples(index=False)
        ]

        if not self.documents:
            raise ValueError("Evidence CSV에 검색할 문서가 없습니다.")

        texts = [doc.text for doc in self.documents]

        embeddings = self.embedding_model.encode(
            texts,
            normalize_embeddings=True,
        )
        embeddings = np.asarray(embeddings, dtype=np.float32)

        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

    def search(
        self,
        claim: str,
        top_k: int = 3,
    ) -> list[EvidenceItem]:
        if self.index is None:
            raise RuntimeError("Build the RAG index before search().")

        if not claim.strip():
            raise ValueError("검색할 주장이 비어 있습니다.")

        top_k = min(top_k, len(self.documents))

        query = self.embedding_model.encode(
            [claim],
            normalize_embeddings=True,
        )
        query = np.asarray(query, dtype=np.float32)

        scores, indices = self.index.search(query, top_k)

        results: list[EvidenceItem] = []

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue

            item = self.documents[int(idx)]

            evidence = EvidenceItem(
                evidence_id=item.evidence_id,
                text=item.text,
                source_name=item.source_name,
                source_tier=item.source_tier,
                url=item.url,
                published_at=item.published_at,
                score=round(float(score), 4),
            )

            results.append(score_evidence_reliability(evidence))

        return results


def reason_code_for(label: str, evidence_items: list[EvidenceItem]) -> str | None:
    if label == "UNVERIFIABLE" and not evidence_items:
        return "INSUFFICIENT_EVIDENCE"
    if label == "UNVERIFIABLE":
        return "NON_VERIFIABLE_CLAIM"
    if label == "MIXED_CONTEXT_DISTORTED":
        return "CONFLICTING_EVIDENCE"
    return None


def run_factground_pipeline(
    article_text: str,
    rag_index: SimpleRAGIndex,
    verifier: FactGroundVerifier,
    top_k: int = 3,
) -> dict[str, Any]:
    claims = extract_claim_candidates(article_text)
    verdicts: list[dict[str, Any]] = []

    for claim in claims:
        evidence_items = rag_index.search(claim.text, top_k=top_k)
        if not evidence_items:
            verdicts.append(
                {
                    **asdict(
                        ClaimVerdict(
                            claim=claim.text,
                            label="UNVERIFIABLE",
                            confidence=0.0,
                            reason_code="INSUFFICIENT_EVIDENCE",
                            selected_evidence=[],
                        )
                    ),
                    "final_confidence": 0.0,
                    "confidence_breakdown": {
                        "model_confidence": 0.0,
                        "evidence_reliability_avg": 0.0,
                        "final_confidence": 0.0,
                    },
                }
            )
            continue

        combined_evidence = "\n".join(item.text for item in evidence_items)
        prediction = verifier.predict(claim.text, combined_evidence)
        confidence_breakdown = calculate_final_confidence(
            model_confidence=prediction["confidence"],
            evidence_items=evidence_items,
        )
        verdict = ClaimVerdict(
            claim=claim.text,
            label=prediction["label"],
            confidence=prediction["confidence"],
            reason_code=reason_code_for(prediction["label"], evidence_items),
            selected_evidence=evidence_items,
        )
        verdict_dict = asdict(verdict)
        verdict_dict["label_scores"] = prediction["label_scores"]
        verdict_dict["final_confidence"] = confidence_breakdown["final_confidence"]
        verdict_dict["confidence_breakdown"] = confidence_breakdown
        verdicts.append(verdict_dict)

    return {
        "claims": [asdict(claim) for claim in claims],
        "claim_verdicts": verdicts,
        "confidence_policy": {
            "evidence_reliability_score": "0.4 * retrieval_score + 0.4 * source_tier_score + 0.2 * temporal_score",
            "final_confidence": "0.5 * model_confidence + 0.5 * evidence_reliability_avg",
        },
        "model_version": "kobert-factground-v0.2",
        "verdict_policy_version": "v0.2-dev",
    }


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    if len(sys.argv) == 1:
        base_dir = Path(__file__).resolve().parent
        train_csv = base_dir / "sample_train.csv"
        output_dir = base_dir / "kobert_factground_verifier"

        print("명령어 없이 실행되어 기본 학습 모드로 시작합니다.")
        print(f"학습 CSV: {train_csv}")
        print(f"모델 저장 폴더: {output_dir}")
        print("설정: epochs=5, batch_size=2, max_length=256, learning_rate=2e-5")
        print()

        train_kobert_verifier(
            train_csv=train_csv,
            output_dir=output_dir,
            epochs=5,
            batch_size=2,
            max_length=256,
            learning_rate=2e-5,
        )
        return

    parser = argparse.ArgumentParser(description="FactGround KoBERT modeling utilities")
    subcommands = parser.add_subparsers(dest="command", required=True)

    train_cmd = subcommands.add_parser("train", help="Train KoBERT verifier from labeled CSV")
    train_cmd.add_argument("--train-csv", required=True)
    train_cmd.add_argument("--output-dir", default="kobert_factground_verifier")
    train_cmd.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    train_cmd.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    train_cmd.add_argument("--epochs", type=int, default=5)
    train_cmd.add_argument("--batch-size", type=int, default=8)
    train_cmd.add_argument("--learning-rate", type=float, default=2e-5)
    train_cmd.add_argument("--seed", type=int, default=DEFAULT_SEED)

    split_cmd = subcommands.add_parser("split-claims", help="Run KSS sentence split and claim filtering")
    split_cmd.add_argument("--article-text", required=True)
    split_cmd.add_argument("--max-claims", type=int, default=5)

    inspect_cmd = subcommands.add_parser("inspect-csv", help="Validate CSV and show label distribution")
    inspect_cmd.add_argument("--train-csv", required=True)

    args = parser.parse_args()

    if args.command == "train":
        train_kobert_verifier(
            train_csv=args.train_csv,
            output_dir=args.output_dir,
            model_name=args.model_name,
            max_length=args.max_length,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
        )

    if args.command == "split-claims":
        claims = extract_claim_candidates(args.article_text, max_claims=args.max_claims)
        print(json.dumps([asdict(claim) for claim in claims], ensure_ascii=False, indent=2))

    if args.command == "inspect-csv":
        df = load_training_csv(args.train_csv)
        print(f"rows={len(df)}")
        print_label_distribution(df)
        print("\n[Sample]")
        print(df[["claim", "evidence", "label"]].head().to_string(index=False))


if __name__ == "__main__":
    main()