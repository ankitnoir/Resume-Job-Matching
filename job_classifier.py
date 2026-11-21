"""
job_classifier.py
=================
Supervised machine-learning model that predicts a JOB CATEGORY from resume text.

Model: TF-IDF (1-2 grams) + Logistic Regression (default) or Complement Naive Bayes,
built as a scikit-learn Pipeline. It is trained on YOUR labelled dataset; nothing is
hard-coded. The output is a probability per category. It is a CLASSIFICATION result and
is not a similarity score or a skill match.

Training data format (CSV or DataFrame)
---------------------------------------
Two columns: the resume text and its category label, e.g.
    Resume_str , Category
    "Python developer with ..." , "Information-Technology"
Column names are configurable (--text-col / --label-col).

Command line training
---------------------
    python job_classifier.py --data data/Resume.csv --text-col Resume_str --label-col Category
    (saves to <this folder>/models/job_classifier.joblib; override with --output)

Python API
----------
clf = JobClassifier()
metrics = clf.train(df, text_col="resume_text", label_col="category")  -> dict
clf.save(DEFAULT_MODEL_PATH)
clf = JobClassifier.load(DEFAULT_MODEL_PATH)      # or load_classifier()
clf.predict(text, top_k=3)        -> [{"category": str, "probability": float}, ...]
clf.predict_proba_dict(text)      -> {"category": probability, ...}
clf.predict_labels(list_of_texts) -> [top-1 category or None, ...]

Security note: the model file is a joblib (pickle) file. Only load files you created yourself.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import joblib
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline

from text_preprocessing import preprocess_batch, preprocess_text

logger = logging.getLogger(__name__)

SUPPORTED_MODELS = ("logreg", "complement_nb")

# Absolute default location (next to this file) so training and the Streamlit app always agree,
# whatever the current working directory is.
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "models" / "job_classifier.joblib"


def load_dataset(csv_path: Union[str, Path], text_col: str = "Resume_str",
                 label_col: str = "Category") -> pd.DataFrame:
    """Read a labelled CSV and return a DataFrame with columns ['resume_text', 'category']."""
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    try:
        df = pd.read_csv(path)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1")
    missing = [c for c in (text_col, label_col) if c not in df.columns]
    if missing:
        raise ValueError(f"Column(s) {missing} not found. Available columns: {list(df.columns)}")
    return df[[text_col, label_col]].rename(columns={text_col: "resume_text", label_col: "category"})


class JobClassifier:
    """TF-IDF + scikit-learn classifier for job-category prediction."""

    def __init__(self, model_type: str = "logreg", max_features: int = 50000,
                 ngram_range: tuple = (1, 2), min_df: int = 2, random_state: int = 42) -> None:
        if model_type not in SUPPORTED_MODELS:
            raise ValueError(f"model_type must be one of {SUPPORTED_MODELS}, got '{model_type}'")
        self.model_type = model_type
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.random_state = random_state
        self.pipeline: Optional[Pipeline] = None
        self.metadata: Dict[str, Any] = {}

    # ------------------------------------------------------------------ training --
    def _build_pipeline(self) -> Pipeline:
        vectorizer = TfidfVectorizer(
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            min_df=self.min_df,
            sublinear_tf=True,  # dampens the effect of very frequent words
        )
        if self.model_type == "logreg":
            estimator = LogisticRegression(
                max_iter=2000, C=5.0, class_weight="balanced",  # balanced: handles uneven classes
                random_state=self.random_state,
            )
        else:
            estimator = ComplementNB(alpha=0.3)
        return Pipeline([("tfidf", vectorizer), ("clf", estimator)])

    @staticmethod
    def _validate_frame(df: pd.DataFrame, text_col: str, label_col: str) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame.")
        missing = [c for c in (text_col, label_col) if c not in df.columns]
        if missing:
            raise ValueError(f"Column(s) {missing} not in DataFrame. Available: {list(df.columns)}")
        data = df[[text_col, label_col]].dropna().copy()
        data[text_col] = data[text_col].astype(str)
        data[label_col] = data[label_col].astype(str).str.strip()
        data = data[(data[text_col].str.strip() != "") & (data[label_col] != "")]
        if data.empty:
            raise ValueError("No valid rows left after removing empty texts/labels.")
        return data

    def train(self, df: pd.DataFrame, text_col: str = "resume_text", label_col: str = "category",
              test_size: float = 0.2, refit_on_full_data: bool = False) -> Dict[str, Any]:
        """
        Train the model and evaluate it on a stratified hold-out split.

        Parameters
        ----------
        refit_on_full_data : after evaluating, refit on ALL rows (usually slightly better for
            deployment; the reported metrics still come from the hold-out evaluation).

        Returns
        -------
        dict with accuracy, macro_f1, weighted_f1, n_train, n_test, n_classes, classes,
        per_class_report and confusion_matrix.
        """
        data = self._validate_frame(df, text_col, label_col)

        logger.info("Preprocessing %d documents ...", len(data))
        texts = preprocess_batch(data[text_col].tolist())
        labels = data[label_col].tolist()

        # Remove documents that became empty after preprocessing.
        keep = [i for i, t in enumerate(texts) if t.strip()]
        texts = [texts[i] for i in keep]
        labels = [labels[i] for i in keep]

        # Stratified splitting needs >= 2 samples per class.
        counts = Counter(labels)
        rare = {c for c, n in counts.items() if n < 2}
        if rare:
            logger.warning("Dropping classes with fewer than 2 samples: %s", sorted(rare))
            pairs = [(t, l) for t, l in zip(texts, labels) if l not in rare]
            texts, labels = [p[0] for p in pairs], [p[1] for p in pairs]
        if len(set(labels)) < 2:
            raise ValueError("At least two categories with >= 2 samples each are required.")

        try:
            x_train, x_test, y_train, y_test = train_test_split(
                texts, labels, test_size=test_size, random_state=self.random_state, stratify=labels
            )
        except ValueError as exc:
            logger.warning("Stratified split failed (%s); using a random split instead.", exc)
            x_train, x_test, y_train, y_test = train_test_split(
                texts, labels, test_size=test_size, random_state=self.random_state
            )

        self.pipeline = self._build_pipeline()
        self.pipeline.fit(x_train, y_train)
        y_pred = self.pipeline.predict(x_test)

        label_order = sorted(set(y_test) | set(y_pred))
        metrics: Dict[str, Any] = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
            "n_train": len(x_train),
            "n_test": len(x_test),
            "n_classes": len(set(labels)),
            "classes": sorted(set(labels)),
            "per_class_report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
            "confusion_matrix": {
                "labels": label_order,
                "matrix": confusion_matrix(y_test, y_pred, labels=label_order).tolist(),
            },
        }

        if refit_on_full_data:
            self.pipeline = self._build_pipeline().fit(texts, labels)

        self.metadata = {
            "model_type": self.model_type,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "sklearn_version": sklearn.__version__,
            "n_documents": len(texts),
            "refit_on_full_data": refit_on_full_data,
            "classes": metrics["classes"],
            "holdout_accuracy": metrics["accuracy"],
            "holdout_macro_f1": metrics["macro_f1"],
        }
        return metrics

    # ---------------------------------------------------------------- prediction --
    @property
    def classes_(self) -> List[str]:
        self._require_trained()
        return list(self.pipeline.named_steps["clf"].classes_)

    def _require_trained(self) -> None:
        if self.pipeline is None:
            raise RuntimeError("The model is not trained or loaded. Call train() or load() first.")

    def _prepare(self, text: Optional[str]) -> str:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Input text is empty.")
        prepared = preprocess_text(text)
        if not prepared.strip():
            raise ValueError("No usable words remain after preprocessing the input text.")
        return prepared

    def predict_proba_dict(self, text: str) -> Dict[str, float]:
        """Return {category: probability} for one document (probabilities sum to 1)."""
        self._require_trained()
        probs = self.pipeline.predict_proba([self._prepare(text)])[0]
        return {str(c): float(p) for c, p in zip(self.classes_, probs)}

    def predict(self, text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Return the `top_k` most likely categories as [{'category', 'probability'}, ...]."""
        probs = self.predict_proba_dict(text)
        ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[: max(1, top_k)]
        return [{"category": c, "probability": round(p, 4)} for c, p in ranked]

    def predict_labels(self, texts: Sequence[str]) -> List[Optional[str]]:
        """Top-1 category for many documents (None for empty documents). Efficient batch version."""
        self._require_trained()
        prepared = preprocess_batch(list(texts))
        valid = [i for i, t in enumerate(prepared) if t.strip()]
        results: List[Optional[str]] = [None] * len(prepared)
        if valid:
            predictions = self.pipeline.predict([prepared[i] for i in valid])
            for i, label in zip(valid, predictions):
                results[i] = str(label)
        return results

    # ----------------------------------------------------------- save and load --
    def save(self, path: Union[str, Path]) -> Path:
        self._require_trained()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": self.pipeline, "metadata": self.metadata,
                     "config": {"model_type": self.model_type}}, path)
        return path

    @classmethod
    def load(cls, path: Union[str, Path]) -> "JobClassifier":
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Model file not found: {path}. Train one first with: "
                "python job_classifier.py --data <csv> --text-col <col> --label-col <col>"
            )
        try:
            payload = joblib.load(path)
            pipeline, metadata = payload["pipeline"], payload.get("metadata", {})
        except Exception as exc:
            raise ValueError(f"Could not load model from '{path}': {exc}") from exc

        saved_version = metadata.get("sklearn_version")
        if saved_version and saved_version != sklearn.__version__:
            logger.warning("Model was trained with scikit-learn %s but %s is installed; "
                           "retrain if you see errors.", saved_version, sklearn.__version__)
        instance = cls(model_type=metadata.get("model_type", "logreg"))
        instance.pipeline = pipeline
        instance.metadata = metadata
        return instance


# --------------------------------------------------------------------------- #
# Convenience functions
# --------------------------------------------------------------------------- #
def train_model(csv_path: Union[str, Path], text_col: str, label_col: str,
                output_path: Union[str, Path] = DEFAULT_MODEL_PATH,
                model_type: str = "logreg", test_size: float = 0.2,
                refit_on_full_data: bool = False) -> Dict[str, Any]:
    """Load a CSV, train, save the model and a `.metrics.json` file next to it."""
    df = load_dataset(csv_path, text_col, label_col)
    clf = JobClassifier(model_type=model_type)
    metrics = clf.train(df, test_size=test_size, refit_on_full_data=refit_on_full_data)
    saved = clf.save(output_path)
    metrics_path = saved.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    metrics["model_path"] = str(saved)
    metrics["metrics_path"] = str(metrics_path)
    return metrics


def load_classifier(path: Union[str, Path] = DEFAULT_MODEL_PATH) -> JobClassifier:
    """Load a previously trained classifier."""
    return JobClassifier.load(path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Train the job-category classifier.")
    parser.add_argument("--data", required=True, help="Path to the labelled CSV file")
    parser.add_argument("--text-col", default="Resume_str", help="Column with resume text")
    parser.add_argument("--label-col", default="Category", help="Column with category labels")
    parser.add_argument("--output", default=str(DEFAULT_MODEL_PATH), help="Where to save the model")
    parser.add_argument("--model-type", default="logreg", choices=SUPPORTED_MODELS)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--refit-on-full-data", action="store_true",
                        help="Refit on all data after hold-out evaluation")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        metrics = train_model(args.data, args.text_col, args.label_col, args.output,
                              args.model_type, args.test_size, args.refit_on_full_data)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"\nTrained on {metrics['n_train']} documents, evaluated on {metrics['n_test']} "
          f"({metrics['n_classes']} classes)")
    print(f"Hold-out accuracy : {metrics['accuracy']:.4f}")
    print(f"Hold-out macro F1 : {metrics['macro_f1']:.4f}")
    print(f"Model saved to    : {metrics['model_path']}")
    print(f"Metrics saved to  : {metrics['metrics_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
