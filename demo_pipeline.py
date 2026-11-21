"""
demo_pipeline.py
================
Run the REAL pipeline from the command line (uses the actual Sentence-Transformer model,
so the first run needs internet to download it).

    python demo_pipeline.py --resume my_resume.pdf --job-file job.txt
    python demo_pipeline.py --resume my_resume.pdf --job-text "Looking for a Python developer ..."
    python demo_pipeline.py --resume my_resume.pdf --jobs-csv data/jobs.csv --top-n 5

Optional: --model path/to/job_classifier.joblib (defaults to the standard location; if no trained
model exists the classification component is skipped and its weight is redistributed).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from job_classifier import DEFAULT_MODEL_PATH, JobClassifier
from job_recommendation import JobRecommender, load_jobs, results_to_dataframe
from pdf_extractor import PDFExtractionError, extract_text_from_file
from scoring import evaluate_match
from semantic_matching import SemanticMatcher


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Resume-job matching demo")
    parser.add_argument("--resume", required=True, help="Resume file (.pdf or .txt)")
    parser.add_argument("--job-file", help="Job description file (.pdf or .txt)")
    parser.add_argument("--job-text", help="Job description as text")
    parser.add_argument("--jobs-csv", help="CSV of job postings (columns: title, description, ...)")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="Trained classifier (.joblib)")
    parser.add_argument("--top-n", type=int, default=5)
    args = parser.parse_args(argv)

    if not (args.job_file or args.job_text or args.jobs_csv):
        parser.error("Provide --job-file, --job-text or --jobs-csv")

    try:
        resume_text = extract_text_from_file(args.resume)
        classifier = JobClassifier.load(args.model) if Path(args.model).is_file() else None
        if classifier is None:
            print(f"[info] No classifier at {args.model}; running without category classification.")
        matcher = SemanticMatcher()

        if args.job_file or args.job_text:
            job_text = extract_text_from_file(args.job_file) if args.job_file else args.job_text
            result = evaluate_match(resume_text, job_text, classifier=classifier, matcher=matcher)
            print(json.dumps(result, indent=2, default=str))

        if args.jobs_csv:
            recommender = JobRecommender(load_jobs(args.jobs_csv), classifier=classifier, matcher=matcher)
            recs = recommender.recommend(resume_text, top_n=args.top_n)
            print(results_to_dataframe(recs).to_string(index=False))
    except (PDFExtractionError, FileNotFoundError, ValueError, RuntimeError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
