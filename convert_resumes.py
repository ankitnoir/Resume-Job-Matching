
from pathlib import Path
import fitz
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "data" / "resume_dataset"
OUTPUT = ROOT / "data" / "resumes.csv"

records = []
failed = 0

for category_folder in DATASET.iterdir():
    if not category_folder.is_dir():
        continue

    category = category_folder.name

    for pdf in category_folder.glob("*.pdf"):
        try:
            with fitz.open(pdf) as doc:
                text = "\n".join(page.get_text() for page in doc).strip()

            if len(text) >= 200:
                records.append({
                    "resume_id": pdf.stem,
                    "category": category,
                    "resume_text": text
                })
            else:
                failed += 1

        except Exception:
            failed += 1

df = pd.DataFrame(records)
df.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

print("Conversion complete!")
print("Resumes extracted:", len(df))
print("PDFs skipped:", failed)
print("Saved to:", OUTPUT)
print("Categories:", df["category"].nunique() if not df.empty else 0)
