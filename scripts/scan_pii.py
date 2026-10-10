"""Scan prompts and outputs for possible personal information, and write a PII-free 50-row sample.

Detectors (applied to LLM_Input and LLM_output separately):
    email          anything shaped like name@domain.tld
    email_real     emails whose domain is not an obvious placeholder (example.com, domain.com, ...)
    phone          US-style phone numbers (123-456-7890, (123) 456-7890, +1 123 456 7890) and +CC international numbers
    url            any http(s):// or www. link
    personal_url   links whose path looks personal: social-media profiles, /users/, /home/, /~name, file:///

Usage (from the project root):
    python scripts/scan_pii.py

Outputs:
    results/pii_scan.csv      counts per detector and column
    data/sample.csv           10 random rows per family (seed 448) with NO detector hit in input or output
"""

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]  # project folder (parent of scripts/)
sys.path.insert(0, str(ROOT))  # so "from src... import" works when run as a script
from src.data import load_data  # noqa: E402

SEED = 448
PER_FAMILY = 10
EMAIL = re.compile(r"[\w.+-]+@([\w-]+\.)+[a-zA-Z]{2,}")
PLACEHOLDER_DOMAIN = re.compile(r"@(example|domain|yourdomain|email|company|test|mail|sample)\.", re.IGNORECASE)
PHONE = re.compile(r"(?<!\d)(\+1[\s.-]?)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}(?!\d)|(?<![\w+])\+\d{2,3}[\s.-]?\d{2,4}[\s.-]?\d{3,4}[\s.-]?\d{3,4}(?!\d)")
URL = re.compile(r"(https?://|www\.)\S+", re.IGNORECASE)
PERSONAL_URL = re.compile(
    # (?<![\w.-]) makes the domain start at a boundary, so "javafx.com/" or "netflix.com/" is not "x.com/".
    r"((?<![\w.-])(linkedin\.com/in/|facebook\.com/(?!share|sharer|groups)\w|instagram\.com/\w|"
    r"twitter\.com/\w|x\.com/\w|tiktok\.com/@)|/users?/\w|/home/\w|/~\w|file:///)",
    re.IGNORECASE)


def real_email(text):
    return any(not PLACEHOLDER_DOMAIN.search(m.group(0)) for m in EMAIL.finditer(text))


DETECTORS = {
    "email": lambda t: bool(EMAIL.search(t)),
    "email_real": real_email,
    "phone": lambda t: bool(PHONE.search(t)),
    "url": lambda t: bool(URL.search(t)),
    "personal_url": lambda t: bool(PERSONAL_URL.search(t)),
}


def main():
    df = load_data()
    flags = pd.DataFrame(index=df.index)
    rows = []
    for column in ["LLM_Input", "LLM_output"]:
        for name, detect in DETECTORS.items():
            flags[f"{column}_{name}"] = df[column].map(detect)
            rows.append({"column": column, "detector": name, "rows_flagged": int(flags[f"{column}_{name}"].sum()),
                         "share": flags[f"{column}_{name}"].mean()})
    scan = pd.DataFrame(rows)
    any_hit = flags.any(axis=1)
    scan.loc[len(scan)] = {"column": "either", "detector": "any", "rows_flagged": int(any_hit.sum()),
                           "share": any_hit.mean()}
    scan.round(4).to_csv(ROOT / "results" / "pii_scan.csv", index=False)
    print("PII scan (saved to results/pii_scan.csv):")
    print(scan.round(4).to_string(index=False))

    # Show the real-looking hits so they can be checked by eye.
    for column in ["LLM_Input", "LLM_output"]:
        for name in ["email_real", "phone", "personal_url"]:
            hits = df.loc[flags[f"{column}_{name}"], column]
            for text in hits.head(5):
                match = {"email_real": EMAIL, "phone": PHONE, "personal_url": PERSONAL_URL}[name].search(text)
                print(f"  {column} {name}: ...{text[max(0, match.start() - 30):match.end() + 30]!r}...")

    # Sample only from rows with no hit of ANY detector (including plain URLs) in input or output.
    clean = df[~any_hit]
    sample = clean.groupby("LLM_name").sample(PER_FAMILY, random_state=SEED)
    columns = ["LLM_name", "LLM_Input", "LLM_output", "model_version", "prompt_id", "domain", "source_id"]
    (ROOT / "data").mkdir(exist_ok=True)
    sample[columns].to_csv(ROOT / "data" / "sample.csv", index=False)
    print(f"\nSaved data/sample.csv: {len(sample)} rows ({PER_FAMILY} per family), all PII-detector-free")


if __name__ == "__main__":
    main()
