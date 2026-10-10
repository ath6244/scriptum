"""Build report/report.pdf from report/report.md.

pandoc (bundled with the pypandoc_binary package) turns the Markdown into one standalone HTML file with
the figures and CSS embedded; headless Chrome or Edge then prints that HTML to PDF. No LaTeX needed.

Usage (from the project root):
    python scripts/build_report.py
Set the CHROME environment variable to the browser executable if it is not found automatically.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pypandoc

ROOT = Path(__file__).resolve().parents[1]  # project folder (parent of scripts/)
REPORT = ROOT / "report"


def find_browser():
    """Chrome, Chromium or Edge: $CHROME, then PATH, then the standard macOS and Windows install locations."""
    names = ["chrome", "google-chrome", "chromium", "chromium-browser", "msedge", "microsoft-edge"]
    candidates = [os.environ.get("CHROME")] + [shutil.which(name) for name in names]
    candidates.append("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
    for base in [os.environ.get("PROGRAMFILES"), os.environ.get("PROGRAMFILES(X86)"), os.environ.get("LOCALAPPDATA")]:
        if base:
            candidates += [Path(base) / "Google/Chrome/Application/chrome.exe",
                           Path(base) / "Microsoft/Edge/Application/msedge.exe"]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(candidate)
    raise SystemExit("No Chrome/Edge found; set the CHROME environment variable to the browser executable.")


def main():
    html = REPORT / "report.html"
    pdf = REPORT / "report.pdf"
    # Run pandoc inside report/ so the figure paths (../figures/...) resolve.
    pypandoc.convert_file(
        "report.md", "html5", outputfile="report.html", cworkdir=str(REPORT),
        extra_args=["--standalone", "--embed-resources", "--css=style.css", "--mathml",
                    "--metadata=lang:en"],
    )
    print(f"Wrote {html.relative_to(ROOT)}")
    subprocess.run([find_browser(), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf}", html.as_uri()], check=True, capture_output=True)
    print(f"Wrote {pdf.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
