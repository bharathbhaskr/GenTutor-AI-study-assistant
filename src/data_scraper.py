#!/usr/bin/env python3
"""
Wikipedia scraper for GenTutor: now can pull either a custom pages.txt
OR a built-in list of Generative AI titles—and writes out .md files.
"""
import argparse
import os
import re
from pathlib import Path
from datetime import datetime, timezone

import tqdm
import wikipediaapi

# Sections to skip
EXCLUDE = {
    "See also", "References", "Bibliography", "External links",
    "Explanatory notes", "Further reading"
}

# Built-in list for generative-AI docs
GEN_AI_PAGES = [
    "Generative artificial intelligence",
    "Transformer (machine learning)",
    "ChatGPT",
    "DALL·E",
    "Stable Diffusion",
    "Diffusion model",
    "Prompt engineering",
    "Large language model",
    "OpenAI",
]

def scrape_page(wiki: wikipediaapi.Wikipedia, title: str) -> list[dict]:
    """
    Walk all subsections of a page, return list of {'section': ..., 'text': ...}
    """
    page = wiki.page(title)
    if not page.exists():
        print(f"[WARN] Missing page: {title}")
        return []
    rows: list[dict] = []

    def walk(sec, path=""):
        sec_title = sec.title.strip()
        if sec_title in EXCLUDE:
            return
        full = f"{path} → {sec_title}" if path else sec_title
        if not sec.sections:
            rows.append({"section": full, "text": sec.text.strip()})
        else:
            for sub in sec.sections:
                walk(sub, full)

    for top in page.sections:
        walk(top)
    return rows

def save_markdown(rows: list[dict], title: str, out_dir: Path) -> Path:
    """
    Given scraped rows, write a .md file with headings & text.
    Returns the Path to the file.
    """
    # sanitize filename
    safe = re.sub(r"[^\w\s\-]", "", title).strip().replace(" ", "_")
    md_path = out_dir / f"{safe}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        for row in rows:
            f.write(f"## {row['section']}\n\n")
            f.write(row["text"] + "\n\n")
    return md_path

def main(pages_file: str, out_root: str, use_gen_ai: bool):
    # Initialize Wikipedia API
    wiki = wikipediaapi.Wikipedia(
        user_agent="GenTutorBot/0.1 (https://example.com)",
        language="en",
        extract_format=wikipediaapi.ExtractFormat.WIKI
    )

    out_root = Path(out_root)
    raw_dir = out_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    combined_md_path = out_root / "combined.md"

    # Decide which titles to scrape
    if use_gen_ai:
        titles = GEN_AI_PAGES
    else:
        titles = [line.strip() for line in open(pages_file, encoding="utf-8") if line.strip()]

    all_combined = []
    for title in tqdm.tqdm(titles, desc="Scraping pages"):
        rows = scrape_page(wiki, title)
        if not rows:
            continue
        md_file = save_markdown(rows, title, raw_dir)
        all_combined.append(md_file.read_text(encoding="utf-8"))

    # Write a single combined Markdown
    with open(combined_md_path, "w", encoding="utf-8") as f:
        f.write("\n\n---\n\n".join(all_combined))

    print(f"[OK] Scraped {len(all_combined)} pages.")
    print(f"Raw .md files in: {raw_dir}")
    print(f"Combined MD at: {combined_md_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Wiki→Markdown scraper for GenTutor")
    p.add_argument(
        "--pages", "-p",
        help="Path to newline-separated titles.txt",
        default="pages.txt"
    )
    p.add_argument(
        "--out", "-o",
        help="Output folder root (will create raw/ & combined.md)",
        default="data/gen_ai"
    )
    p.add_argument(
        "--gen-ai", action="store_true",
        help="Ignore pages.txt and scrape built-in Generative AI titles"
    )
    args = p.parse_args()

    main(args.pages, args.out, args.gen_ai)
