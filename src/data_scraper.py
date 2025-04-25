#!/usr/bin/env python3
"""Wikipedia scraper for GenTutor project."""
import argparse, os, pandas as pd, tqdm, wikipediaapi
from datetime import datetime, timezone

EXCLUDE = {'See also','References','Bibliography','External links',
           'Explanatory notes','Further reading'}

def scrape_page(wiki, title):
    page = wiki.page(title)
    if not page.exists():
        print(f"[WARN] Missing page: {title}")
        return []
    rows=[]
    def walk(sec, path=""):
        full = f"{path} -> {sec.title}" if path else sec.title
        if sec.title in EXCLUDE: 
            return
        if not sec.sections:
            rows.append({"section": full, "text": sec.text})
        else:
            for sub in sec.sections:
                walk(sub, full)
    for top in page.sections:
        walk(top)
    return rows

def main(pages, out_root):
    wiki = wikipediaapi.Wikipedia(
        "GenTutorBot/0.1 (https://example.com)", "en",
        extract_format=wikipediaapi.ExtractFormat.WIKI)
    raw_dir  = os.path.join(out_root, "raw")
    proc_dir = os.path.join(out_root, "processed")
    os.makedirs(raw_dir,  exist_ok=True)
    os.makedirs(proc_dir, exist_ok=True)

    combined=[]
    for title in tqdm.tqdm(pages):
        rows = scrape_page(wiki, title.strip())
        if rows:
            pd.DataFrame(rows).to_csv(f"{raw_dir}/{title}.csv", index=False)
            combined.extend(rows)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    combined_df = pd.DataFrame(combined)
    combined_df.to_csv(f"{proc_dir}/combined_data.csv", index=False)
    try:
        combined_df.to_markdown(f"{proc_dir}/combined_data.md",
                            index=False, tablefmt="pipe")
    except ImportError:
        print("[WARN] 'tabulate' not installed – skipped Markdown export.")
    print("[OK] Scrape complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", type=str, default="pages.txt",
                        help="file with newline‑separated wikipedia titles")
    parser.add_argument("--out", type=str, default="data",
                        help="output root folder")
    args = parser.parse_args()
    pages = open(args.pages).read().splitlines()
    main(pages, args.out)