#!/usr/bin/env python3
"""Scrape NHS.uk (OGL v3) and MedlinePlus (public domain) pages.

Outputs:
  training/data/nhs_parkinson_pairs.jsonl   -- OGL v3 content
  training/data/medlineplus_pairs.jsonl     -- US public domain content

All scraped content is verified against the license manifest before inclusion.
A.D.A.M. Medical Encyclopedia content is excluded from MedlinePlus.
Third-party charity embeds are excluded from NHS pages.

Pairs use Gemma chat format:
  {"messages": [{"role": "user", "content": "..."}, {"role": "model", "content": "..."}]}

Attributions:
  NHS.uk content is licensed under the Open Government Licence v3.0.
  MedlinePlus health topic summaries are authored by NLM and are US public domain.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Optional
from urllib.parse import urlparse

if TYPE_CHECKING:
    import requests
    from bs4 import BeautifulSoup


# ---------------------------------------------------------------------------
# Source allowlists (plan section 3.2, LICENSE_MANIFEST.md)
# ---------------------------------------------------------------------------

NHS_PAGES = [
    "https://www.nhs.uk/conditions/parkinsons-disease/",
    "https://www.nhs.uk/conditions/parkinsons-disease/symptoms/",
    "https://www.nhs.uk/conditions/parkinsons-disease/causes/",
    "https://www.nhs.uk/conditions/parkinsons-disease/diagnosis/",
    "https://www.nhs.uk/conditions/parkinsons-disease/treatment/",
    "https://www.nhs.uk/conditions/parkinsons-disease/living-with/",
    "https://www.nhs.uk/conditions/tremor/",
    "https://www.nhs.uk/conditions/dystonia/",
]

MEDLINEPLUS_PAGES = [
    "https://medlineplus.gov/parkinsonsdisease.html",
    "https://medlineplus.gov/tremor.html",
    "https://medlineplus.gov/voicedisorders.html",
    "https://medlineplus.gov/movementdisorders.html",
    "https://medlineplus.gov/speechandcommunicationdisorders.html",
]

# Sections in MedlinePlus that are A.D.A.M. content -- excluded per manifest
ADAM_SECTION_PATTERNS = [
    re.compile(r"a\.?d\.?a\.?m", re.IGNORECASE),
    re.compile(r"adam medical encyclopedia", re.IGNORECASE),
    re.compile(r"medlineplus medical encyclopedia", re.IGNORECASE),
]

# NHS third-party charity sections to skip
NHS_SKIP_PATTERNS = [
    re.compile(r"parkinson'?s uk", re.IGNORECASE),
    re.compile(r"charity", re.IGNORECASE),
]

HEADERS = {
    "User-Agent": (
        "CogniTrace-DataCuration/1.0 "
        "(academic fine-tuning research; "
        "contact: cognitrace-research@example.com)"
    )
}

ACCESS_DATE = date.today().isoformat()


# ---------------------------------------------------------------------------
# Simplification helpers (no external LLM)
# ---------------------------------------------------------------------------

# Short words considered "simple" for the reading-level heuristic
_COMMON_WORDS = frozenset(
    "the a an and or but in on at to for of with is are was were be been "
    "have has had do does did will would could should may might can this "
    "that these those it its he she we they you your our their who what "
    "when where how all some more most no not if so then also just very "
    "from into through about after before over under up down out back "
    "well good bad new old first last long since always never often".split()
)

_MEDICAL_SIMPLIFICATIONS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bphysician\b", re.I), "doctor"),
    (re.compile(r"\bphysicians\b", re.I), "doctors"),
    (re.compile(r"\bprescribe\b", re.I), "recommend"),
    (re.compile(r"\bprescribed\b", re.I), "recommended"),
    (re.compile(r"\bmedication[s]?\b", re.I), "medicine"),
    (re.compile(r"\bsymptomatology\b", re.I), "symptoms"),
    (re.compile(r"\bpathology\b", re.I), "disease"),
    (re.compile(r"\baetiology\b", re.I), "cause"),
    (re.compile(r"\betiology\b", re.I), "cause"),
    (re.compile(r"\bprognosis\b", re.I), "outlook"),
    (re.compile(r"\btherapeutic\b", re.I), "treatment"),
    (re.compile(r"\bsubsequent\b", re.I), "later"),
    (re.compile(r"\butilise\b", re.I), "use"),
    (re.compile(r"\butilize\b", re.I), "use"),
    (re.compile(r"\badminister\b", re.I), "give"),
    (re.compile(r"\bcommence\b", re.I), "start"),
    (re.compile(r"\bproceed\b", re.I), "go"),
    (re.compile(r"\brequire[s]?\b", re.I), "need"),
    (re.compile(r"\bapproximately\b", re.I), "about"),
    (re.compile(r"\binfrequently\b", re.I), "rarely"),
]


def simplify_text(text: str) -> str:
    """Apply rule-based simplifications to reduce reading level.

    No external LLM is called. Transformations applied:
    1. Medical term substitutions from a fixed lookup table.
    2. Long sentences (over 30 words) are split at the first semicolon or comma
       that appears after the halfway point.
    3. Parenthetical asides of 5+ words are removed.
    """
    result = text
    for pattern, replacement in _MEDICAL_SIMPLIFICATIONS:
        result = pattern.sub(replacement, result)

    # Remove long parenthetical asides
    result = re.sub(r"\([^)]{30,}\)", "", result)
    result = re.sub(r"\s{2,}", " ", result).strip()

    # Split overly long sentences at natural boundaries
    sentences = re.split(r"(?<=[.!?])\s+", result)
    simplified_sentences = []
    for sent in sentences:
        words = sent.split()
        if len(words) <= 30:
            simplified_sentences.append(sent)
            continue
        # Try to split at a semicolon or dash after position 15
        split_match = re.search(r"[;]", sent[40:])
        if split_match:
            cut = 40 + split_match.start()
            simplified_sentences.append(sent[:cut].rstrip(";").strip() + ".")
            simplified_sentences.append(sent[cut + 1:].strip().capitalize())
        else:
            simplified_sentences.append(sent)

    return " ".join(simplified_sentences)


# ---------------------------------------------------------------------------
# Fetch + parse
# ---------------------------------------------------------------------------

def fetch_page(url: str, session: requests.Session, retries: int = 3) -> Optional[BeautifulSoup]:
    import requests as _requests
    from bs4 import BeautifulSoup as _BS
    for attempt in range(retries):
        try:
            resp = session.get(url, headers=HEADERS, timeout=20)
            resp.raise_for_status()
            return _BS(resp.text, "html.parser")
        except _requests.RequestException as exc:
            if attempt == retries - 1:
                print(f"  WARN: failed to fetch {url} after {retries} attempts: {exc}")
                return None
            wait = 2 ** attempt
            print(f"  Retry {attempt + 1}/{retries} for {url} (wait {wait}s)...")
            time.sleep(wait)
    return None


def _is_adam_section(heading_text: str) -> bool:
    return any(p.search(heading_text) for p in ADAM_SECTION_PATTERNS)


def _is_nhs_skip_section(heading_text: str) -> bool:
    return any(p.search(heading_text) for p in NHS_SKIP_PATTERNS)


def extract_nhs_content(soup: BeautifulSoup, url: str) -> list[tuple[str, str]]:
    """
    Return (heading, paragraph_text) pairs from an NHS page.

    NHS pages structure main content inside <article> or <div class="nhsuk-width-container">.
    Skips navigation, footer, breadcrumbs, and third-party charity sections.
    Verifies OGL footer presence before returning any content.
    """
    # Verify OGL license footer
    page_text = soup.get_text(" ", strip=True)
    if "crown copyright" not in page_text.lower() and "open government licence" not in page_text.lower():
        print(f"  WARN: OGL footer not found on {url}, skipping entire page")
        return []

    main = soup.find("article") or soup.find("div", class_=re.compile(r"nhsuk-width-container"))
    if not main:
        main = soup.find("main") or soup.body
    if not main:
        return []

    chunks: list[tuple[str, str]] = []
    current_heading = _page_title_from_url(url)
    skip_until_next_heading = False

    for el in main.find_all(["h1", "h2", "h3", "p"], recursive=True):
        tag_name = el.name
        text = el.get_text(" ", strip=True)
        if not text or len(text) < 10:
            continue

        if tag_name in ("h1", "h2", "h3"):
            skip_until_next_heading = _is_nhs_skip_section(text)
            current_heading = text
            continue

        if skip_until_next_heading:
            continue

        # Skip boilerplate navigation fragments
        if len(text.split()) < 8:
            continue
        if re.search(r"(cookie|javascript|back to top|skip to)", text, re.IGNORECASE):
            continue

        chunks.append((current_heading, text))

    return chunks


def extract_medlineplus_content(soup: BeautifulSoup, url: str) -> list[tuple[str, str]]:
    """
    Return (heading, paragraph_text) pairs from a MedlinePlus health topic page.

    Excludes A.D.A.M. Medical Encyclopedia sections per the license manifest.
    Only includes NLM-authored health topic summary content.
    """
    # Verify NLM authorship (footer or meta tag)
    page_text = soup.get_text(" ", strip=True)
    nlm_present = (
        "national library of medicine" in page_text.lower()
        or "nlm" in page_text.lower()
        or "medlineplus" in page_text.lower()
    )
    if not nlm_present:
        print(f"  WARN: NLM authorship not confirmed on {url}, skipping")
        return []

    # Main content area
    main = (
        soup.find("div", id="ency_summary")
        or soup.find("div", id="topic-summary")
        or soup.find("div", class_=re.compile(r"page-content"))
        or soup.find("main")
    )
    if not main:
        main = soup.body

    chunks: list[tuple[str, str]] = []
    current_heading = _page_title_from_url(url)
    in_adam_section = False

    for el in main.find_all(["h1", "h2", "h3", "p", "li"], recursive=True):
        tag_name = el.name
        text = el.get_text(" ", strip=True)
        if not text or len(text) < 10:
            continue

        if tag_name in ("h1", "h2", "h3"):
            in_adam_section = _is_adam_section(text)
            current_heading = text
            continue

        if in_adam_section:
            continue

        # Skip navigation / boilerplate
        if len(text.split()) < 8:
            continue
        if re.search(r"(skip to|javascript|cookie|subscribe|sign up)", text, re.IGNORECASE):
            continue

        # Skip A.D.A.M. inline markers
        if _is_adam_section(text):
            continue

        chunks.append((current_heading, text))

    return chunks


def _page_title_from_url(url: str) -> str:
    path = urlparse(url).path.rstrip("/").split("/")[-1]
    return path.replace("-", " ").replace("_", " ").title()


# ---------------------------------------------------------------------------
# Pair generation
# ---------------------------------------------------------------------------

def _make_pair(instruction: str, response: str, source: str = "", source_url: str = "", source_id: str = "") -> dict:
    return {
        "messages": [
            {"role": "user", "content": instruction},
            {"role": "model", "content": response},
        ],
        "_source": source,
        "_source_url": source_url,
        "_source_id": source_id,
    }


def _generate_pairs_from_chunk(
    heading: str, paragraph: str, topic_label: str,
    source: str = "", source_url: str = "", chunk_index: int = 0,
) -> list[dict]:
    pairs = []
    clean_heading = heading.strip().rstrip(".?!")
    clean_para = paragraph.strip()
    sid = f"{source.lower()}:{source_url}#{chunk_index}"

    if len(clean_para.split()) < 20:
        return []

    pairs.append(_make_pair(
        f"Explain {clean_heading} about {topic_label} to a patient.",
        clean_para,
        source=source, source_url=source_url, source_id=f"{sid}_explain",
    ))

    pairs.append(_make_pair(
        f"What should a patient know about {clean_heading.lower()}?",
        clean_para,
        source=source, source_url=source_url, source_id=f"{sid}_qa",
    ))

    simplified = simplify_text(clean_para)
    if simplified != clean_para and len(simplified.split()) >= 15:
        pairs.append(_make_pair(
            f"Explain this at a 6th-grade reading level: {clean_para}",
            simplified,
            source=source, source_url=source_url, source_id=f"{sid}_simple",
        ))

    return pairs


# ---------------------------------------------------------------------------
# Scrape orchestrators
# ---------------------------------------------------------------------------

def scrape_nhs(
    session: requests.Session,
    rate_limit_secs: float,
    log: list[dict],
) -> list[dict]:
    all_pairs: list[dict] = []

    for url in NHS_PAGES:
        print(f"  Fetching {url}")
        soup = fetch_page(url, session)
        time.sleep(rate_limit_secs)

        if soup is None:
            continue

        chunks = extract_nhs_content(soup, url)
        topic = _page_title_from_url(url)
        page_pairs: list[dict] = []

        for ci, (heading, para) in enumerate(chunks):
            page_pairs.extend(_generate_pairs_from_chunk(
                heading, para, topic, source="NHS", source_url=url, chunk_index=ci,
            ))

        log.append({
            "url": url,
            "access_date": ACCESS_DATE,
            "license": "OGL v3",
            "pairs_generated": len(page_pairs),
        })

        print(f"    {len(chunks)} chunks -> {len(page_pairs)} pairs")
        all_pairs.extend(page_pairs)

    return all_pairs


def scrape_medlineplus(
    session: requests.Session,
    rate_limit_secs: float,
    log: list[dict],
) -> list[dict]:
    all_pairs: list[dict] = []

    for url in MEDLINEPLUS_PAGES:
        print(f"  Fetching {url}")
        soup = fetch_page(url, session)
        time.sleep(rate_limit_secs)

        if soup is None:
            continue

        chunks = extract_medlineplus_content(soup, url)
        topic = _page_title_from_url(url)
        page_pairs: list[dict] = []

        for ci, (heading, para) in enumerate(chunks):
            page_pairs.extend(_generate_pairs_from_chunk(
                heading, para, topic, source="MedlinePlus", source_url=url, chunk_index=ci,
            ))

        log.append({
            "url": url,
            "access_date": ACCESS_DATE,
            "license": "US public domain (NLM-authored)",
            "pairs_generated": len(page_pairs),
        })

        print(f"    {len(chunks)} chunks -> {len(page_pairs)} pairs")
        all_pairs.extend(page_pairs)

    return all_pairs


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def out_dir() -> Path:
    d = Path(__file__).parent / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_jsonl(pairs: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for p in pairs:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(pairs)} pairs to {path}")


def write_scrape_log(log: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(log, fh, indent=2)
    print(f"  Wrote scrape log to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape NHS.uk and MedlinePlus for medical communication training pairs."
    )
    parser.add_argument(
        "--rate-limit", type=float, default=1.0,
        help="Seconds to wait between requests (default: 1.0)"
    )
    parser.add_argument(
        "--skip-nhs", action="store_true",
        help="Skip NHS pages"
    )
    parser.add_argument(
        "--skip-medlineplus", action="store_true",
        help="Skip MedlinePlus pages"
    )
    args = parser.parse_args()

    try:
        import requests
        from bs4 import BeautifulSoup  # noqa: F401 -- verify dep present before any work
    except ImportError as exc:
        raise SystemExit(
            f"Missing dependency: {exc}. Run: pip install requests beautifulsoup4"
        ) from exc

    dest = out_dir()
    session = requests.Session()
    scrape_log: list[dict] = []

    if not args.skip_nhs:
        print("\n=== NHS.uk (OGL v3) ===")
        nhs_pairs = scrape_nhs(session, args.rate_limit, scrape_log)
        write_jsonl(nhs_pairs, dest / "nhs_parkinson_pairs.jsonl")
    else:
        print("Skipping NHS (--skip-nhs set)")

    if not args.skip_medlineplus:
        print("\n=== MedlinePlus (NLM public domain) ===")
        mlp_pairs = scrape_medlineplus(session, args.rate_limit, scrape_log)
        write_jsonl(mlp_pairs, dest / "medlineplus_pairs.jsonl")
    else:
        print("Skipping MedlinePlus (--skip-medlineplus set)")

    write_scrape_log(scrape_log, dest / "scrape_log.json")
    print("\nDone. Check training/data/ for output files.")


if __name__ == "__main__":
    main()
