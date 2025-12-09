import os
import time
import re
import urllib.parse
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.rbi.org.in"
LIST_URL = "https://www.rbi.org.in/Scripts/BS_ViewMasterDirections.aspx"
OUTPUT_DIR = "rbi_pdfs"


def get_soup(url):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        )
    }
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def ensure_output_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def normalize_pdf_url(href):
    if not href:
        return None
    full_url = urllib.parse.urljoin(BASE_URL, href)
    if full_url.lower().endswith(".pdf"):
        return full_url
    return None


def clean_filename(text):
    """
    Turn link text into a safe filename.
    """
    text = text.strip()

    # Replace whitespace with single spaces
    text = re.sub(r"\s+", " ", text)

    # Remove characters not allowed in filenames on Windows/Unix
    invalid = r'<>:"/\\|?*'
    for ch in invalid:
        text = text.replace(ch, "_")

    # Truncate very long names
    if len(text) > 150:
        text = text[:150].rstrip()

    # Ensure not empty
    if not text:
        text = "document"

    return text


def find_pdf_links_with_names(soup):
    """
    Return list of (pdf_url, display_name) tuples.
    """
    results = []
    seen = set()

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        pdf_url = normalize_pdf_url(href)
        if not pdf_url:
            continue

        if pdf_url in seen:
            continue

        seen.add(pdf_url)
        link_text = a.get_text(strip=True)

        # If no text, we will fallback later to URL-based name
        results.append((pdf_url, link_text))

    return results


def filename_for_pdf(url, link_text):
    """
    Build final filename using link text as base.
    """
    if link_text:
        base = clean_filename(link_text)
    else:
        # Fallback to basename from URL
        parsed = urllib.parse.urlparse(url)
        base = os.path.basename(parsed.path) or "file"
        base = base.split("?")[0].split("#")[0]
        base = os.path.splitext(base)[0]
        base = clean_filename(base)

    filename = base + ".pdf"
    return filename


def download_file(url, dest_path):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        )
    }

    for attempt in range(3):
        try:
            print(f"Downloading: {url}")
            with requests.get(url, headers=headers, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(dest_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            print(f"Saved to: {dest_path}")
            return True
        except Exception as e:
            print(f"Error downloading {url} (attempt {attempt + 1}/3): {e}")
            time.sleep(2)

    print(f"Failed to download after 3 attempts: {url}")
    return False


def main():
    print(f"Fetching press release page: {LIST_URL}")
    soup = get_soup(LIST_URL)

    print("Extracting PDF links and names...")
    pdf_items = find_pdf_links_with_names(soup)
    print(f"Found {len(pdf_items)} PDF links.")

    if not pdf_items:
        print("No PDF links found. Check if the page requires additional parameters.")
        return

    ensure_output_dir(OUTPUT_DIR)

    for pdf_url, link_text in pdf_items:
        filename = filename_for_pdf(pdf_url, link_text)
        dest_path = os.path.join(OUTPUT_DIR, filename)

        # If same filename already exists, add a numeric suffix to avoid collisions
        original_dest_path = dest_path
        counter = 1
        while os.path.exists(dest_path):
            name, ext = os.path.splitext(original_dest_path)
            dest_path = f"{name}_{counter}{ext}"
            counter += 1

        download_file(pdf_url, dest_path)
        time.sleep(1)

    print("Done.")


if __name__ == "__main__":
    main()
