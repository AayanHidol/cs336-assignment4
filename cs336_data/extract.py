"""
Overview:
This code processes a Common Crawl WARC archive — a compressed file format used to store bulk
web crawl data (HTML, headers, metadata). It iterates through HTTP response records, strips 
each page down to plain text (discarding HTML tags), and saves each result to a numbered .txt
file. The key challenge it handles is encoding detection: web pages don't always use UTF-8, 
so it falls back to auto-detecting the charset when naive decoding fails.

"""


import gzip 
from fastwarc.warc import ArchiveIterator, WarcRecordType 
from resiliparse.extract.html2text import extract_plain_text 
from resiliparse.parse.encoding import detect_encoding 


def extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    """  
    Convert raw HTML bytes into clean plain text.
    Falls back to encoding detection if UTF-8 decoding fails.

    Args:
        html_bytes: Raw HTML content as bytes

    Returns:
        Extracted plain text as string, or None if extraction fails
    """
    try:
        # UTF-8 covers most modern web pages - try it first as the fast path
        html_str = html_bytes.decode('utf-8') 
    except UnicodeDecodeError:
        # Some pages use legacy encodings (latin-1, shift-jis, windows-1252, etc.)
        # detect_encoding inspects byte patterns/meta tags to guess the right charset
        encoding = detect_encoding(html_bytes)
        html_str = html_bytes.decode(encoding) 
    
    # Strips all HTML tags, scripts, and boilerplate - returns human-readable text 
    # This is the core NLP pre-processing step before any downstream text analysis
    extracted_text = extract_plain_text(html_str) 
    return extracted_text 


if __name__ == "__main__":
    # LOCAL VERSION: Read from local directory on your laptop
    import os
    from pathlib import Path
    
    # Point this to your local .warc.gz file or directory containing them
    # Use absolute path relative to project root (works from any directory)
    project_root = Path(__file__).parent.parent
    local_data_path = project_root / "CC-MAIN-20250417135010-20250417165010-00065.warc.gz"
    
    # # CLOUD VERSION (commented out for local development):
    # with gzip.open("/data/CC/example.warc.gz", "rb") as gzipped_file:
    
    # LOCAL VERSION:
    with gzip.open(local_data_path, "rb") as gzipped_file:
        count = 0 

        # ArchiveIterator lazily streams records - avoids loading the full archive into RAM 
        # record_types=WarcRecordType.response filters OUT request/metadata records,
        # keeping only the actual HTTP responses (the HTML page content we care about)
        for record in ArchiveIterator(gzipped_file, record_types=WarcRecordType.response):

            # record.reader.read() pulls the raw HTTP response body (HTML bytes) 
            html_bytes = record.reader.read()
            extracted_text = extract_text_from_html_bytes(html_bytes)

            # Each page gets its own numbered file for easy downstream inspection
            # LOCAL: Creates output directory in project root
            output_dir = project_root / "extracted_data"
            output_dir.mkdir(exist_ok=True)
            
            # # CLOUD VERSION (commented out):
            # with open(f"data/extracted_text_{count}.txt", "w") as f:
            #     f.write(extracted_text)
            
            # LOCAL VERSION:
            output_file = output_dir / f"extracted_text_{count}.txt"
            with open(output_file, "w") as f:
                f.write(extracted_text) 

            # Early exit - this is a dev/debug safeguard to avoid processing 
            # thousands of records when you just want to inspect a sample
            if count > 25:
                break 

            count += 1 


"""
Key Takeaways

WARC files are layered: the outer .gz compression wraps a WARC container, which itself holds
many typed records (requests, responses, metadata). Filtering by WarcRecordType.response is 
essential — otherwise you'd try to parse non-HTML records as page content.
Encoding handling is not optional for real-world web data: Common Crawl spans decades of web 
history across many languages. Blindly assuming UTF-8 will silently corrupt or crash on a 
significant fraction of pages. The try → detect → decode pattern here is the standard robust approach.
Streaming over loading: ArchiveIterator processes one record at a time rather than reading the
whole archive into memory — a critical pattern when working with multi-GB crawl files typical
in NLP/ML data pipelines.
"""