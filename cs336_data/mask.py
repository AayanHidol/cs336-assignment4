"""
OVERVIEW:
This code performs PII (Personally Identifiable Information) masking — a standard data 
cleaning step in NLP pipelines before text is used for training or analysis. It uses regular 
expressions to detect and replace emails, phone numbers, and IP addresses with 
placeholder tokens like |||EMAIL_ADDRESS|||. The hardest part is the phone number 
regex, which must handle many real-world formats (dashes, parentheses, spaces) without 
accidentally matching random digit sequences.
"""

import re
from pathlib import Path 

def mask_emails(text):
    """
    Replaces all email addresses with |||EMAIL_ADDRESS|||
    """

    # Anatomy of this pattern:
    #   [a-zA-Z0-9._%+-]+  -> local part (before @): allows dots, plus etc.
    #   @                  -> literal @ symbol
    #   [a-zA-Z0-9.-]+     -> domain name (e.g. 'gmail', 'company.co')
    #   \.[a-zA-Z]{2,}     -> TLD: dot followd by 2+ letters (e.g. .com, .io)
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'

    # findall before sub - re.sub can't directly return a count,
    # so we measure what will be replaced first
    emails_found = re.findall(email_pattern, text) 
    num_masked = len(emails_found) 

    masked_text = re.sub(email_pattern, '|||EMAIL_ADDRESS|||', text)
    return masked_text, num_masked 


def mask_phone_numbers(text):
    """
    Replaces all phone numbers with |||PHONE_NUMBER|||
    """

    # This pattern handles the messiness of real-world US phone formats:
    #   (\s*)               → capture leading whitespace (Group 1) — see sub below
    #   \+?1?               → optional country code: +1 or 1
    #   \s*[\(\s\-]?        → optional opening delimiter: '(', space, or dash
    #   (\d{3})             → area code — 3 digits (Group 2)
    #   [\)\s\-]*\s*        → optional closing delimiter: ')', space, or dash
    #   (\d{3})             → exchange — first 3 digits of local number (Group 3)
    #   [\s\-]*             → optional separator
    #   \d{4}               → subscriber — last 4 digits (not captured)
    phone_pattern = r'(\s*)\+?1?\s*[\(\s\-]?(\d{3})[\)\s\-]*\s*(\d{3})[\s\-]*\d{4}'

    phone_numbers_found = re.findall(phone_pattern, text)
    num_masked = len(phone_numbers_found) 

    # \1 in the replacement re-inserts the captured leading whitespace (Group 1) 
    # Without this, "call me at 123-456-7890" would lose the space before the token
    masked_text = re.sub(phone_pattern, r'\1|||PHONE_NUMBER|||', text) 
    return masked_text, num_masked 


def mask_ips(text):
    """
    Replace all IPv4 addresses with |||IP_ADDRESS|||
    """
    # Each octet must be a valid 0–255 value — the alternation handles this:
    #   25[0-5]   → 250–255
    #   2[0-4]\d  → 200–249
    #   1\d\d     → 100–199
    #   \d\d      → 10–99
    #   \d        → 0–9
    # The {3} repeats "octet + dot" three times, then the last octet closes without a dot
    # \b (word boundary) prevents matching octets inside larger numbers like "1.2.3.4567"
    ip_pattern = r'\b(?:(?:25[0-5]|2[0-4]\d|1\d\d|\d\d|\d)\.){3}(?:25[0-5]|2[0-4]\d|1\d\d|\d\d|\d)\b'

    ips_found = re.findall(ip_pattern, text)
    num_masked = len(ips_found) 

    masked_text = re.sub(ip_pattern, '|||IP_ADDRESS|||', text) 
    return masked_text, num_masked 


if __name__ == "__main__":
    # Sanity-check the phone masker against known formats before touching real data
    numbers = ["2831823829", "(283)-182-3829", "(283) 182 3829", "283-182-3829"]
    for number in numbers:
        test_string = f"Feel free to contact me at {number} if you have any questions."
        masked_text, num_masked = mask_phone_numbers(test_string)
        print(masked_text, num_masked)

    # LOCAL VERSION: Apply all three masks in sequence on locally extracted text
    project_root = Path(__file__).parent.parent
    extracted_data_dir = project_root / "extracted_data"
    masked_data_dir = project_root / "masked_data"
    
    # Create output directory
    masked_data_dir.mkdir(exist_ok=True)
    
    # Get all extracted text files
    text_files = sorted(extracted_data_dir.glob("extracted_text_*.txt"))
    
    if not text_files:
        print(f"No extracted text files found in {extracted_data_dir}")
        print("Please run extract.py first to generate the extracted text files.")
    else:
        print(f"Processing {len(text_files)} files from {extracted_data_dir}\n")
        
        # LOCAL VERSION:
        for file_path in text_files:
            with open(file_path, "r") as f:
                text = f.read() 

            # Apply all three masks in sequence - output of each feeds into the next 
            masked_text = mask_emails(text)[0]                      # [0] discards the count, keeps text 
            masked_text = mask_phone_numbers(masked_text)[0]
            masked_text = mask_ips(masked_text)[0]

            # Write to local masked_data directory
            output_file = masked_data_dir / file_path.name.replace("extracted_", "masked_")
            with open(output_file, "w") as f:
                f.write(masked_text)
            
            print(f"Processed {file_path.name} -> {output_file.name}")
    
    # # CLOUD VERSION (commented out):
    # for i in range(25):
    #     with open(f"data/extracted_text_{i}.txt", "r") as f:
    #         text = f.read() 
    # 
    #     masked_text = mask_emails(text)[0]
    #     masked_text = mask_phone_numbers(masked_text)[0]
    #     masked_text = mask_ips(masked_text)[0]
    # 
    #     with open(f"data/masked_text_{i}.txt", "w") as f:
    #         f.write(masked_text) 


"""
findall before sub is a deliberate two-pass trade-off: re.sub alone can't return a 
match count. Calling findall first adds a second scan, but keeps the functions 
informative (returning counts alongside masked text). At scale you'd use re.subn() 
instead — it replaces and returns the count in one pass.
"""