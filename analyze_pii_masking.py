"""
PII Masking Analysis Script

This script:
1. Runs PII masking on extracted text
2. Identifies documents where replacements were made
3. Shows 20 random examples with original and masked text
4. Lets user identify false positives and false negatives
"""

import random
from pathlib import Path
from cs336_data.mask import mask_emails, mask_phone_numbers, mask_ips

def analyze_pii_masking():
    project_root = Path(__file__).parent
    extracted_data_dir = project_root / "extracted_data"
    
    # Step 1: Get all extracted text files
    text_files = sorted(extracted_data_dir.glob("extracted_text_*.txt"))
    
    if not text_files:
        print("❌ No extracted text files found!")
        print("Please run: python cs336_data/extract.py first")
        return
    
    print(f"Found {len(text_files)} extracted text files\n")
    
    # Step 2: Run masking and collect results
    print("=" * 80)
    print("STEP 1: Running PII masking on all documents")
    print("=" * 80 + "\n")
    
    masking_results = {}
    
    for file_path in text_files:
        with open(file_path, "r") as f:
            original_text = f.read()
        
        # Run all three masks
        masked_emails, email_count = mask_emails(original_text)
        masked_phones, phone_count = mask_phone_numbers(masked_emails)
        masked_ips, ip_count = mask_ips(masked_phones)
        
        total_replacements = email_count + phone_count + ip_count
        
        if total_replacements > 0:  # Only keep files where replacements were made
            masking_results[file_path.name] = {
                "original": original_text,
                "masked": masked_ips,
                "email_count": email_count,
                "phone_count": phone_count,
                "ip_count": ip_count,
                "total": total_replacements
            }
            
            print(f"{file_path.name}: {email_count} emails, {phone_count} phones, {ip_count} IPs (total: {total_replacements})")
    
    if not masking_results:
        print("\n⚠️  No PII found in any documents!")
        print("This might mean the documents don't contain PII, or try with more documents.")
        return
    
    print(f"\nFound replacements in {len(masking_results)} documents\n")
    
    # Step 3: Select 20 random samples
    print("=" * 80)
    print("STEP 2: Selecting 20 random documents with PII matches")
    print("=" * 80 + "\n")
    
    sample_files = random.sample(
        list(masking_results.keys()), 
        min(20, len(masking_results))
    )
    
    print(f"Selected {len(sample_files)} files for review\n")
    
    # Step 4: Display samples and collect feedback
    print("=" * 80)
    print("STEP 3: Review masked text and identify errors")
    print("=" * 80)
    print("\nFor each sample, compare ORIGINAL vs MASKED text.")
    print("Record any FALSE POSITIVES (wrongly masked) or FALSE NEGATIVES (missed PII)\n")
    
    false_positives = []
    false_negatives = []
    
    for i, filename in enumerate(sample_files, 1):
        result = masking_results[filename]
        original = result["original"]
        masked = result["masked"]
        
        # Show context
        print(f"\n{'='*80}")
        print(f"SAMPLE {i} of {len(sample_files)}: {filename}")
        print(f"{'='*80}")
        print(f"Replacements: {result['email_count']} emails, {result['phone_count']} phones, {result['ip_count']} IPs\n")
        
        # Show original text with replacements highlighted
        print("ORIGINAL TEXT:")
        print("-" * 80)
        original_preview = original[:800] + "..." if len(original) > 800 else original
        print(original_preview)
        
        print("\n\nMASKED TEXT:")
        print("-" * 80)
        masked_preview = masked[:800] + "..." if len(masked) > 800 else masked
        print(masked_preview)
        
        print("\n")
        
        # Collect feedback
        while True:
            feedback = input("Any FALSE POSITIVES or FALSE NEGATIVES? (enter 'skip', 'fp: <description>', 'fn: <description>'): ").strip().lower()
            
            if feedback == 'skip':
                break
            elif feedback.startswith('fp:'):
                description = feedback[3:].strip()
                false_positives.append({
                    "file": filename,
                    "description": description
                })
                print(f"✓ Recorded false positive")
                break
            elif feedback.startswith('fn:'):
                description = feedback[3:].strip()
                false_negatives.append({
                    "file": filename,
                    "description": description
                })
                print(f"✓ Recorded false negative")
                break
            else:
                print("Please enter 'skip', 'fp: description', or 'fn: description'")
    
    print("\n")
    
    # Step 5: Summary
    print("=" * 80)
    print("STEP 4: Summary of Findings")
    print("=" * 80 + "\n")
    
    print(f"Documents reviewed: {len(sample_files)}")
    print(f"False positives found: {len(false_positives)}")
    print(f"False negatives found: {len(false_negatives)}")
    
    if false_positives:
        print("\n📙 FALSE POSITIVES (wrongly replaced):")
        for item in false_positives:
            print(f"  • {item['file']}: {item['description']}")
    
    if false_negatives:
        print("\n📓 FALSE NEGATIVES (missed PII):")
        for item in false_negatives:
            print(f"  • {item['file']}: {item['description']}")
    
    # Step 6: Save report
    report_file = project_root / "pii_masking_analysis.txt"
    with open(report_file, "w") as f:
        f.write("PII Masking Analysis Report\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Documents with PII: {len(masking_results)}\n")
        f.write(f"Samples reviewed: {len(sample_files)}\n")
        f.write(f"False positives: {len(false_positives)}\n")
        f.write(f"False negatives: {len(false_negatives)}\n\n")
        
        if false_positives:
            f.write("FALSE POSITIVES:\n")
            for item in false_positives:
                f.write(f"  • {item['file']}: {item['description']}\n")
            f.write("\n")
        
        if false_negatives:
            f.write("FALSE NEGATIVES:\n")
            for item in false_negatives:
                f.write(f"  • {item['file']}: {item['description']}\n")
            f.write("\n")
        
        f.write("\nDETAILED STATISTICS:\n")
        total_emails = sum(r['email_count'] for r in masking_results.values())
        total_phones = sum(r['phone_count'] for r in masking_results.values())
        total_ips = sum(r['ip_count'] for r in masking_results.values())
        
        f.write(f"Total emails masked: {total_emails}\n")
        f.write(f"Total phone numbers masked: {total_phones}\n")
        f.write(f"Total IPs masked: {total_ips}\n")
    
    print(f"\n✅ Report saved to: {report_file}\n")

if __name__ == "__main__":
    analyze_pii_masking()
