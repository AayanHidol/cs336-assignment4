"""
Language Identification Analysis Script

This script:
1. Extracts text from WARC files
2. Runs language identification on extracted text
3. Inspects 20 random samples for manual verification
4. Compares classifier predictions with manual labels
5. Reports statistics and suggests confidence thresholds
"""

import random
from pathlib import Path
from cs336_data.identify import identify_language

def analyze_language_identification():
    project_root = Path(__file__).parent
    extracted_data_dir = project_root / "extracted_data"
    
    # Step 1: Get all extracted text files
    text_files = sorted(extracted_data_dir.glob("extracted_text_*.txt"))
    
    if not text_files:
        print("❌ No extracted text files found!")
        print("Please run: python cs336_data/extract.py first")
        return
    
    print(f"Found {len(text_files)} extracted text files\n")
    
    # Step 2: Run language identification on all files
    print("=" * 80)
    print("STEP 1: Running language identification on all extracted documents")
    print("=" * 80)
    
    predictions = {}
    for file_path in text_files:
        with open(file_path, "r") as f:
            text = f.read()
        
        lang_code, confidence = identify_language(text)
        predictions[file_path.name] = {
            "text": text,
            "predicted_lang": lang_code,
            "confidence": confidence
        }
        print(f"{file_path.name}: {lang_code} (confidence: {confidence:.4f})")
    
    print("\n")
    
    # Step 3: Select 20 random samples for manual verification
    print("=" * 80)
    print("STEP 2: Preparing 20 random samples for manual verification")
    print("=" * 80)
    
    sample_files = random.sample(list(predictions.keys()), min(20, len(predictions)))
    
    print(f"\nSelected {len(sample_files)} files for manual review:")
    for i, filename in enumerate(sample_files, 1):
        print(f"{i}. {filename}")
    
    print("\n")
    
    # Step 4: Display samples for manual annotation
    print("=" * 80)
    print("STEP 3: Manual Language Verification")
    print("=" * 80)
    print("\nFor each sample, read the text and identify the language manually.")
    print("When prompted, enter the language code (e.g., 'en' for English, 'fr' for French, etc.)\n")
    
    manual_labels = {}
    for i, filename in enumerate(sample_files, 1):
        pred = predictions[filename]
        text_preview = pred["text"][:500] + "..." if len(pred["text"]) > 500 else pred["text"]
        
        print(f"\n{'='*80}")
        print(f"SAMPLE {i} of {len(sample_files)}: {filename}")
        print(f"{'='*80}")
        print(f"\nClassifier Prediction: {pred['predicted_lang']} (confidence: {pred['confidence']:.4f})")
        print(f"\nText Preview:")
        print(f"---\n{text_preview}\n---")
        
        while True:
            manual_lang = input(f"\nWhat language is this? (enter code like 'en', 'fr', 'zh', 'es', 'de', 'other', etc.): ").strip().lower()
            if manual_lang:
                manual_labels[filename] = manual_lang
                break
            print("Please enter a language code.")
    
    print("\n")
    
    # Step 5: Compare predictions with manual labels
    print("=" * 80)
    print("STEP 4: Comparing Classifier Predictions vs Manual Labels")
    print("=" * 80 + "\n")
    
    correct = 0
    errors = []
    language_counts = {}
    english_count = 0
    
    print(f"{'File':<35} {'Pred':<8} {'Conf':<8} {'Manual':<8} {'Match':<8}")
    print("-" * 80)
    
    for filename in sample_files:
        pred_lang = predictions[filename]["predicted_lang"]
        confidence = predictions[filename]["confidence"]
        manual_lang = manual_labels[filename]
        
        is_match = pred_lang == manual_lang
        correct += is_match
        
        # Count languages
        if manual_lang not in language_counts:
            language_counts[manual_lang] = 0
        language_counts[manual_lang] += 1
        
        if manual_lang == "en":
            english_count += 1
        
        match_str = "✓" if is_match else "✗"
        print(f"{filename:<35} {pred_lang:<8} {confidence:<8.4f} {manual_lang:<8} {match_str:<8}")
        
        if not is_match:
            errors.append({
                "file": filename,
                "predicted": pred_lang,
                "actual": manual_lang,
                "confidence": confidence
            })
    
    print("\n")
    
    # Step 6: Calculate and report statistics
    print("=" * 80)
    print("STEP 5: Statistics and Analysis")
    print("=" * 80 + "\n")
    
    accuracy = (correct / len(sample_files)) * 100
    english_fraction = (english_count / len(sample_files)) * 100
    
    print(f"Accuracy: {correct}/{len(sample_files)} = {accuracy:.1f}%")
    print(f"Errors: {len(errors)}/{len(sample_files)}")
    print(f"\nEnglish documents: {english_count}/{len(sample_files)} = {english_fraction:.1f}%")
    print(f"\nLanguage distribution in samples:")
    for lang, count in sorted(language_counts.items(), key=lambda x: x[1], reverse=True):
        pct = (count / len(sample_files)) * 100
        print(f"  {lang:<5}: {count:>2} ({pct:>5.1f}%)")
    
    print("\n")
    
    # Step 7: Analyze errors and suggest confidence threshold
    if errors:
        print("=" * 80)
        print("STEP 6: Error Analysis")
        print("=" * 80 + "\n")
        
        print("Classifier Errors:")
        print(f"{'File':<35} {'Predicted':<12} {'Conf':<8} {'Actual':<12}")
        print("-" * 80)
        
        avg_error_confidence = 0
        for error in errors:
            print(f"{error['file']:<35} {error['predicted']:<12} {error['confidence']:<8.4f} {error['actual']:<12}")
            avg_error_confidence += error['confidence']
        
        avg_error_confidence /= len(errors)
        print(f"\nAverage confidence on errors: {avg_error_confidence:.4f}")
        
        # Analyze confidence by correctness
        correct_confidences = []
        error_confidences = []
        
        for filename in sample_files:
            conf = predictions[filename]["confidence"]
            pred_lang = predictions[filename]["predicted_lang"]
            manual_lang = manual_labels[filename]
            
            if pred_lang == manual_lang:
                correct_confidences.append(conf)
            else:
                error_confidences.append(conf)
        
        if correct_confidences:
            min_correct_conf = min(correct_confidences)
            print(f"Minimum confidence on correct predictions: {min_correct_conf:.4f}")
        
        if error_confidences:
            max_error_conf = max(error_confidences)
            print(f"Maximum confidence on errors: {max_error_conf:.4f}")
    
    print("\n")
    
    # Step 8: Recommend confidence threshold
    print("=" * 80)
    print("STEP 7: Confidence Threshold Recommendation")
    print("=" * 80 + "\n")
    
    if errors and correct_confidences:
        max_error = max(error_confidences) if error_confidences else 0
        min_correct = min(correct_confidences)
        
        print(f"Based on your manual verification:")
        print(f"  - Errors occurred at confidences up to: {max_error:.4f}")
        print(f"  - Correct predictions started at: {min_correct:.4f}")
        
        # Suggest threshold
        if max_error > 0:
            suggested_threshold = (max_error + min_correct) / 2
            print(f"\n💡 RECOMMENDED CONFIDENCE THRESHOLD: {suggested_threshold:.4f}")
            print(f"   Using this threshold would filter out {len(error_confidences)} error(s)")
        else:
            print(f"\n💡 No errors detected! Classifier appears reliable on this sample.")
            print(f"   You could use a low threshold like 0.5 for conservative filtering.")
    else:
        print("💡 Insufficient data for threshold recommendation.")
        print("   Please review more samples or use domain knowledge.")
    
    print("\n")
    
    # Save results to file
    results_file = project_root / "language_analysis_results.txt"
    with open(results_file, "w") as f:
        f.write("Language Identification Analysis Results\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Samples Analyzed: {len(sample_files)}\n")
        f.write(f"Accuracy: {accuracy:.1f}%\n")
        f.write(f"English Documents: {english_fraction:.1f}%\n")
        f.write(f"\nLanguage Distribution:\n")
        for lang, count in sorted(language_counts.items(), key=lambda x: x[1], reverse=True):
            pct = (count / len(sample_files)) * 100
            f.write(f"  {lang}: {count} ({pct:.1f}%)\n")
        
        if errors:
            f.write(f"\nErrors ({len(errors)}):\n")
            for error in errors:
                f.write(f"  {error['file']}: predicted {error['predicted']} (conf: {error['confidence']:.4f}), actual {error['actual']}\n")
    
    print(f"✅ Results saved to: {results_file}")

if __name__ == "__main__":
    analyze_language_identification()
