"""
OVERVIEW:

This code uses Meta's fastText model (lid.176.bin) to identify the language of text files
- the same model used in production pipelines at companies like Meta and in datasets like
C4/mC4. It runs inference on the 25 extracted text files from the previous step. The key
engineering insight is the lazy-loading singleton pattern: the 130MB model is loaded once 
and cached on the function itself, avoiding expensive repeated disk reads across calls.
"""

import fasttext
from pathlib import Path 

def identify_language(text: str, model_path: str = None) -> tuple[str, float]:
    """
    Identify the language of a text string using fastText's LID model.
    Returns a (language_code, confidence) tuple, e.g. ('en', 0.98).
    
    Args:
        text: The text to identify language for
        model_path: Path to the fastText model. If None, uses default.
    """
    # Singleton pattern: store the model as a function attribute so it's loaded
    # once on first call and reused on every subsequent call.
    # Loading a 130MB model on every call would be extremely slow at scale.
    if not hasattr(identify_language, 'model'):
        if model_path is None:
            # # CLOUD VERSION (commented out):
            # model_path = '/data/classifiers/lid.176.bin'
            
            # LOCAL VERSION:
            # Model is stored in cs336_data/assets/ directory
            model_path = Path(__file__).parent / 'assets' / 'lid.176.bin'
        
        identify_language.model = fasttext.load_model(str(model_path)) 
    
    # fastText treats newlines as sentence boundaries, which can confuse the
    # classifier - collapsing to spaces presents the text as a single input
    text = text.replace('\n', ' ') 

    # k=1 means "return only the top prediction"
    # predictions[0] holds label strings, predictions[1] holds confidence scores
    # e.g. predictions = (['__label__en'], array([0.98]))
    predictions = identify_language.model.predict(text, k=1) 
    lang_code = predictions[0][0] 
    confidence = float(predictions[1][0]) 

    # fastText prefixes all labels with '__label__' internally (a training artifact) 
    # Strip it so callers get clean codes like 'en', 'fr', 'zh' instead of '__label__en'
    lang_code = lang_code.replace('__label__', '') 

    return lang_code, confidence 


if __name__ == "__main__":
    # LOCAL VERSION: Read from extracted_data folder
    from pathlib import Path
    
    project_root = Path(__file__).parent.parent
    extracted_data_dir = project_root / "extracted_data"
    
    # Get all extracted text files and process them
    text_files = sorted(extracted_data_dir.glob("extracted_text_*.txt"))
    
    if not text_files:
        print(f"No extracted text files found in {extracted_data_dir}")
        print("Please run extract.py first to generate the extracted text files.")
    else:
        print(f"Processing {len(text_files)} files from {extracted_data_dir}\n")
        
        # # CLOUD VERSION (commented out):
        # for i in range(25):
        #     with open(f"data/extracted_text_{i}.txt", "r") as f:
        #         text = f.read() 
        #         lang_code, confidence = identify_language(text) 
        #         print(f"Language: {lang_code}, Confidence: {confidence}")
        
        # LOCAL VERSION:
        for file_path in text_files:
            with open(file_path, "r") as f:
                text = f.read()
                lang_code, confidence = identify_language(text)
                print(f"{file_path.name}: Language: {lang_code}, Confidence: {confidence:.4f}")


"""
KEY TAKEAWAYS:
* Singleton via function attributes: Attaching the model to the function object with 
  `identify_langugae.model` is a clean Pythonic way to persist state without a global 
   variable or a class. It's a common pattern in ML pipelines where initialization is 
   expensive but the function may be called thousands of times.

* fastText's output format requires cleanup: The __label__ prefix is an artifact of how
  fastText formats training data — every class label must have it during training. Always 
  strip it before using the output, or downstream code comparing against 'en' will 
  silently never match '__label__en'.

* Confidence scores are not probabilities: The score from `predict()` is a softmax
  output, not a calibrated probability. A score of `0.98` means the model is very certain 
  relative to the other 175 languages, but it doesn't mean there's a true 98% chance of 
  correctness - treat it as a relative confidence ranking, not an absoulute truth metric. 
"""