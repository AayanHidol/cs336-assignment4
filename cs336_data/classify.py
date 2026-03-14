import fasttext
from typing import Tuple
from pathlib import Path 


def _load_model(path: str) -> fasttext.FastText._FastText:
    """  
    Helper to load a fastText model with a per-path singleton cache.
    Storing on the function itself avoids globals while surviving multiple calls
    """
    # _load_model.cache is a dict mapping model path -> loaded model object
    # This handles multiple models (nsfw, toxicity) unlike the langugae ID code
    # which only needed one model and used a simpler single-attribute check
    if not hasattr(_load_model, 'cache'):
        _load_model.cache = {} 

    if path not in _load_model.cache:
        _load_model.cache[path] = fasttext.load_model(path) 
    
    return _load_model.cache[path] 


def classify_nsfw(text: str) -> Tuple[str, float]:
    """
    Classifies text as 'nsfw' or 'non-nsfw'.
    Uses a model trained on the Jigsaw dataset via the Dolma pipeline
    """
    # # CLOUD VERSION (commented out):
    # model = _load_model("/data/classifiers/dolma_fasttext_nsfw_jigsaw_model.bin")
    
    # LOCAL VERSION:
    model_path = Path(__file__).parent / 'assets' / 'dolma_fasttext_nsfw_jigsaw_model.bin'
    model = _load_model(str(model_path)) 

    # Newlines break fastText's input assumptions - it expects single-line strings
    text = text.replace('\n', ' ').strip() 

    predictions = model.predict(text) 
    label = predictions[0][0].replace('__label__', '')
    confidence = float(predictions[1][0])

    return label, confidence 


def classify_toxic_speech(text: str) -> Tuple[str, float]:
    """
    Classifies text as 'toxic' or 'non-toxic'.
    Separate model from NSFW - hate speech and adult content are distinct signal types.
    """
    # # CLOUD VERSION (commented out):
    # model = _load_model("/data/classifiers/dolma_fasttext_hatespeech_jigsaw_model.bin")
    
    # LOCAL VERSION:
    model_path = Path(__file__).parent / 'assets' / 'dolma_fasttext_hatespeech_jigsaw_model.bin'
    model = _load_model(str(model_path)) 

    text = text.replace('\n', '').strip() 

    predictions = model.predict(text)
    label = predictions[0][0].replace('__label__', '')
    confidence = float(predictions[1][0]) 

    return label, confidence 


if __name__ == "__main__":
    # LOCAL VERSION: Read from extracted_data folder
    project_root = Path(__file__).parent.parent
    extracted_data_dir = project_root / "extracted_data"
    
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
            
            nsfw_label, nsfw_conf = classify_nsfw(text)
            toxic_label, toxic_conf = classify_toxic_speech(text)
            print(f"{file_path.name}: NSFW: {nsfw_label} ({nsfw_conf:.2f}) | Toxic: {toxic_label} ({toxic_conf:.2f})")
        
        # # CLOUD VERSION (commented out):
        # for i in range(25):
        #     with open(f"data/extracted_text_{i}.txt", "r") as f:
        #         text = f.read() 
        #
        #     nsfw_label, nsfw_conf = classify_nsfw(text)
        #     toxic_label, toxic_conf = classfiy_toxic_speech(text) 
        #     print(f"[{i}] NSFW: {nsfw_label} ({nsfw_conf:.2f}) | Toxic: {toxic_label} ({toxic_conf:.2f})") 