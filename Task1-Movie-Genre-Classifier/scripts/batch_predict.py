import pandas as pd
import sys
sys.path.insert(0, '.')

from src.models import GenreClassifier
from src.data_preprocessing import TextPreprocessor

def batch_predict_fast(test_csv: str, model_path: str, output_path: str = 'submission.txt'):
    print("🔄 Loading model...")
    classifier = GenreClassifier.load(model_path)
    preprocessor = TextPreprocessor()
    
    df = pd.read_csv(test_csv)
    print(f"🔄 Predicting {len(df):,} samples (fast mode)...")
    
    # Clean all texts at once
    print("   Cleaning text...")
    clean_plots = preprocessor.transform(df['plot'])
    
    # Predict all at once (much faster!)
    print("   Running predictions...")
    predictions = classifier.predict(clean_plots)
    
    # Build submission lines
    lines = []
    for idx, (row, genre) in enumerate(zip(df.itertuples(), predictions)):
        lines.append(f"{row.id} ::: {row.title} ::: {genre} ::: {row.plot}")
        
        if (idx + 1) % 5000 == 0:
            print(f"   Processed {idx + 1:,} / {len(df):,} samples...")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"\n✅ Done! Submission saved to: {output_path}")
    print(f"   Total predictions: {len(lines):,}")

if __name__ == "__main__":
    batch_predict_fast('data/test.csv', 'models/logistic_regression_tfidf.pkl', 'submission.txt')