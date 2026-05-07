import pandas as pd
import re
import os

def parse_train(file_path: str) -> pd.DataFrame:
    data = []
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line or ':::' not in line:
                continue
            parts = re.split(r'\s*:::\s*', line, maxsplit=3)
            if len(parts) == 4:
                _, title, genre, description = parts
                data.append({'plot': description.strip(), 'genre': genre.strip()})
    df = pd.DataFrame(data)
    print(f"✅ Parsed {len(df):,} training samples. Unique genres: {df['genre'].nunique()}")
    return df

def parse_test(file_path: str) -> pd.DataFrame:
    data = []
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line or ':::' not in line:
                continue
            parts = re.split(r'\s*:::\s*', line, maxsplit=2)
            if len(parts) == 3:
                id_, title, description = parts
                data.append({'id': id_.strip(), 'title': title.strip(), 'plot': description.strip()})
    df = pd.DataFrame(data)
    print(f"✅ Parsed {len(df):,} test samples")
    return df

# Main execution
print("🔄 Preparing data...")

train_df = parse_train('data/train_data.txt')
train_df.to_csv('data/train.csv', index=False)
print("💾 Saved data/train.csv")

test_df = parse_test('data/test_data.txt')
test_df.to_csv('data/test.csv', index=False)
print("💾 Saved data/test.csv")

print("\n✅ Data preparation complete!")