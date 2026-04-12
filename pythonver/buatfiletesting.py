from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_PATH = PROJECT_ROOT / "dataset_labeled_wsn.xlsx"
OUTPUT_PATH = PROJECT_ROOT / "dataset_labeled_wsn_testing.csv"

df = pd.read_excel(DATA_PATH)

_, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    shuffle=True,
    stratify=df["label"]
)

test_df.to_csv(OUTPUT_PATH, index=False)
print(f"Berhasil dibuat: {OUTPUT_PATH}")
