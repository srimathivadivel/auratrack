import os
import pandas as pd

CSV_FILE_PATH = "Data/train.csv"

# Check if file exists
if not os.path.exists(CSV_FILE_PATH):
    print(f"File not found: {CSV_FILE_PATH}")
else:
    print(f"File found: {CSV_FILE_PATH}")

    # Try loading the CSV
    try:
        df = pd.read_csv(CSV_FILE_PATH)
        print("CSV loaded successfully! Here are the first few rows:")
        print(df.head())
    except Exception as e:
        print(f"Error loading CSV: {e}")
