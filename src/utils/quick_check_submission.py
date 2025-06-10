import pandas as pd
import numpy as np
import os

SUB_PATH = '/kaggle/working/submission.csv'

print(f"Checking submission file: {SUB_PATH}")

# 1. Check file exists
if not os.path.exists(SUB_PATH):
    print("❌ submission.csv not found!")
    exit(1)
else:
    print("✅ submission.csv found.")

# 2. Load and check columns
try:
    df = pd.read_csv(SUB_PATH)
except Exception as e:
    print(f"❌ Failed to read submission.csv: {e}")
    exit(1)

expected_cols = ['oid_ypos'] + [f'x_{x}' for x in range(1, 70, 2)]
if list(df.columns) != expected_cols:
    print(f"❌ Columns do not match expected format.\nExpected: {expected_cols}\nFound: {list(df.columns)}")
    exit(1)
else:
    print("✅ Columns are correct.")

# 3. Check for NaNs or Infs
if df.isna().any().any():
    print("❌ Found NaN values in submission.")
    exit(1)
if np.isinf(df.select_dtypes(include=[np.number])).any().any():
    print("❌ Found Inf values in submission.")
    exit(1)
print("✅ No NaNs or Infs found.")

# 4. Check number of rows is a multiple of 70
if df.shape[0] % 70 != 0:
    print(f"❌ Number of rows ({df.shape[0]}) is not a multiple of 70.")
    exit(1)
else:
    print(f"✅ Number of rows: {df.shape[0]} (multiple of 70)")

# 5. Print a few sample rows
print("\nSample rows:")
print(df.head(5).to_string(index=False))

print("\nAll checks passed! 🚀") 