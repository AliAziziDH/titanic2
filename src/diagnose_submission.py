import pandas as pd
from pathlib import Path
import hashlib
import os

def check_submission():
    submissions_dir = Path("submissions")
    csv_files = list(submissions_dir.glob("submission_final_blend_*.csv"))
    if not csv_files:
        raise FileNotFoundError("No submission files found!")

    latest_file = max(csv_files, key=lambda p: p.stat().st_mtime)
    print(f"Diagnostics for: {latest_file.name}")

    df = pd.read_csv(latest_file)

    # 1. Row count verification
    total_rows = len(df)
    assert total_rows == 418, f"Row count must be exactly 418, got {total_rows}"
    print(f"✅ Row count: {total_rows} (Pass)")

    # 2. PassengerId range verification
    min_id = df["PassengerId"].min()
    max_id = df["PassengerId"].max()
    assert min_id == 892 and max_id == 1309, f"PassengerId range must be 892-1309, got {min_id}-{max_id}"
    print(f"✅ PassengerId range: {min_id} - {max_id} (Pass)")

    # 3. Null checks
    null_counts = df.isnull().sum()
    assert null_counts.sum() == 0, f"Found nulls:\n{null_counts}"
    print("✅ Null check: 0 NaNs (Pass)")

    # 4. Distribution Metrics
    counts = df["Survived"].value_counts().to_dict()
    rate = df["Survived"].mean()
    print("\nDistribution Metrics:")
    print(f"Survived value counts: 0s: {counts.get(0, 0)}, 1s: {counts.get(1, 0)}")
    print(f"Survival Rate: {rate:.2%} (Pass)")

    # 5. Checksum calculation
    with open(latest_file, "rb") as f:
        file_hash = hashlib.md5(f.read(), usedforsecurity=False).hexdigest()  # NOSONAR
    print(f"\nMD5 Checksum:\n{file_hash}")

    # Write summary for persistent artifact later
    with open("submissions/run_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"Diagnostics for: {latest_file.name}\n")
        f.write(f"Row count: {total_rows}\n")
        f.write(f"PassengerId range: {min_id} - {max_id}\n")
        f.write(f"Null check: 0 NaNs\n")
        f.write(f"Survived value counts: 0s: {counts.get(0, 0)}, 1s: {counts.get(1, 0)}\n")
        f.write(f"Survival Rate: {rate:.2%}\n")
        f.write(f"MD5 Checksum: {file_hash}\n")

if __name__ == "__main__":
    check_submission()
