"""
patch_dataset.py
----------------
Adds clean, verified-legitimate URLs to dataset.csv to fix
false positives caused by underrepresentation of major platforms.

Run once before training:
    python patch_dataset.py

train_final.py calls this automatically, so you don't need
to run it manually.
"""

import pandas as pd
import os

DATASET_PATH = "dataset.csv"

# ─────────────────────────────────────────────
# CLEAN LEGITIMATE URLs TO ADD
# These are real, safe URLs from verified organisations
# that were being falsely flagged due to class imbalance
# in the original training data.
# ─────────────────────────────────────────────
CLEAN_URLS = [
    # Google (heavily abused in dataset — needs more legit examples)
    "https://www.google.com",
    "https://mail.google.com",
    "https://docs.google.com",
    "https://drive.google.com",
    "https://www.google.com/maps",
    "https://calendar.google.com",
    "https://meet.google.com",
    "https://classroom.google.com",

    # GitHub
    "https://github.com",
    "https://github.com/torvalds/linux",
    "https://github.com/features",

    # LinkedIn
    "https://www.linkedin.com",
    "https://www.linkedin.com/feed",
    "https://www.linkedin.com/jobs",

    # Amazon
    "https://www.amazon.com",
    "https://www.amazon.com/s?k=laptop",

    # Wikipedia
    "https://www.wikipedia.org",
    "https://en.wikipedia.org/wiki/Python_(programming_language)",

    # YouTube
    "https://www.youtube.com",
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",

    # Netflix
    "https://www.netflix.com",
    "https://www.netflix.com/browse",

    # Twitter / X
    "https://twitter.com",
    "https://twitter.com/home",

    # Facebook / Instagram
    "https://www.facebook.com",
    "https://www.instagram.com",

    # Microsoft
    "https://www.microsoft.com",
    "https://outlook.live.com",
    "https://www.office.com",

    # Apple
    "https://www.apple.com",
    "https://support.apple.com",

    # ── Nigerian Banks ────────────────────────
    "https://www.gtbank.com",
    "https://www.gtbank.com/personal-banking",
    "https://ibank.gtbank.com",

    "https://www.zenithbank.com",
    "https://www.zenithbank.com/personal-banking",

    "https://www.accessbank.com",
    "https://www.accessbank.com/nigeria",

    "https://www.firstbanknigeria.com",
    "https://firstbanknigeria.com/personal",

    "https://www.uba.com",
    "https://www.uba.com/nigeria",

    "https://www.fcmb.com",
    "https://www.fidelitybank.ng",
    "https://www.sterling.ng",

    # ── Nigerian Fintechs ─────────────────────
    "https://www.kuda.com",
    "https://www.kuda.com/signup",

    "https://www.opay.com",
    "https://merchant.opay.ng",

    "https://www.palmpay.com",

    "https://www.moniepoint.com",
    "https://moniepoint.com/ng",

    "https://www.piggyvest.com",
    "https://www.piggyvest.com/investment",

    "https://www.cowrywise.com",

    "https://www.flutterwave.com",
    "https://dashboard.flutterwave.com",

    "https://www.paystack.com",
    "https://dashboard.paystack.com",

    "https://www.interswitch.com",
]


def patch_dataset(path: str = DATASET_PATH) -> pd.DataFrame:
    """
    Load dataset, append clean URLs not already present,
    shuffle, and save back to disk.

    Returns the updated DataFrame.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at: {path}")

    df      = pd.read_csv(path)
    before  = len(df)
    existing = set(df["url"].tolist())

    # Only add URLs not already in dataset
    new_entries = [u for u in CLEAN_URLS if u not in existing]

    if not new_entries:
        print("✅ Dataset already patched — no new URLs to add.")
        return df

    patch = pd.DataFrame({"url": new_entries, "label": 0})  # 0 = legitimate
    df    = pd.concat([df, patch], ignore_index=True)

    # Shuffle so new entries aren't all at the end
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(path, index=False)

    after = len(df)
    print(f"✅ Dataset patched: {before} → {after} URLs (+{after - before} clean entries)")
    print(f"   Label distribution:")
    print(f"   Legitimate : {(df['label']==0).sum()} ({(df['label']==0).mean()*100:.1f}%)")
    print(f"   Phishing   : {(df['label']==1).sum()} ({(df['label']==1).mean()*100:.1f}%)")
    return df


if __name__ == "__main__":
    patch_dataset()