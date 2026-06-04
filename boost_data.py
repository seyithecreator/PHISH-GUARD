import pandas as pd
import random

# Load current dataset
df = pd.read_csv("dataset.csv")

# Real-world phishing components
brands = ['opay', 'palmpay', 'kuda', 'moniepoint', 'zenith', 'gtb', 'firstbank', 'access']
# Sneaky substitutions
fake_brands = ['0pay', 'pampay', 'kuda-bank', 'monie-point', 'gtbank-mobile']
suffixes = ['-bvn-portal.ng', '-secure.com', '-update.net', '-login.live', '.top', '.xyz']
paths = ['/login', '/verify', '/update-account', '/bvn-check']

synthetic_phish = []

# Generate 1,500 varied, "hard" samples
for _ in range(1500):
    # Mix real brands and sneaky versions
    brand = random.choice(brands + fake_brands)
    suffix = random.choice(suffixes)
    path = random.choice(paths)
    
    # Randomly choose between http and https (phishers use both now)
    protocol = random.choice(['http://', 'https://'])
    url = f"{protocol}{brand}{suffix}{path}"
    
    synthetic_phish.append({'url': url, 'label': 1})

# Combine and drop duplicates to keep it clean
new_data = pd.concat([df, pd.DataFrame(synthetic_phish)], ignore_index=True)
new_data = new_data.drop_duplicates(subset='url')

# Save it back
new_data.to_csv("dataset.csv", index=False)

print(f"🚀 BOOSTER ENGAGED: Total dataset size is now {len(new_data)} rows.")
print("Now run 'python3 train_final.py' to see the accuracy jump!")