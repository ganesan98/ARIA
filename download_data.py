import os
import requests
from tqdm import tqdm

# We're using the CTU-IoT-23 dataset, specifically scenario 34
# which contains labeled benign + malicious IoT traffic
# Source: Stratosphere Lab, CTU University

URL = "https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset/IndividualScenarios/CTU-IoT-Malware-Capture-34-1/bro/conn.log.labeled"
OUTPUT = "data/conn.log.labeled"

os.makedirs("data", exist_ok=True)

def download():
    if os.path.exists(OUTPUT):
        print("File already downloaded.")
        return

    print("Downloading IoT-23 scenario 34...")
    r = requests.get(URL, stream=True)
    total = int(r.headers.get('content-length', 0))

    with open(OUTPUT, 'wb') as f, tqdm(total=total, unit='B', unit_scale=True) as bar:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))

    print(f"Saved to {OUTPUT}")

if __name__ == "__main__":
    download()