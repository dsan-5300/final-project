"""
Output: data/raw/
  - DEMO_J.csv
  - OHXDEN_J.csv
  - OHQ_J.csv
  - OHXREF_J.csv
"""

import io
import os
import warnings

import pandas as pd
import requests


BASE_URL = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/"

FILES = {
    "DEMO_J": "DEMO_J.xpt",
    "OHXDEN_J": "OHXDEN_J.xpt",
    "OHQ_J": "OHQ_J.xpt",
    "OHXREF_J": "OHXREF_J.xpt",
}


def fetch_xpt(url: str) -> pd.DataFrame:
    # Download an XPT file over HTTPS and return it as a DataFrame
    with warnings.catch_warnings():
        # NHANES site has an expired SSL cert, so disable warnings about that here
        warnings.simplefilter("ignore", requests.packages.urllib3.exceptions.InsecureRequestWarning)
        response = requests.get(url, verify=False, timeout=120) # long timeout since some files are large
    response.raise_for_status() # raise an error if the download failed
    return pd.read_sas(io.BytesIO(response.content), format="xport", encoding="utf-8") # read the XPT file into a DataFrame


def download_raw_nhanes(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True) # ensure the output directory exists
    # Loop through each file, download it, convert to pd.DataFrame, and save as CSV
    for name, filename in FILES.items():

        url = BASE_URL + filename # construct the full URL for the file
        print(f"Downloading {filename}...")
        df = fetch_xpt(url) # download the file
        print(f"  -> {df.shape[0]:,} rows, {df.shape[1]} cols") # print the shape of the DataFrame

        csv_path = os.path.join(output_dir, f"{name}.csv")
        df.to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}")


def main():
    # Determine the output directory (data/raw/) relative to the root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # assumes the script is located at scripts/download_raw_nhanes.py
    output_dir = os.path.join(project_root, "data", "raw")
    print("Downloading raw NHANES files (no cleaning or transformation)...")
    download_raw_nhanes(output_dir)
    print("\nDone. All four raw CSVs saved to:", output_dir)

if __name__ == "__main__":
    main()
