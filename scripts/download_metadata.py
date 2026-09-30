import urllib.request
from pathlib import Path
import pandas as pd
from src.config import DATA_ROOT

def main():
    labels_dir = DATA_ROOT / "picai_labels" / "clinical_information"
    labels_dir.mkdir(parents=True, exist_ok=True)
    marksheet_path = labels_dir / "marksheet.csv"
    
    url = "https://raw.githubusercontent.com/DIAGNijmegen/picai_labels/main/clinical_information/marksheet.csv"
    print(f"Téléchargement du marksheet depuis {url}...")
    urllib.request.urlretrieve(url, marksheet_path)
    print(f"Marksheet sauvegardé dans {marksheet_path}")

    df = pd.read_csv(marksheet_path)
    df["label"] = (df["case_ISUP"].astype(int) >= 3).astype(int)
    print("\n=== BALANCE GLOBALE DU MARKSHEET (1500 PATIENTS) ===")
    print(df["label"].value_counts())
    print(df["label"].value_counts(normalize=True).map("{:.2%}".format))

if __name__ == "__main__":
    main()
