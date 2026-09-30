import pandas as pd

def analyze():
    url = "https://raw.githubusercontent.com/DIAGNijmegen/picai_labels/main/clinical_information/marksheet.csv"
    print(f"Chargement du marksheet officiel depuis GitHub...")
    df = pd.read_csv(url)
    
    # Nouveau Seuil : ISUP >= 2 -> 1 (csPCa), sinon 0
    df["label"] = (df["case_ISUP"].astype(int) >= 2).astype(int)
    
    total = len(df)
    counts = df["label"].value_counts()
    percentages = (df["label"].value_counts(normalize=True) * 100).round(2)
    
    print("\n" + "="*55)
    print("   NOUVELLE RÉPARTITION DES CLASSES (SEUIL ISUP >= 2)")
    print("="*55)
    print(f"Nombre total de patients dans le dataset : {total}")
    print("-" * 55)
    print(f"Classe 0 (Non-csPCa / ISUP < 2 - Bénin / ISUP 1) : {counts[0]} patients ({percentages[0]}%)")
    print(f"Classe 1 (csPCa / ISUP >= 2 - Gleason 3+4+)      : {counts[1]} patients ({percentages[1]}%)")
    print("-" * 55)
    ratio = counts[0] / counts[1]
    print(f"Nouveau ratio de déséquilibre : {ratio:.2f} pour 1 (soit ~{ratio:.1f}:1)")
    print("="*55)

if __name__ == "__main__":
    analyze()
