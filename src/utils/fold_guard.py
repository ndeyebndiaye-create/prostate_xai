"""
Guards contre la fuite de données (data leakage) entre folds.

La fuite de données est le piège n°1 en machine learning médical :
si un patient apparaît à la fois dans le train et le val d'un même fold,
le modèle "triche" en ayant déjà vu ce patient. L'AUROC rapporté sera
optimiste et ne reflètera pas les performances réelles sur de nouveaux patients.

Ce module fournit deux fonctions :
  - assert_no_patient_overlap : vérifie l'absence de fuite (lève RuntimeError)
  - log_fold_class_distribution : affiche la distribution train/val pour vérifier
    que StratifiedKFold a bien conservé le ratio de classes dans chaque fold.
"""

from __future__ import annotations


def assert_no_patient_overlap(train_ids: list[str], val_ids: list[str]) -> None:
    """
    Vérifie qu'aucun patient n'apparaît simultanément dans train et val.

    Lève RuntimeError avec les IDs fautifs si une fuite est détectée.
    Dans un usage normal avec StratifiedKFold, cette assertion ne doit
    jamais échouer — c'est une vérification de sécurité, pas un test.

    Parameters
    ----------
    train_ids:
        Liste des IDs patients du fold d'entraînement.
    val_ids:
        Liste des IDs patients du fold de validation.
    """
    overlap = set(train_ids) & set(val_ids)
    if overlap:
        sample = sorted(overlap)[:5]
        extra = "..." if len(overlap) > 5 else ""
        raise RuntimeError(
            f"[fold_guard] FUITE DÉTECTÉE : {len(overlap)} patient(s) dans "
            f"train ET val : {sample}{extra}\n"
            f"Vérifier le split — aucun patient ne doit apparaître dans les deux."
        )


def log_fold_class_distribution(
    labels: list[int],
    fold: int,
    split: str = "val",
) -> None:
    """
    Affiche la distribution de classes dans un fold pour validation visuelle.

    Avec StratifiedKFold le ratio est normalement conservé, mais avec un
    petit nombre de patients par fold (~20-30 pour PI-CAI) un déséquilibre
    résiduel est possible. Cette fonction rend le déséquilibre visible.

    Parameters
    ----------
    labels:
        Liste de labels binaires (0 ou 1) pour les patients du fold.
    fold:
        Numéro du fold (1-based, pour l'affichage).
    split:
        "train" ou "val" — indique quelle partie du fold est décrite.
    """
    n_total = len(labels)
    n_pos = sum(labels)
    n_neg = n_total - n_pos
    ratio = n_neg / max(n_pos, 1)
    print(
        f"  [fold {fold} | {split}] "
        f"total={n_total}, label=0: {n_neg}, label=1: {n_pos}, "
        f"ratio={ratio:.2f}:1"
    )
