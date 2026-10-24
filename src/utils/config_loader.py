"""
Chargeur de configuration YAML — Option A (coexistence argparse + YAML optionnel).

Principe de priorité (du plus bas au plus haut) :
  1. Valeurs par défaut argparse (hardcodées dans le script)
  2. Valeurs du fichier YAML (si --config est fourni)
  3. Arguments CLI explicites (passés par l'utilisateur sur la ligne de commande)

Exemple concret :
  YAML: n_epochs: 50
  CLI: (rien)
  → n_epochs = 50   (YAML prend le dessus sur le défaut argparse de 30)

  YAML: n_epochs: 50
  CLI: --n_epochs 10
  → n_epochs = 10   (CLI prend le dessus sur le YAML)

  (aucun YAML)
  CLI: (rien)
  → n_epochs = 30   (défaut argparse)

Ce comportement est implémenté via `parser.set_defaults()` après la lecture
du YAML : les valeurs YAML deviennent les nouveaux defaults de l'argparse,
puis argparse parse normalement — les args CLI explicites écrasent ces defaults.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def load_yaml_config(yaml_path: Path) -> dict:
    """
    Charge un fichier YAML et retourne son contenu sous forme de dict plat.

    Parameters
    ----------
    yaml_path:
        Chemin vers le fichier YAML de configuration.

    Returns
    -------
    dict avec les clés/valeurs du YAML. Retourne {} si le fichier est vide.

    Raises
    ------
    FileNotFoundError si le fichier n'existe pas.
    """
    import yaml  # import local pour ne pas imposer pyyaml si config_loader n'est pas utilisé

    if not yaml_path.exists():
        raise FileNotFoundError(
            f"Fichier de config introuvable : {yaml_path}\n"
            f"Vérifier le chemin ou créer le fichier depuis configs/."
        )
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    return cfg or {}


def apply_yaml_to_parser(parser, yaml_path: Path | None) -> None:
    """
    Applique les valeurs d'un YAML comme nouveaux defaults d'un ArgumentParser.

    Doit être appelé après avoir défini tous les arguments du parser,
    mais AVANT d'appeler parser.parse_args(). Les clés YAML inconnues
    (pas dans le parser) sont silencieusement ignorées.

    Parameters
    ----------
    parser:
        Instance argparse.ArgumentParser avec tous ses arguments définis.
    yaml_path:
        Chemin vers le fichier YAML, ou None si pas de config (mode argparse pur).
    """
    if yaml_path is None:
        return

    cfg = load_yaml_config(yaml_path)
    # Récupère les noms de paramètres valides du parser pour ne pas injecter
    # des clés YAML qui ne correspondent à aucun argument.
    valid_dests = {action.dest for action in parser._actions}
    filtered = {k: v for k, v in cfg.items() if k in valid_dests}
    parser.set_defaults(**filtered)


def get_git_hash() -> str:
    """
    Retourne le hash court du commit Git HEAD courant.

    Utilisé pour tracer quel état exact du code a produit un résultat dans
    /workspace/results/baselines.csv. Si la commande git échoue (repo non-git ou git
    non installé), retourne "unknown" sans planter.

    Returns
    -------
    Hash court (7 caractères), ex. "bf67ce3", ou "unknown".
    """
    try:
        project_root = Path(__file__).resolve().parents[2]
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=project_root,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"
