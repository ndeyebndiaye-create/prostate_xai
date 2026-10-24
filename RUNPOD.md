

Ce guide cible le template officiel **RunPod PyTorch 2.8.0** et un GPU
**RTX 4090 24 Go**. Il n'installe rien, ne crée aucun pod et ne lance pas
d'entraînement automatiquement.

## 1. Ce que le code entraîne réellement

Les entrées existantes sont :

```bash
python -m src.training.train_unimodal_baseline --config configs/phase1_t2w.yaml
python -m src.training.train_unimodal_baseline --config configs/phase1_adc.yaml
python -m src.training.train_unimodal_baseline --config configs/phase1_hbv.yaml
python -m src.training.train_fusion_baseline --config configs/phase3_fusion_simple.yaml
```

Dans ce dépôt, la séquence DWI high-b-value est appelée `hbv`, pas `dwi`.
La fusion disponible ici est une concaténation simple de canaux, pas
l'architecture future à cross-attention.

Chaque configuration prévoit **5 folds et 30 epochs par fold** : une commande
complète entraîne donc 150 fold-epochs. Les quatre commandes ci-dessus
représentent jusqu'à 600 fold-epochs. Avant une expérience scientifique, il
faut aussi résoudre la divergence existante du seuil de labels : le code et les
tests considèrent `ISUP >= 2` positif, tandis que le README décrit `ISUP >= 3`.
Ce guide ne change pas ce comportement.

Les entrées sont des volumes 3D `.mha`, normalisés puis recadrés/complétés en
`(96, 96, 16)` par défaut. Le code n'utilise pas des images 2D `224 x 224`.
Le `batch_size` par défaut est 8.

## 2. Créer le pod et vérifier le stockage

Dans RunPod, choisir le template **RunPod PyTorch 2.8.0**
(`runpod-torch-v280`), le GPU RTX 4090, activer SSH et noter le tarif affiché
pour l'offre et la région choisies. Aucun port HTTP n'est nécessaire pour un
entraînement en ligne de commande.

Vérifier à la création la distinction entre **Container Disk**, **Volume Disk**
et éventuel **Network Volume**. Le Container Disk ne doit pas être considéré
comme une sauvegarde. Un network volume monté sur `/workspace` remplace le
montage `/workspace` habituel et est facturé séparément. La taille réelle des
données n'est pas connue depuis ce dépôt : mesurer après transfert avant de
commencer l'entraînement.

Dans le pod :

```bash
df -h / /workspace
```

Si le dataset dépasse l'espace local disponible, arrêter avant l'entraînement
et choisir un stockage persistant assez grand, dans le même datacenter que le
pod. Si les données restent sur le disque du pod, copier les résultats hors du
pod avant toute suppression.

## 3. Transférer le code sans secrets ni gros fichiers inutiles

Depuis un terminal local Ubuntu, dans le dossier `prostate_xai` du projet,
transférer le code sur le pod. Remplacer les valeurs entre chevrons par les
informations SSH affichées par RunPod :

```bash
rsync -av \
  --exclude='.env' \
  --exclude='.venv/' \
  --exclude='__pycache__/' \
  --exclude='.pytest_cache/' \
  --exclude='checkpoints/' \
  --exclude='results/' \
  --exclude='*.mha' \
  -e 'ssh -p <PORT_SSH> -i <CLE_SSH>' \
  ./ root@<IP_OU_HOTE_SSH>:/workspace/prostate_xai/
```

Cela transfère les scripts, configurations, dépendances et tests sans envoyer
le dataset, les anciens checkpoints, les résultats locaux, l'environnement
virtuel ni le `.env`. Ne jamais copier ici de clé API, mot de passe ou fichier
de configuration rclone.

## 4. Récupérer le dataset depuis Google Drive

Google Drive Desktop n'est pas automatiquement monté sur RunPod. Une option est
de configurer `rclone` sur le pod avec une autorisation Google Drive valide,
sans afficher ni partager le fichier de configuration ou les jetons. Le projet
utilise déjà le nom de remote `gdrive_thesis` pour son mode de lecture rclone.

Après configuration, vérifier d'abord que le dossier source est le bon :

```bash
rclone lsf 'gdrive_thesis:thesis_data/picai/images/' --dirs-only | head
```

Puis copier les données dans le stockage retenu :

```bash
mkdir -p /workspace/data
rclone copy 'gdrive_thesis:thesis_data/picai' /workspace/data/picai --progress
du -sh /workspace/data/picai
```

La copie doit préserver cette structure :

```text
/workspace/data/picai/images/<patient_id>/*_0000.mha
/workspace/data/picai/images/<patient_id>/*_0001.mha
/workspace/data/picai/images/<patient_id>/*_0002.mha
/workspace/data/picai/picai_labels/clinical_information/marksheet.csv
```

Le code attend `_0000` = T2W, `_0001` = ADC et `_0002` = HBV. Le dataset doit
contenir les trois modalités pour qu'un patient soit retenu comme complet,
même pour un entraînement unimodal. Le marksheet doit être à l'emplacement
ci-dessus : `scripts/download_metadata.py` utilise actuellement un autre
emplacement.

Après avoir confirmé la structure, créer un `.env` propre au pod à la racine
du projet :

```bash
cd /workspace/prostate_xai
printf 'DATA_ROOT=/workspace/data\n' > .env
```

`DATA_ROOT` est un chemin absolu dans l'environnement RunPod. Aucun changement
de chemin dans le code source n'est nécessaire.

## 5. Vérifier PyTorch, CUDA et le GPU

Sur le pod, vérifier d'abord le périphérique :

```bash
nvidia-smi
```

Puis confirmer que le PyTorch CUDA du template est visible par l'interpréteur
du système :

```bash
python3 -c 'import torch; print("torch:", torch.__version__); print("CUDA runtime:", torch.version.cuda); print("CUDA disponible:", torch.cuda.is_available()); print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "aucun"); assert torch.cuda.is_available()'
```

Le template PyTorch contient déjà PyTorch avec CUDA. Vérifier cette étape avant
d'installer les autres dépendances et ne pas créer un venv ordinaire qui
masquerait ce PyTorch. Si des dépendances du projet manquent, l'installation
dans le Python du template se fait, après accord, par :

```bash
cd /workspace/prostate_xai
python3 -m pip install --break-system-packages -r requirements.txt
```

Le requirement `torch>=2.1.0` est satisfait par PyTorch 2.8.0. Refaire ensuite
la vérification CUDA ci-dessus.

## 6. Tester quelques vrais batches, sans lancer la CV complète

Ce test lit les patients du dataset, exécute deux batches et vérifie une passe
avant/arrière sur GPU. Il ne lance ni les 5 folds ni les 30 epochs :

```bash
cd /workspace/prostate_xai
python3 - <<'PY'
import torch
from torch.utils.data import DataLoader
from src.config import DATA_ROOT
from src.data.picai_paths import find_complete_patients, load_marksheet
from src.data.unimodal_dataset import PicaiUnimodalDataset
from src.models.simple_cnn3d import SimpleCNN3D

assert torch.cuda.is_available(), "CUDA n'est pas disponible"
complete_ids, _ = find_complete_patients(DATA_ROOT)
marksheet = load_marksheet(DATA_ROOT, complete_ids=complete_ids)
patient_ids = marksheet["patient_id"].tolist()
dataset = PicaiUnimodalDataset(DATA_ROOT, patient_ids, marksheet, modality="t2w")
loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
model = SimpleCNN3D(in_channels=1, n_classes=2).cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scaler = torch.amp.GradScaler("cuda")

for batch_index, (images, labels) in enumerate(loader):
    images, labels = images.cuda(), labels.cuda()
    optimizer.zero_grad()
    with torch.amp.autocast("cuda"):
        loss = torch.nn.functional.cross_entropy(model(images), labels)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    print(f"batch={batch_index + 1}, shape={tuple(images.shape)}, loss={loss.item():.4f}")
    if batch_index == 1:
        break
PY
```

La shape attendue est de la forme `(batch, 1, 16, 96, 96)` pour T2W. Ce test
peut prendre du temps car les volumes sont lus et prétraités à la demande.

## 7. Mesurer avant de lancer 30 epochs

Ne pas lancer les quatre CV complètes d'un coup. Après validation du smoke test
et du seuil de labels, effectuer manuellement un petit benchmark, par exemple
une seule modalité avec deux folds et une epoch :

```bash
cd /workspace/prostate_xai
/usr/bin/time -p python3 -m src.training.train_unimodal_baseline \
  --config configs/phase1_t2w.yaml --n_folds 2 --n_epochs 1
```

Ce benchmark entraîne réellement deux folds, mais pas l'expérience complète.
Relever les temps par epoch affichés dans le journal. Pour estimer les quatre
expériences, si `T` est la durée moyenne en secondes d'une epoch sur un fold :

```text
temps GPU estimé ≈ 600 × T / 3600 heures
coût GPU estimé  ≈ tarif RTX 4090 affiché ($/h) × 600 × T / 3600
```

Cette estimation doit inclure une marge pour le téléchargement, l'installation,
les chargements de données et les sauvegardes. Vérifier le prix RTX 4090 affiché
dans la Console RunPod juste avant le lancement; les tarifs varient selon l'offre
et la région.

## 8. Résultats et sauvegarde avant suppression

Le code écrit automatiquement sous `/workspace` :

```text
/workspace/checkpoints/<experience>/fold_<N>/last.pt
/workspace/checkpoints/<experience>/fold_<N>/best.pt
/workspace/logs/<experience>_fold<N>.csv
/workspace/results/baselines.csv
```

`last.pt` contient le modèle, l'optimizer, le scaler et l'epoch afin de
reprendre le fold après une interruption. `best.pt` garde le meilleur AUROC
de validation atteint. Les journaux CSV enregistrent la loss d'entraînement
et l'AUROC de validation à chaque epoch. Le CSV des résultats contient les
métriques des folds et une ligne de résumé.

## 9. VS Code Remote-SSH depuis Ubuntu

Installer dans VS Code les extensions **Remote - SSH**, **Python** et
**Jupyter** (Jupyter est optionnel si les notebooks ne sont pas utilisés).
Ajouter au fichier local `~/.ssh/config` :

```ssh-config
Host runpod-prostate
    HostName <IP_DU_POD>
    Port <PORT_SSH>
    User root
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes
```

Remplacer les placeholders par l'IP et le port SSH fournis par RunPod; utiliser
le chemin de la clé privée effectivement enregistrée sur Ubuntu. Dans VS Code,
ouvrir la palette de commandes, choisir **Remote-SSH: Connect to Host...**,
puis `runpod-prostate`. Dans la fenêtre distante, choisir **File > Open Folder**
et ouvrir `/workspace/prostate_xai`. Avec l'extension Python, sélectionner
l'interpréteur du Pod indiqué par `command -v python3` (souvent `/usr/bin/python3`).
Garder le Python PyTorch du template pour conserver la version CUDA fournie.

Configurer les dépendances manquantes depuis le terminal distant, sans créer
un venv qui masquerait PyTorch :

```bash
cd /workspace/prostate_xai
python3 -m pip install --break-system-packages -r requirements.txt
```

Vérifier ensuite `nvidia-smi` et `python3 -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))'`.
Pour RTX 4090, le modèle et les hyperparamètres restent ceux des YAML; vérifier
le batch de test avant toute modification mémoire.

## 10. Lancement résistant aux coupures et récupération sur Ubuntu

Le contrôleur `scripts/runpod_train_and_stop.sh` se lance depuis le PC Ubuntu.
Il démarre les quatre entraînements dans une session tmux distante, synchronise
les sorties toutes les 5 minutes, fait une copie finale, vérifie un `best.pt`,
puis appelle `runpodctl stop pod` uniquement si les entraînements ont réussi et
la sauvegarde a été vérifiée. Sur le PC, installer/configurer `runpodctl` et
remplacer les placeholders en haut du script :

```bash
cd /chemin/local/vers/prostate_xai
export RUNPOD_POD_ID="<ID_DU_POD>"
bash scripts/runpod_train_and_stop.sh
```

Le script d'entraînement distant est `scripts/runpod_train.sh`; il écrit un
statut dans `/workspace/TRAINING_STATUS` et ne stoppe pas lui-même le Pod.
Le Pod doit avoir `tmux` installé (`command -v tmux` pour vérifier).
Le contrôleur local garde la connexion de supervision; l'entraînement continue
dans `tmux` même si cette connexion est coupée. Pour suivre/rejoindre la session,
connectez-vous au Pod et exécutez `tmux attach -t prostate-train` (détacher :
`Ctrl+b`, puis `d`). En cas d'arrêt/interruption du contrôleur, reprendre la
synchronisation depuis le PC avec le script indépendant ci-dessous; n'arrêtez
pas le Pod avant d'avoir copié et vérifié les sorties.

Pour lancer seulement une synchronisation périodique depuis un autre terminal
local Ubuntu, renseigner les variables au début de `scripts/sync_pull.sh`, puis :

```bash
bash scripts/sync_pull.sh --watch
```

Le mode `--watch` resynchronise checkpoints, logs et résultats toutes les
5 minutes vers `./runpod_backup/`. Sans `--watch`, la synchronisation ne se
fait qu'une fois. Après copie, vérifier un modèle (adapter le chemin si besoin) :

```bash
python3 scripts/check_backup.py \
  runpod_backup/checkpoints/fusion_simple/fold_1/best.pt
```

Checklist avant **Terminate** :

- [ ] `runpod_backup/` contient les checkpoints, logs et résultats attendus.
- [ ] `check_backup.py` confirme qu'un `best.pt` se charge sur CPU.
- [ ] Le code et les configurations ont été poussés sur GitHub.
- [ ] Aucun entraînement n'est encore actif; les résultats sont copiés hors du Pod.

`requirements.txt` contient les dépendances Python du projet; le PyTorch/CUDA
est fourni par l'image RunPod. Ne pas y inscrire de clés, tokens ou mots de passe.
