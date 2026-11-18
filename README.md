# Confidently Wrong in the Field

Code and results for the study of how quantization affects calibration and
selective prediction of crop-disease models under laboratory-to-field shift
(PlantVillage -> PlantDoc).

**Status:** pilot stage. Results in `results/` are added as runs finish.

## Setup

```bash
conda create -n crop python=3.11 -y && conda activate crop
# install torch + torchvision with the CUDA command from https://pytorch.org
pip install -r requirements.txt
```

Download the datasets as described in `data/README.md`.

## Pilot: FP32 vs INT8, lab vs field

**Easiest:** open `pilot.ipynb` in Jupyter or Colab and run the cells top to bottom. It downloads the data, runs a quick test, the full pilot, shows the verdict, and copies results into `results/`.

Or from the command line:

```bash
python pilot.py --pv data/plantvillage/color --pd data/PlantDoc-Dataset --out runs/pilot_seed0 --seed 0 --workers 2
```

This fine-tunes MobileNetV3-Large on the classes shared by both datasets,
converts it to INT8 with static post-training quantization, and reports
accuracy, ECE, AURC, error-detection AUROC, the flip rate by confidence bin,
and the shift gap with 95% bootstrap confidence intervals.

## Saving a run to the repository

```bash
mkdir -p results/pilot_seed0
cp runs/pilot_seed0/{results.json,pilot.png,classes.json,logits.npz} results/pilot_seed0/
echo "python pilot.py --pv ... --pd ... --seed 0" > results/pilot_seed0/command.txt
git add results/pilot_seed0 && git commit -m "Add pilot results, seed 0" && git push
```

## Repository layout

| Path | Contents |
| --- | --- |
| `pilot.py` | Pilot experiment (train, quantize, evaluate) |
| `pilot.ipynb` | Notebook that runs the whole pilot end to end |
| `results/` | Metrics, plots and logits for each finished run |
| `data/` | Dataset download instructions (data itself is not committed) |

## Known limits of the pilot

- PlantVillage is split at random, not by perceptual-hash groups.
- Quantization uses PyTorch eager-mode PTQ on CPU, not the on-device runtime.
- The PlantVillage-PlantDoc class mapping at the top of `pilot.py` includes a few approximate pairs.
