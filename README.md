#  Drug-Conditioned Gene Expression Modeling with scGPT

This repository implements a  perturbation prediction model built on top of scGPT, using adapter-based conditioning with molecular embeddings.

The model predicts how a drug perturbation affects gene expression and associated biological signals.

---

##  Overview

We extend scGPT with:

-  Drug-conditioned adapters (FiLM-based)
-  Multi-task learning
-  Gene + expression as primary inputs

### Tasks

- `expr` — Gene expression prediction (regression)
- `direction` — Direction classification (down / neutral / up)
- `go` — Gene Ontology classification
- `moa_broad` — Mechanism of Action classification

---

##  Architecture

```
Gene IDs + Expression Values
            ↓
   Transformer Backbone
            ↓
   Adapter Layers (drug-conditioned)
            ↓
     Shared Representation
            ↓
     ┌──────────────┬──────────────┬──────────────┬──────────────┐
     │ Expression   │ Direction    │ GO           │ MOA Broad    │
     └──────────────┴──────────────┴──────────────┴──────────────┘
```



## Repository Structure

```
DCA/
├── scgpt/
├── perturbation/
├── scripts/
├── configs/
```

---

## ⚙️ Installation

```bash
conda create -n dca python=3.10
conda activate dca

pip install torch scanpy pandas numpy tqdm
```

---

##  Data

The model expects:

- `.h5ad` file (AnnData)
- Preprocessed datasets:
  - `averaged_df_train.pkl`
  - `averaged_df_test.pkl`
- Drug embeddings (e.g., MolFormer)

---

## Training

```bash
export PYTHONPATH=.
python perturbation/scripts/train_perturbation.py --config configs/perturbation.yaml
```



