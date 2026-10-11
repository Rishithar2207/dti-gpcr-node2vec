# Deep Learning Methods for Drug–Target Interaction Prediction

Node2Vec embeddings over a similarity-augmented bipartite drug–target graph, with a stacked
ensemble classifier for link prediction on the GPCR benchmark dataset.

MSc Data Science individual project, King's College London (7CCSMPRJ), supervised by Dr Sophia Tsoka.
Full dissertation: [`report/`](report/).

## Problem

Predicting which drugs bind which protein targets. Experimental screening is slow and expensive, so
the task is framed as **link prediction on a bipartite graph**. The hard part is sparsity: only
~8.5% of possible drug–target pairs are known interactions.

## Dataset

Yamanishi et al. GPCR benchmark — three tab-separated matrices:

| File | Contents | Shape |
|---|---|---|
| `gpcr_admat_dgc.txt` | Binary interaction matrix | 95 targets × 223 drugs |
| `gpcr_simmat_dc.txt` | Drug–drug similarity (molecular fingerprints) | 223 × 223 |
| `gpcr_simmat_dg.txt` | Target–target similarity (sequence alignment) | 95 × 95 |

- **1,795 known interactions out of 21,185 possible pairs → 91.5% sparse**
- Graph after top-5 similarity edge augmentation: **318 nodes, 1,795 edges**
  (635 drug–target, 605 drug–drug, 555 target–target)

The data files are not committed here. Download them from the
[Yamanishi benchmark release](https://web.kuicr.kyoto-u.ac.jp/supp/yoshi/drugtarget/) and drop the
three `.txt` files into `data/`.

## Method

1. **Graph construction** — bipartite drug/target graph from the interaction matrix, enriched with
   top-5 most-similar neighbours per drug and per target so the walks can traverse chemical and
   sequence similarity, not just known interactions.
2. **Node2Vec embeddings** — biased random walks, tuned over p/q/dimension/walk length.
   Best: `p=0.5, q=2.0, dim=64`.
3. **Feature vectors** — concatenated drug ‖ target embeddings, standardised.
4. **Negative sampling** — stratified balanced sampling, plus a hard-negative-mining variant that
   picks the non-interacting pairs most similar to true positives.
5. **Classifiers** — Logistic Regression, Random Forest, XGBoost, MLP, stacked with Logistic
   Regression as meta-learner.
6. **Evaluation** — ROC-AUC, PR-AUC, F1, MCC under stratified cross-validation, plus error
   analysis, ablations, and paired t-tests against baselines.

## Results

| Model | ROC-AUC |
|---|---|
| **Node2Vec + stacked ensemble** | **0.9566 ± 0.0147** |
| Node2Vec + hard negative mining | 0.8950 |
| SVD matrix factorisation baseline | 0.6335 |
| DeepWalk baseline | 0.6253 |

PR-AUC 0.9471 ± 0.0211, F1 0.9062, MCC 0.8111. Improvement over baselines significant at
p = 0.0321 (two-tailed paired t-test). Error analysis: 54 false positives, 42 false negatives.

PR-AUC is reported alongside ROC-AUC because the class skew makes ROC-AUC optimistic on its own.

## Running it

```bash
pip install -r requirements.txt
# place the three GPCR .txt files in data/, then:
cd data && python ../src/dti_pipeline.py
```

The script reads the data files from the working directory and writes `node_embeddings.csv`,
`optimal_node_embeddings.csv`, `dti_samples.csv` and `dti_graph.gexf` beside them.

Runtime: Node2Vec training measured at ~1,600 nodes/second on 2 CPU cores with 12.67 GB RAM. No
GPU required.

## Repo layout

```
src/dti_pipeline.py   end-to-end pipeline: load → graph → embed → sample → train → evaluate → plot
data/                 GPCR benchmark files (not committed)
results/              generated embeddings, graphs, figures
```

## Known limitations

- The pipeline is a single linear script exported from a notebook, not an importable package, and
  has no automated test suite.
- Paths are relative to the working directory rather than configurable.
- `evaluate_external_dataset()` supports the Nuclear Receptor and Ion Channel benchmarks but was
  not run as part of the submitted results.
- Predictions are computational only and need wet-lab validation before any experimental or
  clinical use.
