<div align="center">
<img src="figs/icon.png" border="0" width=600px;/>
</div>

<div align="center">
    <a href="https://arxiv.org/pdf/2505.21366"><img src="https://img.shields.io/static/v1?label=ICLR'26&message=Paper&color=red"></a>
    <a href="https://planetalign.readthedocs.io/en/latest/"><img src="https://img.shields.io/badge/Documentation-PlanetAlign-blueviolet"></a>
    <a href="https://github.com/yq-leo/PlanetAlign/blob/main/LICENSE.txt"><img src="https://badgen.net/github/license/yq-leo/PlanetAlign?color=green"></a>
</div>

# PlanetAlign + Many-to-Many Network Alignment

This repository is a research fork of [**PlanetAlign**](https://github.com/yq-leo/PlanetAlign) — a comprehensive
Python library for network alignment (NA) — extended with a full pipeline for **many-to-many (M2M)
network alignment**: aligning *entities* that each correspond to a *set* of nodes on either side, rather than
the one-to-one node correspondences the original library (and most of the NA literature) assume.

The M2M work is the content of this branch. The upstream library and its APIs are unchanged and documented
below; everything M2M lives in the `m2m_*` modules, the `scripts/` runners, and the `docs/m2m_*` design notes.

---

## TL;DR — what this fork adds

**The problem.** Every one-to-one NA method fails on many-to-many data, and they fail for what looks like four
different reasons (permutation-locking OT marginals, row-argmax readouts, contrastive losses that repel
co-referent nodes, an injective evaluation protocol). We show these are **one failure in four disguises: the
node-level exclusivity prior is applied at the wrong level.**

**The fix.** One-to-one exclusivity is *correct again at the quotient level* — a ground-truth entity pairs one
source node-group with one target node-group. So the right learning object is an **equivalence relation per
graph plus a one-to-one matching between the quotient graphs**, and the grouping evidence is *already inside
any trained aligner's similarity matrix* `S`: co-referent nodes map to the same region of the other graph, so
their rows of `S` are near-collinear. No new module is needed — only a readout aimed at the right object.

**The final system:**

```
                 PARROT  (position-aware regularized-OT base aligner; deterministic, seconds)
                    │  S  (node similarity matrix)
      arbitrate_sharpen   (raw vs softmax(S/T), chosen blind on training anchors)
                    │
             QuotientDecode  (the M2M readout)
   ┌────────────────┴─────────────────────────────────────────────┐
   │ Otsu-adaptive profile clustering + average-linkage anti-chain  │
   │ RMT null-calibrated threshold + gated global candidates        │
   │ anchor-arbitrated evidence selection (chance-corrected + Occam) │
   │ rectangular Hungarian on the quotient  +  alternating refine    │
   │ evidence-conditional outlier eviction                           │
   └────────────────────────────────────────────────────────────────┘
                    │
             entity map  →  blind ACS / MSF1 / MicroF1 / M2M-SGS / M2M-EGS
```

**One principle spans four layers** — grouping evidence, merge thresholds, sharpening temperature, and base
aligner are *all* selected by the same blind signal: the chance-corrected agreement of the **training anchors**
lifted to group level (the ground-truth entities never enter the prediction path).

### Results (blind MicroF1, corrected supervision protocol, 4 seeds; PARROT is deterministic → std = 0)

| Dataset | Start (best 1-1 readout) | **Final system** | Note |
|---|---|---|---|
| douban | 0.318 | **0.730** | 2.3× the original best readout |
| cora | 0.893 | **0.981** (→ 0.992 with the ensemble, anchor-selectable) | attribute ceiling was 0.991 |
| airport | 0.27 | **0.798** | |
| pems08 | ~0.29 | **0.601** | ensemble 0.643 is an honest residual (anchor saturation; docs §5.13) |
| ppi | — | **0.930** | |
| arenas | 0.003 | **0.851** | held-out 1-1-entity Hits@1 = 0.981 |
| phone-email | 0.000 | **0.424** | blind readout **beats** the GT-leaking adapter (0.230) |
| italy | 0.000 | **0.372** | same — weak-signal amplification via group pooling |
| foursquare | 0.051 | **0.404** | same |

One base (PARROT) + one readout, **zero per-dataset tuning**, all nine datasets non-trivial — four of them were
previously (and wrongly) declared infeasible. See [`docs/m2m_research_summary_for_paper.md`](docs/m2m_research_summary_for_paper.md)
for the full retrospective.

---

## Repository map (M2M)

### Library modules (`PlanetAlign/`)
| File | Role |
|---|---|
| `m2m.py` | M2M interface: `EntityMap`, GT alignment, `use_full_anchor_supervision`, metric glue |
| `m2m_blind.py` | Blind protocol: group discovery from `S` profiles, entity decode, blind evaluation |
| `m2m_quotient.py` | **QuotientDecode** — the core M2M readout (all components above), plus `evaluate_quotient_blind` |
| `m2m_base.py` | Base-aligner axis: `train_base_S`, `arbitrate_sharpen`, `arbitrate_base` (PARROT and others) |
| `m2m_contrastive.py` | `JOENAPC` — training-side fix: symmetry-broken inputs + uniformity contrastive + group-level Sinkhorn marginals |
| `utils/many2many_builder.py` | M2M benchmark builder (node-splitting) with a hard cross-set leakage invariant |
| `metrics/` | ACS, MSF1, MicroF1, M2M-SGS, M2M-EGS |

### Scripts (`scripts/`)
**Formal experiment pipeline**
| Script | Purpose |
|---|---|
| `build_m2m_benchmarks.py` | Build the M2M datasets from the 1-1 sources (leakage-guarded) |
| `run_quotient_compare.py` | Main runner: readouts + ablations over a frozen `S`; `--base PARROT` / `--model {joena,joena-pc,auto,ensemble}` |
| `probe_base_aligners.py` | Probe base aligners (IsoRank/FINAL/PARROT/REGAL/BRIGHT/…) on a dataset |
| `validate_base_selection.py` | The base-as-anchor-arbitrable-axis experiment (docs §5.13) |
| `compare_parrot_vs_ensemble.py` | PARROT vs JOENA-family ensemble head-to-head |
| `aggregate_unified_grid.py` | Aggregate the multi-seed grid into the decision table |
| `sweep_readout_variants.py` | Paired multi-seed sweep over readout variants (docs §5.10) |
| `run_m2m_experiments.py` | Full multi-algorithm M2M benchmark runner |

**Diagnostics**
| Script | Purpose |
|---|---|
| `diagnose_m2m_failure.py` | Evidence behind "why one-to-one fails" |
| `diagnose_rmt.py` | Random-matrix-theory spectral diagnostics of `S` (docs §5.5) |

### Tests (`tests/`) — 52 tests, each guarding a live invariant
`test_m2m_quotient.py` (one assertion per design claim), `test_m2m_base.py` (base axis + sharpen arbitration),
`test_m2m_blind.py`, `test_m2m_interface.py`, `test_many_to_many_metrics.py`, `test_m2m_failure_modes.py`,
`test_m2m_builder_leakage.py` (data-integrity regression).

```bash
python -m unittest discover -s tests -p "test_*.py"
```

### Design docs (`docs/`)
| Doc | Content |
|---|---|
| `m2m_why_one_to_one_fails.md` | Failure diagnosis — every claim backed by experiment **and** the source paper's theory |
| `m2m_quotient_align_design.md` | The design + experiment log (§5.x: RMT, JOENA-PC, protocol correction, base unification …) |
| `m2m_research_summary_for_paper.md` | Paper-ready retrospective: contributions, negative-results list, theory hooks, figure ideas |
| `m2m_core_contradiction.pptx` | The core-contradiction logic deck |

---

## Quickstart (M2M)

```bash
pip install -e .            # installs PlanetAlign + POT, torch, torch_geometric, …

# 1) Build the M2M benchmarks (writes data/m2m_no_overlap/ and data/m2m_overlap_0.05/)
python scripts/build_m2m_benchmarks.py

# 2) Run the unified system (PARROT base + QuotientDecode) on one dataset
python scripts/run_quotient_compare.py \
    --root data/m2m_no_overlap --dataset ppi_m2m --base PARROT

# 3) Reproduce the base-selection grid (docs §5.13)
python scripts/validate_base_selection.py --seeds 42 0 1 2 --bases PARROT
python scripts/compare_parrot_vs_ensemble.py --seeds 42 0 1 2
python scripts/aggregate_unified_grid.py
```

Programmatic use:

```python
import json
from PlanetAlign.data import Dataset
from PlanetAlign.m2m import use_full_anchor_supervision
from PlanetAlign.m2m_base import train_base_S, arbitrate_sharpen
from PlanetAlign.m2m_quotient import evaluate_quotient_blind
from PlanetAlign.utils import get_anchor_pairs

ds = Dataset(root="data/m2m_no_overlap", name="ppi_m2m", train_ratio=0.2, seed=42)
use_full_anchor_supervision(ds)                        # anchors = original 20% train split
g_src, g_tgt = ds.pyg_graphs[0], ds.pyg_graphs[1]
anchors = get_anchor_pairs(ds.train_data, 0, 1)
gt = json.load(open("data/m2m_no_overlap/ppi_m2m_gt_many2many.json"))["entities"]

S = train_base_S("PARROT", ds, seed=42)                # base similarity
S, T, _ = arbitrate_sharpen(S, g_src, g_tgt, anchors)  # blind sharpening choice
scores = evaluate_quotient_blind(S, gt, g_src, g_tgt,   # blind M2M readout
                                 metrics=["MicroF1", "ACS", "MSF1"], anchors=anchors)
print(scores)
```

### M2M datasets

18 datasets are provided in two overlap regimes, `data/m2m_no_overlap/` and `data/m2m_overlap_0.05/`:
`douban, cora, airport, pems08, ppi, arenas, phone-email, italy, foursquare-twitter, acm-dblp, arxiv,
flickr-lastfm, flickr-myspace, ggi, sacchcere, dbp15k_{zh,ja,fr}-en`. The current experiment grid uses the
nine with `max(n) ≲ 6k` (both PARROT and JOENA are dense `O(n²)`); the eight large graphs (`n ≈ 24k`) await a
sparse base. Each dataset ships a `<name>_gt_many2many.json` (entity ground truth) and a build manifest.

> **Note.** The datasets under `data/` are generated artifacts and are git-ignored; regenerate them with
> `scripts/build_m2m_benchmarks.py`.

---

## Upstream PlanetAlign library

The base library (consistency / embedding / OT-based NA methods, standardized datasets and metrics, robustness
and sensitivity utilities) is unchanged. A minimal one-to-one example:

```python
from PlanetAlign.datasets import Douban
from PlanetAlign.algorithms import FINAL

data = Douban(root="./data")
model = FINAL()
model.train(data, gids=[0, 1])
print("Evaluation Results:", model.test(data, gids=[0, 1]))
```

### Installation

```bash
git clone <this-fork> && cd PlanetAlign
pip install -e .
```

### Documentation

Upstream docs and tutorial: https://planetalign.readthedocs.io/en/latest/

### Citation

```bibtex
@article{yu2025planetalign,
  title={PLANETALIGN: A Comprehensive Python Library for Benchmarking Network Alignment},
  author={Yu, Qi and Zeng, Zhichen and Yan, Yuchen and Liu, Zhining and Jing, Baoyu and Qiu, Ruizhong and Azad, Ariful and Tong, Hanghang},
  journal={arXiv preprint arXiv:2505.21366},
  year={2025}
}
```

The many-to-many extension in this fork is research code; if it is useful in your work, please also reference
this repository and the design notes in `docs/`.
