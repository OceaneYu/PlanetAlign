# One-to-One Baseline Calibration Master

This record freezes the completed one-to-one calibration audit that precedes the M2M Master Table. It does not contain any newly rerun experiment, hyperparameter search, or inferred missing result.

## Record metadata

- Generated: 2026-08-17
- Repository HEAD at audit: `f72d2e0785dee4c608068dba4168689f8b2a6f61`
- Scope: the 14 one-to-one methods evaluated by PLANETALIGN, plus the prepared GADL baseline
- Reference protocol: 20% training anchors, ordinary-argsort ranking, bidirectional mean Hits@K/MRR unless a row explicitly states otherwise
- PLANETALIGN reference aggregation: five dataset splits and five runs per split; hyperparameter search used the paper's fixed budget
- Current shared evaluator caveat: `BaseModel.test` defaults to optimistic tie handling, whereas the PLANETALIGN paper results used ordinary argsort
- Result provenance policy: reuse the PLANETALIGN tables and reliable existing logs; completed terminal-only sanity results are explicitly marked; missing values remain missing

## One-to-One Calibration Master Table

`Gap` is Current minus Paper for Hits@1, expressed in percentage points. PLANETALIGN Table 5 rows compare the result of the baseline's official implementation with the corresponding PLANETALIGN implementation under the paper's unified protocol.

| Method | Reference | Dataset | Metric | Paper Result | Current Result | Gap | Protocol Difference | Calibration Status | M2M Admission |
|---|---|---|---|---:|---:|---:|---|---|---|
| IsoRank | PLANETALIGN Table 8 | PeMS08 | Bidirectional ordinary Hits@1/Hits@10/MRR | 0.2537 / 0.7353 / 0.4197 | 0.253676 / 0.735294 / 0.419663 | approximately 0.00 pp | Completed audit run used `data`, 20% anchors, seed 0, plain graph, `alpha=0.4`, 100 epochs, ordinary argsort. The same matrix scores 0.264706 Hits@1 under the current optimistic default. | `CALIBRATED_WITH_PROTOCOL_GAP` | `READY_FOR_M2M` |
| FINAL | PLANETALIGN Table 8 | PeMS08 | Bidirectional Hits@1 | 0.1985 | 0.209559 | +1.11 pp | Existing current result is seed 0, plain graph, `alpha=0.5`, float64, optimistic tie; the paper used ordinary argsort and aggregated splits/runs. Phone-Email and Italy results also closely match the paper. | `CALIBRATED_WITH_PROTOCOL_GAP` | `READY_FOR_M2M` |
| IONE | PLANETALIGN Table 8 | PeMS08 | Bidirectional ordinary Hits@1 | 0.3824 +/- 0.0252 | missing | missing | No reliable current one-to-one result. The implementation defaults to 100 outer epochs times 100,000 base updates; no shortened run is treated as calibration evidence. | `UNCHECKED` | `PENDING_CALIBRATION` |
| REGAL | PLANETALIGN Table 5 | 18-dataset average | Bidirectional Hits@1 | 0.079 official | 0.080 PLANETALIGN implementation | +0.10 pp | Direct official-versus-PLANETALIGN comparison. Current evaluator defaults to optimistic ties; current seed-0 Phone-Email and Italy logs corroborate the published reproduction. | `CALIBRATED_WITH_PROTOCOL_GAP` | `READY_FOR_M2M` |
| CrossMNA | PLANETALIGN Table 5 | 18-dataset average | Bidirectional Hits@1 | 0.220 official | 0.222 PLANETALIGN implementation | +0.20 pp | Direct official-versus-PLANETALIGN comparison; current shared evaluator default differs from the paper-era ordinary argsort. | `CALIBRATED_WITH_PROTOCOL_GAP` | `READY_FOR_M2M` |
| NetTrans | PLANETALIGN Table 5 | 18-dataset average | Bidirectional Hits@1 | 0.373 official | 0.374 PLANETALIGN implementation | +0.10 pp | Direct official-versus-PLANETALIGN comparison; current shared evaluator default differs from the paper-era ordinary argsort. | `CALIBRATED_WITH_PROTOCOL_GAP` | `READY_FOR_M2M` |
| WAlign | PLANETALIGN Table 5 | 18-dataset average | Bidirectional Hits@1 | 0.271 official | 0.270 PLANETALIGN implementation | -0.10 pp | Direct official-versus-PLANETALIGN comparison; current shared evaluator default differs from the paper-era ordinary argsort. | `CALIBRATED_WITH_PROTOCOL_GAP` | `READY_FOR_M2M` |
| BRIGHT | PLANETALIGN Table 5 | 18-dataset average | Bidirectional Hits@1 | 0.362 official | 0.362 PLANETALIGN implementation | 0.00 pp | Direct official-versus-PLANETALIGN comparison; current shared evaluator default differs from the paper-era ordinary argsort. | `CALIBRATED_WITH_PROTOCOL_GAP` | `READY_FOR_M2M` |
| NeXtAlign | PLANETALIGN Table 5 | 18-dataset average | Bidirectional Hits@1 | 0.391 official | 0.391 PLANETALIGN implementation | 0.00 pp | Direct official-versus-PLANETALIGN comparison; current shared evaluator default differs from the paper-era ordinary argsort. | `CALIBRATED_WITH_PROTOCOL_GAP` | `READY_FOR_M2M` |
| WLAlign | PLANETALIGN Table 5 | 18-dataset average | Bidirectional Hits@1 | 0.328 official | 0.322 PLANETALIGN implementation | -0.60 pp | Direct official-versus-PLANETALIGN comparison. The difference remains within a reasonable reproduction range; current evaluator defaults to optimistic ties. | `CALIBRATED_WITH_PROTOCOL_GAP` | `READY_FOR_M2M` |
| PARROT | PLANETALIGN Table 8 | PeMS08 | Bidirectional ordinary Hits@1/Hits@10/MRR | 0.7647 / 0.9228 / 0.8209 | 0.757353 / 0.930147 / 0.816960 | -0.73 pp | Completed audit run used `data`, 20% anchors, seed 0, plain graph and the fixed project-default configuration; no tuning. | `CALIBRATED_WITH_PROTOCOL_GAP` | `READY_FOR_M2M` |
| SLOTAlign | PLANETALIGN Table 5 | 18-dataset average | Bidirectional Hits@1 | 0.200 official | 0.200 PLANETALIGN implementation | 0.00 pp at displayed precision | Direct official-versus-PLANETALIGN comparison; current shared evaluator default differs from the paper-era ordinary argsort. | `CALIBRATED_WITH_PROTOCOL_GAP` | `READY_FOR_M2M` |
| HOT | PLANETALIGN Table 5 | 18-dataset average | Bidirectional Hits@1 | 0.173 official | 0.172 PLANETALIGN implementation | -0.10 pp | Table 5 supports the implementation. The PeMS08 MRR cell in Table 8 is internally inconsistent with its Hits@1 and is not used as calibration evidence. | `CALIBRATED_WITH_PROTOCOL_GAP` | `READY_FOR_M2M` |
| JOENA | PLANETALIGN Tables 5 and 9 | 18-dataset average / Douban | Bidirectional Hits@1 | 0.583 official / 0.6542 +/- 0.0474 Douban | 0.583 PLANETALIGN implementation / 0.63184 completed audit Douban run | 0.00 pp / -2.24 pp | The JOENA-paper value 0.761 is source-to-target and is not directly comparable with PLANETALIGN's bidirectional mean. Existing calibration is sufficient. | `CALIBRATED_WITH_PROTOCOL_GAP` | `READY_FOR_M2M` |
| GADL (official implementation) | GADL paper | Douban | Source-to-target public-optimistic Hits@1 | 0.5331 | 0.537865 +/- 0.003748 | +0.48 pp | Unsupervised, graph-consistent public-code reproduction. The public code configuration and the paper-declared configuration differ. | `CALIBRATED_WITH_PROTOCOL_GAP` | not assigned; reference-only row |
| GADL (PlanetAlign integration) | GADL paper | Douban | Source-to-target optimistic Hits@1 | 0.5331 | 0.319320 +/- 0.003347 | -21.38 pp | Tie handling explains only a small part of the gap; graph/spectral preprocessing and training semantics differ from the public implementation. | `SUSPICIOUS` | `DO_NOT_USE` |

## Coverage Summary

- Major methods audited: 15 (PLANETALIGN's 14 methods plus GADL).
- Current implementations present: 15/15; `NOT_IMPLEMENTED`: 0.
- Algorithm implementations with evidence in a reasonable performance range: 13.
- `CALIBRATED_WITH_PROTOCOL_GAP`: IsoRank, FINAL, REGAL, CrossMNA, NetTrans, WAlign, BRIGHT, NeXtAlign, WLAlign, PARROT, SLOTAlign, HOT, JOENA.
- `SUSPICIOUS`: current PlanetAlign GADL integration.
- `UNCHECKED`: IONE.
- Methods with reliable current results but still awaiting a calibration decision: none. GADL has been diagnosed and deliberately remains suspicious.
- PLANETALIGN Table 5 supplies official-code comparisons for REGAL, CrossMNA, NetTrans, WAlign, BRIGHT, NeXtAlign, WLAlign, SLOTAlign, HOT, and JOENA.
- An independent official checkout and graph-consistent reproduction are available for GADL.
- Official checkout availability was not independently re-audited for IsoRank, FINAL, IONE, and PARROT; this is not interpreted as absence of official code.

## Minimal Rerun List

No experiment was rerun while generating this record. The only future calibration run still required is:

| Method | Why Rerun | Dataset | Config | Seeds | Expected Purpose |
|---|---|---|---|---|---|
| IONE | No reliable current one-to-one result | PeMS08 | `data`, 20% anchors, plain graph, `out_dim=100`, `total_epochs=100`, `base_epochs=100000`; ordinary bidirectional Hits@1/Hits@10/MRR | Seed 0 first; add 1 and 2 only if seed 0 is ambiguous | Determine whether the current implementation lies in the reasonable range of the published 0.3824 +/- 0.0252 result |

GADL is not on the rerun list: the existing multi-seed and controlled-profile evidence already demonstrates an implementation gap. Repeating the unchanged implementation would not add calibration information.

## Suspicious Implementation

### GADL (current PlanetAlign integration)

- Reference behavior: paper Douban source-to-target Hits@1 is 0.5331; graph-consistent public-code reproduction is 0.537865 +/- 0.003748.
- Current behavior: optimistic Hits@1 is 0.319320 +/- 0.003347; average-tie Hits@1 is 0.291294 +/- 0.003748.
- Confirmed implementation/protocol differences:
  - PReLU in the PlanetAlign implementation versus identity activation in the public implementation.
  - Dense unweighted BCE versus weighted reconstruction BCE.
  - Constant versus linear-floor functional-map schedule.
  - Functional-map initialization, spectral-basis normalization, and loss-weight mapping differences.
  - Paper full-graph preprocessing versus public baseline-compatible graph preprocessing.
  - The paper-declared `hidden_dim=256`, six layers and `lr=1e-3` disagree with the public code's `hidden_dim=512`, three layers and `lr=1e-4`.
- Experimental support: the isolated public-compatible PlanetAlign wrapper reaches only 0.362254 +/- 0.005978. Tie policy accounts for approximately 2.8 percentage points, not the approximately 21-point main gap.
- Recommended minimal fix: keep the current implementation and logs untouched; add an isolated official-behavior adapter/profile that exports the public implementation's similarity matrix. Require a three-seed Douban calibration before M2M admission.
- Impact: only future GADL results and caches. Existing GADL logs remain diagnostic/reproduction artifacts and must not be relabeled as valid baselines.

### Reference-table caveat

The HOT PeMS08 MRR value in PLANETALIGN Table 8 is smaller than its Hits@1 value, which is mathematically impossible for the same ranked predictions. This cell is treated as a reference-table inconsistency, not evidence of a HOT implementation defect.

## M2M Admission Decision

| Baseline | Decision | Reason |
|---|---|---|
| IsoRank | `READY_FOR_M2M` | Exact ordinary-argsort PeMS08 sanity reproduction; evaluator default must be pinned explicitly. |
| FINAL | `READY_FOR_M2M` | Existing results fall in the published range with an explained tie/split gap. |
| IONE | `PENDING_CALIBRATION` | No reliable current one-to-one result. |
| REGAL | `READY_FOR_M2M` | Official-versus-PLANETALIGN reproduction is calibrated; adapter must preserve `emb1 @ emb2.T`. |
| CrossMNA | `READY_FOR_M2M` | Official-versus-PLANETALIGN reproduction is calibrated. |
| NetTrans | `READY_FOR_M2M` | Official-versus-PLANETALIGN reproduction is calibrated. |
| WAlign | `READY_FOR_M2M` | Official-versus-PLANETALIGN reproduction is calibrated. |
| BRIGHT | `READY_FOR_M2M` | Official-versus-PLANETALIGN reproduction is calibrated. |
| NeXtAlign | `READY_FOR_M2M` | Official-versus-PLANETALIGN reproduction is calibrated. |
| WLAlign | `READY_FOR_M2M` | Official-versus-PLANETALIGN reproduction is within a reasonable range. |
| PARROT | `READY_FOR_M2M` | PeMS08 sanity reproduction is within one percentage point without tuning. |
| SLOTAlign | `READY_FOR_M2M` | Official and PLANETALIGN results match at displayed precision. |
| HOT | `READY_FOR_M2M` | Table 5 official comparison supports the implementation; the anomalous Table 8 MRR cell is excluded. |
| JOENA | `READY_FOR_M2M` | Existing official comparison and direct Douban calibration are sufficient. |
| GADL (current PlanetAlign integration) | `DO_NOT_USE` | Large unexplained implementation gap remains after evaluator and profile controls. |

`READY_FOR_M2M` means only that the one-to-one backbone passed calibration. It does not authorize reuse of old M2M numbers or any dataset-specific readout/configuration change.

## Evidence Inventory

- PLANETALIGN paper: `/Users/yukexin/Documents/akeyan/papers/15073_PLANETALIGN_A_Comprehens.pdf`
- JOENA paper: `/Users/yukexin/Documents/akeyan/papers/Joint_Optimal_Transport_And_Embedding_for_na.pdf`
- GADL paper: `/Users/yukexin/Documents/akeyan/papers/dual-pass_spectral_encoding.pdf`
- FINAL/REGAL existing results: `.reference/Alpine/planetalign_benchmark/results/existing_baselines_all_seed0.json`
- GADL official graph-consistent results: `logs/gadl_one_to_one/gadl_reference_graph_consistent.json`
- GADL current integration results: `logs/gadl_one_to_one/douban_full_training_seeds_0_1_2.json`
- GADL public-compatible profile results: `logs/gadl_one_to_one/douban_public_reference_compatible_seeds_0_1_2.json`
- GADL profile definitions: `PlanetAlign/algorithms/gadl/profiles.py` and `PlanetAlign/algorithms/gadl/paper_profile.py`
- Current shared evaluator default: `PlanetAlign/algorithms/base_model.py`
- IsoRank, PARROT, and the direct JOENA sanity numbers were produced earlier in the completed audit with `save_log=false`; they did not create or overwrite raw logs.

## Change Scope

No algorithm, evaluator, metric definition, configuration, cache, or original log was modified to create this calibration record.
