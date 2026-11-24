# LAVEN implementation status

Local upgrade branch: codex/laven-research-upgrade.
Preserved baseline: 06a50f97e973baf2755e34905adf7fda1442d71e.
Default application mode is the baseline LAVEN answer path.

| Phase | Implementation and verification | Remaining evidence or work |
| --- | --- | --- |
| 0 | Baseline commit and architecture recorded, separate branch, original backend tests retained. | Disposable live baseline verification. |
| 1 | Release download/import CLI, checksum manifests, raw spans and grouped leakage audits. Measured CUAD 510 documents/20,910 question rows and LegalBench train counts 8/8/6, test 80/142/220. | Import the terms-gated ContractNLI release, validate an approved LegalBench-RAG archive, and obtain a reviewed clean BIPIA derivative if needed. Cross-corpus source adjudication pending. |
| 2 | Real TF-IDF and BiLSTM CPU training, exported checkpoints, vocabulary, predictions and figures. Tiny fictional test had 9 rows, 9/9 TF-IDF and 8/9 BiLSTM correct. | ContractNLI runs, repeat seeds and measured held-out results. Smoke execution is not evidence of legal accuracy. |
| 3 | Transformer training and pretrained NLI comparison CLI, token overflow handling, artifact export and optional FastAPI loading. | Full GPU fine-tuning, general NLI run and final legal model/calibration evaluation. |
| 4 | Deterministic 100-family, 800-row fictional pilot, reviewed-only runner and bounded benign Qwen paraphrase tooling. | Independent review, broader data diversity, Qwen proposals and defense evaluation on approved fixed controls. The pilot marker is inert. |
| 5 | Strict structured claims, neural gate and bounded Z3 verification. Unit tests exercise supported, inconsistent, contradicted and insufficient-premise outcomes. | Model-based semantic rule extraction accuracy and end-to-end calibration. |
| 6 | Additive owner-scoped migration, full-version registration, comparison and selection UI, as-of date, deterministic precedence tests and CS/IS/VRS scoring. | Staging Supabase acceptance, revocation history and clause-level lineage. |
| 7 | Sampled Qwen answers, empirical entropy, held-data temperature scaling, calibration and selective decision utilities. Rendered smoke reliability and risk-coverage figures. | Validate semantic grouping and thresholds on reviewed eligible data. |
| 8 | Bounded corrective rerun, explicit abstention, authenticated JSON/PDF verification and deterministic replay. Frontend shows citations, version, unresolved conditions, outcomes and interactive evidence links. | Verification against a real trained checkpoint and real services. |
| 9 | Research Laboratory accepts approved aggregate measurements, supports 45 figure slots, and shows honest empty states. Production frontend build and local backend regression tests run. | Staged service/browser acceptance, ingestion and visualization mappings for remaining experiment metrics as real results exist. |
| 10 | Budgeted M1–M7 fictional runner, selected ablations, grouped intervals and actual classifier plotting. | Reviewed pilot and native matched-budget M1–M7 runs, remaining isolated ablations, true held-out native measurements and user approval before large paid runs. |
| 11 | Architecture, data/model cards, training setup, methods/metrics, experiment protocol, literature/reference notes and CO1–CO5/viva guide. | Confirmatory results, full systematic literature screening and publication-ready claims once the experiment is completed. |

## Files by component

- Python integration: backend/.env.example, backend/requirements-verification.txt and backend/legalshield/{api,config,groq,research_artifacts,upgraded,verification}.py.
- Research tools: research/{aggregate,calibration,datasets,experiments,fixture,nli,pilot,plots,run_pilot,scenario_paraphrases}.py and research/requirements.txt.
- Application: frontend/app/page.tsx, frontend/lib/api.ts and frontend/components/{legal-versions,research,research-artifacts,verification-details}.tsx.
- Storage and verification: supabase/002_laven_versions.sql and scripts/replay_verification.py.
- Tests: backend/tests/{test_upgrade,test_upgrade_routes}.py and research/tests/test_research.py.
- Reproduction: seven notebooks in notebooks/, README.md and the research documents in docs/.

Next live gate: independently review fictional scenarios, obtain the approved
ContractNLI dataset, train and calibrate the verifier, then apply migration 002
to a disposable Supabase project with researcher and standard user fixtures.
Run the existing scripts/verify_live.py and enhanced UI/API checks there before
enabling LAVEN_ENHANCED outside staging.



## Local handoff verification

- Backend regression suite: 21 passed, including optional Z3 tests with the isolated solver installed.
- Research regression suite: 4 passed.
- Next.js production build and its TypeScript phase passed.
- All seven notebooks passed nbformat validation and code-cell compilation. Full Colab runs depend on the listed inputs and runtime.
- Local outputs are in the ignored artifacts/local-research directory. It includes the fictional pilot, verified public dataset count manifests, a dataset-only Research Laboratory index, and separately identified CPU smoke checkpoints, predictions, metrics and figures.
- No paid Groq experiment ran. Pilot estimate mode found zero independently approved instances. No Supabase migration was applied to a live project.


## Actual ContractNLI training progress

- Official archive downloaded and checksum-verified. Prepared 10,217 eligible
  pairs from 10,319 official pairs. Six near-duplicate training contracts
  were excluded; all 2,091 official test pairs remain. Validation and
  calibration use disjoint development contracts (629 and 408 pairs).
- TF-IDF on CPU finished on the real eligible data: accuracy 68.63%, macro F1
  64.04% over the 2,091 held-out pairs. See artifacts/contractnli/tfidf.
- CUDA verified on RTX 4050 (Torch 2.9.1+cu128). BiLSTM completed six
  epochs and selected epoch 3 by validation macro F1. Frozen general-domain
  DeBERTa completed evaluation on the same 2,091 test examples.
- Existing fictional pilot validated: 800 unique cases, 100 unsplit families,
  matching raw evidence offsets, all pending human review.
- The existing DeBERTa notebook now provides full and smoke workflows,
  verified dataset alignment, GPU-dependent precision and batch settings,
  epoch checkpoints, optional Drive, and calibration/ZIP export.


| Real held-out run | Accuracy | Macro F1 | Weighted F1 |
| --- | ---: | ---: | ---: |
| TF-IDF logistic regression, CPU | 68.63% | 64.04% | 67.93% |
| BiLSTM, RTX 4050 | 67.72% | 63.19% | 67.85% |
| Frozen general-domain DeBERTa, RTX 4050 | 51.89% | 35.72% | 46.50% |
| Legal DeBERTa fine-tuning | Pending Colab | Pending | Pending |

Comparison verified matching dataset hashes, context settings, test IDs and
gold labels. Predictions, checkpoints, CSV summaries and figures are stored
in artifacts/contractnli. BiLSTM resume was tested with identical reload
predictions on a small separate fixture. Real BiLSTM and general-domain
calibration diagnostics are available. Only a matching, deployable legal
DeBERTa export can enable the enhanced /ask path. Supabase live and paid
Groq experiments have not run.

Final local verification: the Colab notebook and exact source ZIP were
validated. A separate one-epoch GPU smoke with the locally available
same-architecture checkpoint verified BF16 training, gradient checkpointing
and Safetensors export after retaining FP32 optimizer weights. Its
calibration is non-deployable. The full Microsoft-base Colab run is pending.
Frontend type-check passed. Local FastAPI configuration read all 18 measured
Research Laboratory figures and reported LAVEN_ENHANCED=false.


## Validation-only optimization of legal DeBERTa

- The three approved metrics and baseline artifacts are preserved. No new
  official test predictions were generated, and test outcomes were excluded
  from optimization.
- Validation audit confirmed checkpoint label order, input ordering and
  hypothesis token preservation. Learned top-five all-span coverage was
  81.22%, versus 58.63% from the original lexical top-four. See
  artifacts/contractnli/validation-diagnostics.json.
- The existing Colab notebook now uses NLI-pretrained weights, supervised
  hypothesis-aware evidence retrieval, overlapping windows, a relevance
  head and document-level aggregation. Two bounded candidates are selected
  by validation macro F1 before the test cell opens. The larger NLI
  alternative is optional and validation-only.
- Matching source ZIP rebuilt under artifacts/contractnli. The legal
  checkpoint, its 2,091-example evaluation and backend integration remain
  pending the Colab export.

- Final RTX 4050 smoke completed a BF16 epoch using three training and
  three validation pairs, saved and resumed the full checkpoint, and
  exported model, evidence head and retrieval weights. No smoke test
  predictions file exists. Notebook schema/code and source ZIP members were
  checked against the working tree.


## Two-hour FAST_TRAIN mode

- The single-trial FP16 T4 path defaults to two epochs and a 90-minute
  training phase. Validation admitted learned top-four with a 512-token
  maximum and dynamic padding. The 256/384 candidates increased actual
  token count through overflow.
- RTX 4050 FP16 smoke passed batch probing, one training/validation epoch,
  checkpoint resume and export. It did not access official test outcomes.
- Notebook and matching source ZIP updated. Full mode is still available
  via FAST_TRAIN=False. T4 wall time and the 85–90% research target remain
  unmeasured until the Colab run.


## Imported Colab checkpoint

- Imported and verified the ZIP, model, evidence head, retriever and all 2,091 official test and 408 calibration IDs and labels. The ignored output is artifacts/contractnli/deberta-colab.
- Validation accuracy 86.33%, macro F1 80.87%. Test accuracy 87.18%, macro F1 82.98%, macro precision 81.98%, macro recall 84.45%, weighted F1 87.33%. Tesla T4 processed two epochs and selected epoch 1.
- Updated the four-system figures and 18-entry Research Laboratory index. Preserved the three baseline reports.
- Integrated evidence-head pooling. Actual checkpoint passed local synthetic /ask, citation/PDF export and role-denial checks with service doubles. 22 backend tests, 4 research tests and the production frontend build passed.
- Groq model lookup and one live synthetic JSON completion passed. Supabase Auth and baseline table checks passed. Anonymous document reads were denied. Migration 002 was absent (PGRST205).
- Enhanced remains off pending reviewed cited-evidence calibration, migration 002 and disposable-account live staging acceptance. Independent pilot review and paid research experiments did not run.

## Generative workflows and performance dashboards

- AI Performance & Analytics surfaces measured official ContractNLI scores, high-resolution Matplotlib images, training, confusion and calibration plots. Prior experiment submission remains available on the researcher API. Saved runs and exports remain accessible.
- Legal Generation Studio and Legal Reasoning Simulator reuse Qwen and owner-scoped source retrieval. The interface charts measured outputs and supports draft, evaluation and scenario export. Independent verification remains governed by the existing cited-passage deployment gate.
- My Documents reports flagged span counts, affected pages and retained candidate evidence. Reconstruction provides separate derived images and a VAE latent sampler after verified checkpoint import. No original evidence or deployed inference flag was changed.
- Reconstruction notebook 08, source package, synthetic source partitions, resume, metrics and import are implemented. Before the training run, a CPU smoke completed one epoch per model on 10 / 5 / 5 source documents.
- Verification completed with 28 backend and research tests, seven UI render checks, a production frontend build, notebook/source ZIP validation and local server checks. The signed-out browser smoke reached the login screen. Live signed-in Studio and Simulator operations remain unverified.

## Imported document reconstruction checkpoints

- Ran FAST_TRAIN locally on CPU for eight epochs each. Evaluated both models on all 120 held-out fictional layouts after selection on 120 validation layouts. The training split contained 480 layouts. CER and WER were Not Evaluated.
- Verified the ZIP, checkpoint hashes, disjoint source splits and predictions. Imported both models into the ignored artifacts/reconstruction/trained directory and configured the local backend.
- Tested actual trained inference for both models on a fictional PDF, controlled metrics, reference-free unscored inference, VAE latent PNG output, analytics PNG serving, unauthorized and out-of-scope denial, and preservation of source bytes. The test used a disposable store fixture.
- Added a separate Document Restoration sidebar page and an authenticated direct-upload endpoint for PDF, PNG and JPEG. Verified the trained models on an already blurred PNG and an image-only PDF, plus regression checks for unsupported formats, upload limits and page selection.

## Text-preserving restoration retraining

- Verified legacy checkpoint hashes and unit-interval normalization. Identity patch stitching reproduced the input within 3e-7. Legacy decoders lost text on real contracts and produced no recognized text on the 12-page controlled test.
- Replaced direct pixel decoding with a text-preserving residual path and retrained both models on real ContractNLI PDF rasters with paired blur. Eight epochs each and distinct 30/8/12 source document splits. Confirmed training source selection excluded all previously audited overlapping families.
- Measured both image similarity and Windows OCR against clean sources. The new AE and VAE improved the blurry-input mean MSE, PSNR, SSIM, CER and WER. Imported text-v2, tested the upload, latent and research endpoints with the real weights and a local auth fixture, and activated the new models in local configuration. Original uploaded evidence remains untouched.

## Default DAE and live generation verification

- Trained a multi-scale DAE with residual skips, foreground/edge losses and SSIM on 54/12/18 actual ContractNLI sources. On the new held-out comparison, DAE improved all five requested metrics over both the degraded input and previous AE. Kept the VAE checkpoint byte-for-byte and retained optional comparisons and latent generation. Installed the DAE artifact and figures under artifacts/reconstruction/dae-v3 and made it the default in the existing restoration interface.
- Fixed Studio's Groq JSON-mode instruction and gave baseline Simulator a scenario-specific response schema. Verified live Groq draft generation and two source-cited hypothetical scenarios using fictional data and a local auth/retrieval fixture. Thirty backend and research tests and the frontend type check passed.

## Simulator removal and README rewrite

- Removed the Legal Reasoning Simulator from navigation, frontend rendering, and the FastAPI surface. Legal Generation Studio, AI Legal Assistant and the independent verification path remain available according to their existing configuration. Offline research notebooks were retained.
- Rewrote the root README around the current features, user workflows, GenAI topics, setup, datasets, measured results, and deployment gates. All 30 regression tests and the frontend type check passed.

## Scanner coverage and upload progress

- Expanded rule coverage and allowed whitespace changes in model quotes while preserving source offsets. On the supplied 18-page test PDF, local rules covered all 20 labeled cases in 24 spans. The live Groq-assisted count can differ.
- Indexing scans up to three pages concurrently per document. The upload dialog distinguishes transfer from indexing and shows elapsed time. The backend emits page and chunk progress. Fifteen pipeline tests and the frontend type check passed. The README documents these changes and how to reindex.
