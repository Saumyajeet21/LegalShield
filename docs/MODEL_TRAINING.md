# Training, export and artifact import

From the repository root, use Python 3.11+ with research/requirements.txt in a separate environment.
See notebooks/ for seven Colab stages. Set PROJECT to the checkout containing
this branch and persist outputs to a safe workspace. ContractNLI release
access and a CUDA runtime are required for the intended full DeBERTa run.

    python -m research.datasets contractnli research-data/contractnli artifacts/nli.jsonl --revision 2021-10-05 --calibration-fraction 0.4
    python -m research.datasets audit artifacts/nli.jsonl artifacts/leakage.json
    python -m research.nli tfidf artifacts/nli.jsonl artifacts/tfidf
    python -m research.nli bilstm artifacts/nli.jsonl artifacts/bilstm --epochs 20 --batch-size 32
    python -m research.nli general artifacts/nli.jsonl artifacts/general
    python -m research.nli deberta artifacts/nli.jsonl artifacts/deberta --device cuda --batch-size 4 --accumulation 4 --mixed-precision
    python -m research.calibration artifacts/deberta/predictions-calibration.csv artifacts/deberta/calibration.json --test artifacts/deberta/predictions-test.csv
    python -m research.plots nli artifacts/deberta artifacts/deberta/figures
    python -m research.plots calibration artifacts/deberta/calibration.json artifacts/deberta/figures

Use a fresh run directory per seed (13, 29, 47) or track. Fix or explicitly
document eligibility from the leakage report. The lexical retrieval track is
shared across models and distinct from production pgvector. Oracle coverage
is restricted by annotation support. NotMentioned uses the same lexical
context fallback on the oracle track. DeBERTa uses overlapping tokenizer overflow windows and mean aggregation
when a selected excerpt exceeds the configured token ceiling. Interpret models on the
same named track. Record model revision with --revision. Choose the small
Microsoft checkpoint via --source-model microsoft/deberta-v3-small if base
exceeds memory.

Run directories include run.json, metrics.json, history.json and split
predictions CSV. TF-IDF exports a local trusted joblib artifact, BiLSTM exports
best.pt and vocabulary.json, and DeBERTa exports model/ with safetensors and
its tokenizer. Import only the intended local artifact. On the FastAPI host
install backend/requirements-verification.txt, set LAVEN_VERIFIER_PATH to
model/, LAVEN_CALIBRATION_PATH to calibration.json from the corresponding
run, and apply migration 002 before enabling LAVEN_ENHANCED. Activate only
when deployable is true and real held-out metrics support the comparison.
Keep the original Qwen and Supabase environment values.


Keep run.json and calibration predictions in the same run folder before
calibrating. Deployment verifies the calibration's model-weight SHA-256.
A smoke-only calibration cannot be deployed.


## Current ContractNLI protocol

The official archive is in research-data/contractnli/official-git.zip.
The prepared Windows/Colab corpus has SHA-256
6d825a619cf29c97faf50c63e03bbac46ee7195a1b8fe8c393c20f95646436eb.
Compare all models on the same 2,091 official test IDs and labels with four
lexically selected windows and the 512-token setting. Near-duplicate review
excluded six training contracts. The untouched official development documents
are split once by document into validation and calibration subsets.
Raw annotation offsets remain anchored to the original text.

PowerShell from the repository root:

    $py = '.\backend\.venv\Scripts\python.exe'
    & $py -m research.prepare_contractnli research-data/contractnli/official-git.zip research-data/contractnli/eligible.jsonl --sha256 e03fc77bbf8b53e2976a250e81d8a294bc3d5e5fb014521e477dee9340d6287b
    & $py -m research.nli tfidf research-data/contractnli/eligible.jsonl artifacts/contractnli/tfidf --device cpu --batch-size 32
    $env:PYTHONPATH = (Resolve-Path 'artifacts/cuda-pytorch').Path
    & $py -m research.nli bilstm research-data/contractnli/eligible.jsonl artifacts/contractnli/bilstm --device cuda --batch-size 32 --epochs 20 --patience 3

Use a fresh run directory for a new run. For an interrupted BiLSTM or
DeBERTa run, append --resume to the same arguments. The latest epoch
checkpoint includes optimizer, scheduler where applicable, scaler, history
and Torch RNG state.

## Legal DeBERTa study and Colab handoff

The approved TF-IDF, BiLSTM and frozen DeBERTa artifacts are preserved.
Their official held-out accuracy/macro F1 are 68.63%/64.04%, 67.72%/63.19%,
and 51.89%/35.72%. The evidence-supervised legal model is a new experimental
configuration on the same 2,091 official test IDs. Its evidence retrieval
and aggregation differ from those of the baselines, so compare it as an
end-to-end system. No legal DeBERTa test score has been measured.

### Training and validation diagnosis

The frozen checkpoint uses native entailment/neutral/contradiction logits
at indices 0/1/2, which the original run correctly evaluated in canonical
order [0, 2, 1]. The audit verified premise first, hypothesis second. All
hypothesis tokens survived tokenizer overflow, which affected 3 of 297
selected windows on the audited sample. Of 7,089 eligible training pairs,
832 are contradictions. Domain and task mismatch remain plausible: MNLI
neutral is a judgment about a pair of texts, while NotMentioned is a
contract-wide absence label. Retrieval quality and aggregation also affect
the result, so this audit does not identify a single cause.

On a fixed balanced validation probe of 75 pairs, mean window logits
achieved 45.33% accuracy and 36.82% macro F1. Top-1 achieved 52.00% and
46.23%, respectively. Reversing input order reduced mean-logit accuracy
to 33.33%. The frozen mean-logit probe got 0 of 25 contradictions correct.
These are diagnostic results on a stratified validation subset.

| Evidence coverage on 394 annotated validation pairs | Any span | All spans | Span recall |
| --- | ---: | ---: | ---: |
| Original lexical top 4 | 78.93% | 58.63% | 63.86% |
| Learned top 4 | 92.89% | 78.43% | 84.62% |
| Learned top 5 | 93.91% | 81.22% | 87.42% |

The selected method fits a hypothesis-conditioned ranker on annotated
training contracts. It retrieves five overlapping 160-token windows per
pair, then uses 64-token overlap on Transformer overflow. It initializes
from the published NLI-pretrained base checkpoint and physically reorders
the classifier rows to preserve the learned semantic label order.
A relevance head learns from annotated windows and high-ranked hard
negatives on training only. Partial-span overlaps and unannotated
NotMentioned examples are masked for evidence supervision. Document
probabilities pool window probabilities using predicted relevance. The
model combines document NLI loss with 0.35 times evidence BCE.
Gold spans never select or weight windows in normal validation, calibration,
or test inference. The window-level auxiliary task follows the principle of
evidence-weighted aggregation in the ContractNLI paper. The paper's stronger
xlarge configuration reported 89.2% under its setup. The 85–90% range for this
run remains a research target.
[Paper](https://aclanthology.org/2021.findings-emnlp.164.pdf)

### Execute the notebook

1. In Colab, use **File → Upload notebook** on notebooks/03_deberta_finetuning.ipynb.
   Set **Runtime → Change runtime type → GPU**. Prefer A100 40 GB. The two
   base trials may take roughly 1–3 hours on A100 or 4–10+ hours on T4.
   This is a planning estimate, with actual time reported after epoch 1.
   The optional large trial adds time. Colab availability and limits vary.
   [Colab runtime FAQ](https://research.google.com/colaboratory/faq.html)
2. Run **Config** and **Setup**. Leave SMOKE=False for research. Upload
   artifacts/contractnli/laven-colab-source.zip when prompted. Either upload
   the authorized ContractNLI archive as /content/contract-nli.zip or review
   the official terms and set ACCEPT_CONTRACTNLI_TERMS=True.
   Set MOUNT_DRIVE=True only if you approve storing intermediate checkpoints
   in Drive. Drive helps recover after Colab disconnection. Resume with
   the same run folder, batch settings, GPU precision and runtime versions.
3. Run **Dataset**, then **Train**. Two paired candidates test 1e-5 with
   square-root inverse-frequency class weights and 2e-5 without weights.
   Each uses up to four epochs with patience two. The notebook configures
   AdamW, warmup/decay, batch size, gradient accumulation, BF16 or FP16,
   and checkpointing based on available GPU memory. Each epoch prints
   accuracy, loss, macro F1 and elapsed fit time.
4. Read both validation results. If the validation winner is below 85%,
   optionally set RUN_LARGE_TRIAL=True before freezing and rerun Config
   through Train with the same paths and compatible base settings.
   This adds one pinned NLI-pretrained large-checkpoint trial on an
   A100-class GPU. Its benefit on ContractNLI is unmeasured. If proceeding
   with a below-target checkpoint instead, set EVALUATE_BELOW_TARGET=True in
   Config. Use the notebook's actual validation macro F1 selection rule.
   [Optional large checkpoint](https://huggingface.co/MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli)
5. Run **Freeze** once the trial set is final. It records the choice by
   validation macro F1, with accuracy as a tie-breaker. Only then does it
   evaluate the winner on the 408 calibration pairs and all 2,091 official
   test pairs. Keep the winner frozen regardless of the test score.
6. Run **Export**. Download laven-deberta-contractnli.zip, or find it under
   /content if the browser download prompt fails. Return the ZIP for backend
   integration. It contains the legal model, tokenizer, evidence head,
   retriever, exact selection/configuration, calibration, predictions,
   histories and figures. The application currently uses pair-level
   inference. The document-bag calibration is deliberately non-deployable
   in that path pending integration and matched validation.


### Two-hour T4 session

Leave FAST_TRAIN=True, FAST_EPOCHS=2 and SMOKE=False. This runs one
NLI-pretrained base trial in FP16. It tests real training-shaped batches
for memory headroom, retains FP32 weights and enables gradient checkpointing
only if ordinary batches do not fit. Use FAST_TRAIN=False for the full
multi-trial workflow above.

The fast configuration checks train-fitted retrieval on validation only.
Four ranked windows at a 512-token ceiling retained 78.43% full-span
coverage versus 81.22% for five. Batched overflow at 256 or 384 retained
the same coverage but used more encoded tokens. Dynamic padding keeps
batches near their actual lengths. Cached retrieval selections and tokens
are reused across epochs and resume.

Run Config, Setup, Dataset, Train, Freeze and Export in that order. The
training phase is capped at 90 minutes and also stops by roughly minute
95 from Config to leave about 25 minutes for finalization. It prints the
first-steps ETA and reserves time within the training phase for full
validation and atomic checkpoint writes. If the deadline interrupts an
epoch, it finishes the current accumulation group, validates it and
reports the processed count, winner exposure, completed epochs and
FP16-skipped examples. All 7,089 eligible pairs enter the schedule. The
selected checkpoint may receive fewer pairs when time is insufficient.
An interrupted Colab session can resume the latest saved epoch if Drive
was enabled. The managed GPU and export throughput are variable.

Freeze selects by validation macro F1 and evaluates all 2,091 official
test pairs. Export writes laven-deberta-contractnli-fast.zip with the
training coverage, metrics, plots, model and calibration. The fast
configuration does not require reaching 85% validation before reporting
the frozen result.


## Verified Colab import and deployment gate

The FAST_TRAIN export is imported at artifacts/contractnli/deberta-colab.
Matched official test accuracy is 87.18% and macro F1 is 82.98%. Run
research.import_colab for a fresh extraction and --compare-systems with
research.plots model-comparison to compare distinct retrieval tracks on
the same held-out IDs and labels.

Serving uses the ordered retained passages cited by the draft. To create
compatible thresholds, provide two independently reviewed JSONL files
with disjoint families and at least ten cases per label each. Rows need
sample_id, family_id, human_review_status (approved), passages (ordered
source strings), claim and label. From the root:

    $py = '.\backend\.venv\Scripts\python.exe'
    & $py scripts/calibrate_cited_verifier.py artifacts/contractnli/deberta-colab/model artifacts/staging/cited-calibration.jsonl artifacts/staging/cited-validation.jsonl artifacts/staging/cited-verifier-calibration.json

Both entailment and contradiction decisions must reach 90% validation
precision with at least ten selected examples each. A passed offline gate
still needs approved staging migration 002 and owner, access, retrieval,
generation and download checks. Leave LAVEN_ENHANCED=false until these
pass. The Colab research calibration remains non-deployable for serving.

## Document reconstruction

[Notebook 08](../notebooks/08_document_reconstruction.ipynb) trains a convolutional denoising AE and a convolutional VAE on locally generated fictional layout patches. Each source document belongs to exactly one train, validation or test split. A fixed seed controls partitioning and five synthetic degradations. The AE minimizes pixel MSE. The VAE samples z = mean + exp(log variance / 2) * noise during training and adds beta-weighted KL divergence to unit-normal prior. Inference uses the latent mean for reconstruction and sampled latent vectors for new layouts.

1. Upload notebook 08 and the scoped source ZIP to Colab. The ZIP contains backend/legalshield/reconstruction.py and research/reconstruction.py at their repository-relative paths.
2. Select **Runtime → Change runtime type → T4 GPU**, then run all cells. Keep FAST_TRAIN for 480 training, 120 validation and 120 held-out documents, eight epochs per model and automatic FP16 on CUDA. Set FULL_TRAIN for 3,000 / 500 / 500 and 20 epochs. Drive checkpoints and resume are enabled by default. The notebook prints epoch and batch progress and a measured ETA.
3. Download laven-reconstruction.zip. From the project root on Windows run:

   ~~~powershell
   .\backend\.venv\Scripts\python.exe -m research.import_reconstruction .\laven-reconstruction.zip .\artifacts\reconstruction\trained
   ~~~

4. Set LAVEN_RECONSTRUCTION_PATH to the imported directory in backend/.env and restart the backend. Open **Document Restoration**. The researcher dashboard publishes measured held-out plots and metrics from the same import.

MSE and MAE measure pixel error on the 0–1 scale. PSNR is −10 log10(MSE). SSIM compares local luminance, contrast and structure using a 7×7 window. Latency measures model execution on the reported hardware. CER and WER use normalized edit distance against known text with the optional Tesseract OCR run. Image similarity alone cannot establish that reconstructed text preserves legal meaning. The one-epoch smoke run used 10 train, 5 validation and 5 held-out fictional documents and is ineligible for serving.

### Legacy fictional-layout FAST_TRAIN benchmark

The project ran both models locally on CPU for eight epochs each. The seed-2026 split contained 480 training, 120 validation and 120 held-out fictional layouts. Each model used its best validation-loss checkpoint. The verified import is in artifacts/reconstruction/trained and its Matplotlib figures are available in AI Performance & Analytics.

| Held-out mean | MSE | MAE | PSNR (dB) | SSIM |
| --- | ---: | ---: | ---: | ---: |
| Degraded input | 0.018206 | 0.077510 | 21.06 | 0.6472 |
| Autoencoder | 0.011281 | 0.028155 | 19.73 | 0.7130 |
| VAE | 0.010509 | 0.032216 | 20.02 | 0.6638 |

Both models lowered mean pixel error and increased mean SSIM. Both returned lower mean PSNR, which averages the per-image logarithmic values. CER and WER were Not Evaluated because this run did not have Tesseract available. These results measure the fictional layout benchmark.

### Text-preserving restoration on contract pages

The v2 restoration models were trained for eight epochs each on an RTX 4050 using 30 ContractNLI source PDFs for training, eight for validation and 12 for testing. Each split uses one distinct page per source document. The input is a matched degraded raster and the target is its clean original raster. Training uses a spatial residual decoder, a skip path from the actual input, foreground-weighted and edge losses, plus a small VAE KL and auxiliary decoder loss. The larger page render preserves more glyph detail. Checkpoints were selected by validation MSE before the test run. The legacy weights and their fictional benchmark remain available separately.

| Method | Mean MSE | Mean PSNR (dB) | Mean SSIM | OCR CER | OCR WER |
| --- | ---: | ---: | ---: | ---: | ---: |
| Blurry input | 0.01043 | 20.54 | 0.8963 | 34.16% | 44.83% |
| Legacy Autoencoder | 0.03478 | 15.28 | 0.7978 | 100.00% | 100.00% |
| Legacy VAE | 0.03427 | 15.32 | 0.7283 | 100.00% | 100.00% |
| Text-preserving Autoencoder | 0.00385 | 24.94 | 0.9667 | 10.70% | 20.97% |
| Text-preserving VAE | 0.00424 | 24.49 | 0.9650 | 13.25% | 27.12% |

The test uses real contract pages with controlled synthetic blur. Windows Media OCR English results were compared against the clean PDF text. Clean raster OCR itself had 7.12% CER and 8.59% WER. The v2 checkpoints, metrics, comparison and figures are in artifacts/reconstruction/text-v2.

### Default denoising autoencoder

The deployed DAE uses U-Net-style spatial skips, a bounded residual decoder, foreground-weighted pixel loss, gradient loss and SSIM loss. Training used 54 ContractNLI source PDFs and 12 distinct validation PDFs for 24 epochs on the RTX 4050. Inputs combine defocus, horizontal blur, noise, contrast loss and JPEG compression. Targets are matched clean rasters. The model uses 96-pixel patches with overlap on page renders up to 1,700 pixels. Minimum validation MSE determined the checkpoint. VAE weights were retained byte for byte.

| Method on 18 held-out pages | Mean MSE | Mean PSNR (dB) | Mean SSIM | OCR CER | OCR WER |
| --- | ---: | ---: | ---: | ---: | ---: |
| Degraded input | 0.01232 | 19.59 | 0.7726 | 30.76% | 36.08% |
| Previous AE | 0.00441 | 24.50 | 0.8781 | 21.18% | 25.93% |
| DAE | 0.00184 | 28.70 | 0.9829 | 14.73% | 20.78% |
| Unchanged VAE | 0.00443 | 24.36 | 0.8942 | 22.44% | 30.12% |

The 18 test pages came from different source documents than the v2 test. All methods used identical degraded inputs, clean references, and Windows Media OCR. OCR accuracy on a new user scan requires a clean transcript. Outputs, per-page predictions and figures are in artifacts/reconstruction/dae-v3. Reproduce with research.text_restoration --dae --archive research-data/contractnli/official-git.zip --legacy artifacts/reconstruction/text-v2 --out RUN_DIRECTORY --epochs 24.
