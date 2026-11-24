# LegalShield AI / LAVEN

LegalShield AI is a private legal-document workspace for asking source-backed questions, drafting documents, inspecting suspicious passages, and making blurry scans easier to read. **LAVEN** stands for **Legal Authority Verification and Evidence Neutralization**. It connects document screening, evidence retrieval, generation, and verification while keeping original documents available for review.

[Use the website](#use-the-website) · [How it works](#how-it-works) · [GenAI topics](#generative-ai-topics) · [Measured results](#measured-results) · [Installation](#install-and-run) · [Datasets and training](#datasets-and-training) · [Research and limitations](#research-and-limitations)

## Use the website

| Page | What it does | How to try it |
| --- | --- | --- |
| **Dashboard** | Shows workspace document, conversation, and security finding counts and links to the tools. | Sign in, open Dashboard, and select a tool. |
| **My Documents** | Uploads, screens and indexes text PDFs, DOCX and UTF-8 TXT. Provides preview, download, reindexing, deletion and detection results. | Upload a fictional agreement. Wait for processing, then open **Preview**. When findings exist, expand **Detection results**. |
| **AI Legal Assistant** | Answers questions about selected documents, with clickable citations and conversation history. | Select the agreement and ask “How many days of notice does the agreement require?” Open the cited source and verify the wording. |
| **Legal Generation Studio** | Uses Qwen to generate up to three editable drafts, compare them and regenerate individual clauses. Exports PDF, DOCX and evaluation JSON. | Choose **Supplier agreement**, enter fictional parties and payment or termination terms, then generate drafts. Edit a clause or regenerate it before exporting. |
| **Document Restoration** | Uses a Denoising Autoencoder (DAE) to enhance blurry PDFs and images. Shows the input alongside a downloadable grayscale reconstruction, with optional VAE comparison. | Upload a PDF, PNG or JPEG directly, choose a page for a PDF, and click **Restore page**. Use **Compare DAE and VAE** in the existing model selector if you want both outputs. |
| **Security Alerts** | Shows flagged passages and the actions taken on them. | Open an alert and inspect the highlighted passage in its original context. |
| **AI Performance & Analytics** | Shows researcher-only dataset summaries, saved benchmarks, training and calibration plots, reconstruction metrics and research exports. | Open it with a researcher account. Compare model bars, then expand **Confusion matrices**, **Calibration and uncertainty**, or **Document reconstruction benchmarks**. |

The account menu has **Check connections** for Supabase and Groq. To try the main workflow, upload a fictional agreement that requires 30 days of written notice. Ask about the notice period, inspect the citation, and then ask about a late-payment penalty the agreement never mentions to see how the assistant handles missing evidence.

### Upload and restoration choices

- **My Documents** accepts files up to 20 MiB. PDFs can have up to 300 pages, and extraction is limited to 200,000 characters. Use a text layer or OCR before indexing an image-only scan for legal Q&A.
- Click **Upload and scan**. After the transfer completes, the dialog shows elapsed time while the backend screens and indexes the document. Use **Reindex** (the circular-arrow icon) on an existing document to refresh its findings after a scanner update.
- **Document Restoration** accepts image-only PDFs and PNG/JPEG files up to 20 MiB, with a limit of 25 megapixels per image. It restores one selected PDF page at a time. Direct uploads stay separate from the document library.
- For a clean indexed file, **Document Restoration → Controlled benchmark and VAE explorer** can create a degraded copy, compare reconstruction metrics against the clean render, and generate fictional VAE layout patches. For a real blurry upload, reference metrics require a matching clean image or transcript.

## How it works

~~~mermaid
flowchart LR
    UI["Next.js workspace"] --> Backend["FastAPI"]
    Backend --> Auth["Supabase Auth"]
    Backend --> Files["Private originals and source offsets"]
    Files --> Screen["Rule and Qwen-assisted screening"]
    Screen --> Index["MiniLM + pgvector retrieval"]
    Index --> Draft["Qwen via Groq"]
    Draft --> Ground["Citation and authority checks"]
    Ground --> Answer["Source-linked answer"]
    Ground -. "Independent verification when enabled" .-> Verify["Legal DeBERTa + Z3"]
    Backend --> Restore["DAE restoration and optional VAE"]
    Restore --> Derived["Separate derived PNG"]
    Research["Training and held-out evaluation"] --> Analytics["Measured charts and exports"]
~~~

### Document ingestion and security

1. **Validate and preserve.** The backend checks ownership, file type and limits. It stores original bytes privately and records a SHA-256 digest. PyMuPDF extracts PDF text and page coordinates. The document viewer can highlight the original position of a finding or citation.
2. **Screen.** Pattern rules and Qwen-assisted screening flag instruction overrides, spoofed roles, answer and citation manipulation, disclosure attempts, and other suspicious instructions. Indexing screens up to three pages at a time per document. The backend validates proposed quotes against source text across PDF line breaks, records original offsets, and can isolate individual spans or exclude affected chunks from retrieval.
3. **Index.** It splits the document into clause-aware excerpts. MiniLM embeds the retained text as normalized 384-dimensional vectors for pgvector similarity search. The original remains available through an authenticated, owner-scoped route.
4. **Report.** Detection results show suspicious span counts, affected pages, isolated or quarantined spans, and evidence retention. A single planted test case can produce multiple spans. Retention is the fraction of unique candidate character ranges retained in the retrieval index. **Detector confidence** is **Not Evaluated** because the scanner has no calibrated confidence score.

### Legal Q&A and verification

- **Default answer flow.** The assistant retrieves relevant passages from the selected documents and can consult explicitly trusted references. Qwen produces structured claims with citations. The backend checks source IDs, source quotes, evidence support and available trusted authority before returning claims. It can report insufficient evidence. The result includes clickable sources, conversation history, and measured latency for new answers.
- **Independent verification.** The optional enhanced path adds the fine-tuned legal DeBERTa verifier, Z3 checks for supported typed rules, bounded corrective regeneration, and uncertainty from multiple sampled candidates when configured. It can withhold unsupported answers. A researcher can register reviewed document versions so chat can use the agreement applicable on an as-of date. Verification details show claim status and source provenance, with JSON and PDF exports. Matched baseline comparisons can be requested when this backend path is active.
- **Activation requirements.** The independent verifier is disabled in the current local configuration. Deploying it requires reviewed cited-passage calibration, Supabase migration 002 in staging, and live access tests. The ContractNLI benchmark scores are still available in analytics. [Verification deployment](docs/MODEL_TRAINING.md#verified-colab-import-and-deployment-gate)

### Legal Generation Studio

Studio supports NDAs, supplier and employment agreements, legal notices, amendments, custom drafts, and alternative clauses. It sends the document type, structured requirements, optional retrieved evidence, and sampling settings to Qwen. The backend validates the JSON response before returning an editable draft. Users can adjust temperature, top-p and token limits, request draft variations, and regenerate an individual clause.

The comparison view measures literal requirement-value matches, heading completeness, latency, reported token use and MiniLM draft similarity when available. It reports independent source checks only when the calibrated verifier is active and evidence supports evaluation. These measurements describe the generated draft. Edits are exported separately from its original evaluation. Drafts require human legal review.

### Image restoration and generation

The primary **DAE** is trained to recover text from paired degraded and clean contract rasters. It uses a U-Net-style multiscale encoder and decoder with spatial skip connections. The decoder learns a bounded correction to the input image. Foreground-weighted pixel loss, edge loss, and SSIM loss encourage letter and stroke preservation. Overlapping 96-pixel patches are stitched with weighted blending on page renders up to 1,700 pixels. The website exports a separate grayscale PNG and reports inference time.

The optional **VAE** encodes a distribution over representations and samples from a latent space. It can reconstruct a supplied image or generate a new fictional layout patch in the latent explorer. Generated samples are labeled separately from restorations. Check monetary amounts, dates, names and legal text against the original. Image denoising can change glyphs when a scan is severely degraded.

## Generative AI topics

| Topic | Implementation |
| --- | --- |
| **Transformers and large language models** | Qwen via Groq generates legal answers and drafts, and proposes structured screening findings. |
| **Prompt engineering and structured generation** | System instructions separate supplied evidence from control instructions. JSON outputs are checked before they enter document, citation or drafting workflows. |
| **Retrieval-augmented generation (RAG)** | MiniLM embeddings and pgvector select relevant retained passages. Qwen drafts with that context and includes citations for review. |
| **Conditional generation and sampling** | Studio conditions drafts on structured requirements and optional sources. Users can compare alternatives with different temperature or top-p settings. |
| **Self-refinement and controlled editing** | Studio regenerates a particular clause. The independent verification path can retry a draft using bounded verification feedback when enabled. |
| **Denoising autoencoders** | The DAE learns image-to-image restoration from blurry or noisy page images with clean targets. Skip connections and residual learning preserve spatial information. |
| **Variational autoencoders** | The VAE learns latent mean and variance using reparameterization and KL regularization. Latent sampling generates fictional page layouts. |
| **Transfer learning and NLI** | Legal DeBERTa fine-tunes an NLI-pretrained Transformer with evidence supervision. It classifies support, contradiction, or lack of evidence. NLI is a verification task. |
| **Uncertainty and neuro-symbolic verification** | The optional independent path uses semantic disagreement, calibrated evidence checks and Z3 over explicitly encoded premises. |

TF-IDF and BiLSTM are comparison baselines in the research workflow. They help separate lexical and sequence learning performance from Transformer transfer learning.

## Measured results

### ContractNLI text classification

All four runs were evaluated on the **2,091 official test pairs**. Fine-tuned legal DeBERTa uses a different evidence-selection and pooling method, so the comparison measures the complete systems.

| Model | Accuracy | Macro precision | Macro recall | Macro F1 |
| --- | ---: | ---: | ---: | ---: |
| TF-IDF + logistic regression | 68.63% | 65.11% | 64.46% | 64.04% |
| BiLSTM | 67.72% | 62.39% | 64.35% | 63.19% |
| Frozen general-domain DeBERTa | 51.89% | 44.87% | 39.98% | 35.72% |
| Fine-tuned legal DeBERTa | **87.18%** | **81.98%** | **84.45%** | **82.98%** |

The selected legal DeBERTa checkpoint scored **86.33% validation accuracy** and **80.87% validation macro F1**. Test and validation results measure document-level NLI, so they do not establish chat answer accuracy. Exported checkpoints, predictions, training curves, confusion matrices, and calibration figures are kept in the local ContractNLI artifact directory. [Text model training and calibration](docs/MODEL_TRAINING.md#current-contractnli-protocol)

### Document restoration

Means on **18 held-out ContractNLI pages** with matched clean rasters and controlled defocus, motion blur, noise or scan degradation. Windows Media OCR outputs were compared to the clean PDF text.

| Method | MSE ↓ | PSNR ↑ | SSIM ↑ | OCR character error ↓ | OCR word error ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Degraded input | 0.01232 | 19.59 dB | 0.7726 | 30.76% | 36.08% |
| Previous autoencoder | 0.00441 | 24.50 dB | 0.8781 | 21.18% | 25.93% |
| **Default DAE** | **0.00184** | **28.70 dB** | **0.9829** | **14.73%** | **20.78%** |
| Optional VAE | 0.00443 | 24.36 dB | 0.8942 | 22.44% | 30.12% |

MSE measures squared pixel error. PSNR expresses reconstruction error on a logarithmic scale. SSIM compares local luminance, contrast and structure. Character and word error rates count OCR edit operations against a reference. The benchmark quantifies controlled restorations on this corpus. A real upload needs its own clean reference to score these metrics. [Restoration protocol and import](docs/MODEL_TRAINING.md#default-denoising-autoencoder)

## Datasets and training

| Dataset | Project use | Source |
| --- | --- | --- |
| **ContractNLI** | 607 original documents and 10,319 original document–hypothesis pairs across 17 hypotheses. Three labels are Entailment, Contradiction, and NotMentioned. Primary NLI benchmark and source of contract PDF pages for the DAE. | [Official release and terms](https://stanfordnlp.github.io/contract-nli/) · [Paper](https://aclanthology.org/2021.findings-emnlp.164/) |
| **CUAD** | 510 contracts and 20,910 clause/question rows across 41 categories. Imported for clause research and benign legal-language controls. | [The Atticus Project release](https://github.com/The-Atticus-Project/cuad) · [Paper](https://arxiv.org/abs/2103.06268) |
| **LegalBench** | Three imported subsets. Test counts are 80 for contract QA, 142 for compelled-disclosure notice and 220 for competitive-restriction exceptions. | [Official task catalog](https://hazyresearch.stanford.edu/legalbench/tasks/) · [Paper](https://arxiv.org/abs/2308.11462) |
| **Fictional LAVEN pilot** | 800 cases in 100 document families for offline research. Structure verified. Independent case review and native research measurements remain pending. | [Dataset card](docs/DATASET_CARD.md) · [Generator](research/pilot.py) |

The ContractNLI preparation checks source hashes, raw annotations, and document-family overlap. Six near-duplicate training contracts were excluded. The eligible NLI data contains **7,089 train, 629 validation, 408 calibration, and 2,091 test pairs**. The official test set is intact. Legal DeBERTa selects a validation checkpoint before final test evaluation. Its evidence retriever uses matching ranked windows at inference without gold evidence.

The current DAE ran for **24 epochs** on an RTX 4050. It used **54 training, 12 validation, and 18 test PDFs**, with one selected page per document and distinct source splits. The retained VAE weights were evaluated on the same input images for the comparison above. Generation through Qwen uses the configured hosted model.

| Notebook | Purpose |
| --- | --- |
| [01 Dataset preparation](notebooks/01_dataset_preparation.ipynb) | Build eligible NLI inputs, source manifests and leakage audits. |
| [02 BiLSTM training](notebooks/02_bilstm_training.ipynb) | Train and evaluate the sequence baseline. |
| [03 DeBERTa fine-tuning](notebooks/03_deberta_finetuning.ipynb) | Train the evidence-aware legal model using the full or time-bounded Colab workflow. |
| [04 Model evaluation](notebooks/04_model_evaluation.ipynb) and [05 Uncertainty calibration](notebooks/05_uncertainty_calibration.ipynb) | Score measured predictions and fit or inspect calibration. |
| [06 Counterfactual experiments](notebooks/06_counterfactual_experiments.ipynb) and [07 Ablation study](notebooks/07_ablation_study.ipynb) | Run offline research protocols and component comparisons. |
| [08 Document reconstruction](notebooks/08_document_reconstruction.ipynb) | Reproduce the original fictional-layout AE/VAE experiment in Colab. The currently deployed DAE is trained by [the real-document restoration runner](research/text_restoration.py). |

To reproduce the v3 DAE benchmark, use the authorized ContractNLI archive and the v2 model bundle for historical comparisons and the retained VAE. The runner evaluates OCR through Windows Media OCR with English installed. See the exact commands, output formats, and artifact import instructions in the [model training guide](docs/MODEL_TRAINING.md#default-denoising-autoencoder). [Dataset sources and licenses](docs/DATASETS.md)

## Install and run

### Prerequisites

- Python 3.11 or newer and Node.js 22.16+ or 24 LTS
- A Supabase project with Email Auth, private storage and the baseline schema
- A Groq API key and access to the configured Qwen model
- Local verified reconstruction artifacts for image restoration, plus optional research artifacts for analytics

From the repository root in PowerShell, create any missing local configuration files:

~~~powershell
if (-not (Test-Path backend/.env)) { Copy-Item backend/.env.example backend/.env }
if (-not (Test-Path frontend/.env.local)) { Copy-Item frontend/.env.local.example frontend/.env.local }
~~~

Fill in your credentials and artifact paths in the local files. Apply [Supabase migration 001](supabase/001_legalshield.sql) once on a new project and create or confirm a test user. [Migration 002](supabase/002_laven_versions.sql) is the staged legal-version extension for independent verification.

| Configuration | Purpose |
| --- | --- |
| Backend SUPABASE_URL, SUPABASE_ANON_KEY, SUPABASE_SERVICE_ROLE_KEY | Auth, database and storage. Keep the service-role key server-side. |
| Backend GROQ_API_KEY, GROQ_MODEL | Shared Qwen generation and screening provider. |
| Backend EMBEDDING_MODEL, EMBEDDING_DEVICE, ALLOWED_ORIGINS | Retrieval embeddings and allowed frontend origins. |
| Backend LAVEN_RECONSTRUCTION_PATH | Path to the verified reconstruction directory. In the current workspace this points to artifacts/reconstruction/dae-v3. |
| Backend LAVEN_RESEARCH_INDEX | Path to the approved aggregate research index. |
| Backend LAVEN_ENHANCED, LAVEN_VERIFIER_PATH, LAVEN_CALIBRATION_PATH and sampling settings | Optional independently validated verification configuration. See the deployment gate above. |
| Frontend NEXT_PUBLIC_SUPABASE_URL, NEXT_PUBLIC_SUPABASE_ANON_KEY, NEXT_PUBLIC_API_URL | Public Supabase client configuration and the FastAPI URL. |

Backend terminal:

~~~powershell
cd backend
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn legalshield.api:app --host 127.0.0.1 --port 8000
~~~

Frontend terminal:

~~~powershell
cd frontend
npm ci
npm run dev
~~~

Open [LegalShield AI](http://localhost:3000) and sign in. Use **Check connections** in the account menu. The [FastAPI schema](http://127.0.0.1:8000/docs) describes the document, chat, generation, restoration and research routes.

To reset a password, choose **Forgot password?**, open the newest email link, and enter and confirm a new password. After **Password updated** appears, choose **Continue to workspace**. If the link opens the wrong site, check the frontend origin in Supabase Auth's URL configuration.

### Access to analytics

The backend reads the researcher role from Supabase Auth app metadata. A workspace administrator can grant access to a confirmed account using its exact User UID:

~~~sql
update auth.users
set raw_app_meta_data = coalesce(raw_app_meta_data, '{}'::jsonb)
  || '{"legalshield_role":"researcher"}'::jsonb
where id = 'REPLACE_WITH_USER_UID'::uuid;
~~~

Sign out and back in, then open **AI Performance & Analytics**. Saved controlled research runs can be viewed and exported, and the experiment API is researcher-only. The dashboard charts show measured values from approved exports. Sections without measured data say **Not Evaluated**. [Research protocol](docs/EXPERIMENT_PROTOCOL.md)

### Checks and troubleshooting

~~~powershell
cd backend
.\.venv\Scripts\python.exe -m pytest -q
cd ..\frontend
npm run typecheck
npm run build
~~~

| Symptom | What to check |
| --- | --- |
| Sign-in or workspace fails | Supabase URLs and keys must match in both applications. Check the user is confirmed, the API is running, and its allowed origins include the frontend. |
| Studio or Q&A does not generate | Use **Check connections** and verify that Groq accepts the configured Qwen model. Chrome or Edge can refresh the UI after a local deployment. |
| Upload stays on **Scanning and indexing** | The transfer has completed and the backend is working through the pages. Check the backend terminal for page progress and retained-chunk indexing. A completed new upload logs `POST /documents 201 Created`. If the dialog reports an error, use **Check connections** and check the backend log before retrying. |
| A scan cannot be indexed | Image-only scans can use **Document Restoration** directly. Run OCR to add a text layer before uploading to **My Documents** for Q&A. |
| Restoration weights are unavailable | Verify the absolute LAVEN_RECONSTRUCTION_PATH, checkpoint hashes and artifact manifest. Restart the backend after configuration changes. |
| Detector confidence or image quality is **Not Evaluated** | Confidence requires a calibrated detector and image quality requires a clean reference. Counts, source coverage, and actual latency are separate measurements. |
| Researcher access required | Check the account's researcher app metadata and refresh its session. |

## Research and limitations

- The hosted model receives text needed for screening or generation. Use documents cleared for the configured provider. The restoration models run locally and return separate derived images.
- Supabase Auth, row-level security, and backend ownership checks protect workspace data. Research routes require the researcher role. Keep API keys and service-role credentials in backend configuration. Local checkpoints and dataset downloads are Git-ignored.
- Exact quotes, independently verified evidence where enabled, and explicit unresolved conditions help trace answers. Legal conclusions and draft language require qualified review.
- ContractNLI concentrates on NDAs, and the restoration benchmark uses controlled degradations on a small held-out collection. Strong image similarity or low OCR error does not establish faithful recovery of every date or amount from a new scan.
- LegalBench-RAG and BIPIA are referenced research sources. No results for those benchmarks are published by this project. [References](docs/REFERENCES.md) · [Research methodology](docs/RESEARCH_METHODOLOGY.md) · [Metrics](docs/METRICS.md)

## Repository map

| Path | Contents |
| --- | --- |
| [frontend/](frontend/) | Next.js, React and TypeScript workspace, comparison charts, document viewer and account UI. |
| [backend/legalshield/](backend/legalshield/) | FastAPI routes, Supabase integration, document parsing, Groq/Qwen, retrieval, verification and restoration. |
| [backend/tests/](backend/tests/) | Backend regression tests, access checks and model integration tests. |
| [supabase/](supabase/) | Database schema, storage and RLS, plus the staged legal-version migration. |
| [research/](research/) | Dataset preparation, training, import, OCR evaluation, plots and controlled experiments. |
| [notebooks/](notebooks/) | Training and evaluation notebooks. |
| [corpus/](corpus/) and [scripts/](scripts/) | Fictional research fixtures, run summaries, verification replay and live acceptance tooling. |
| [docs/](docs/) | Architecture, methods, dataset and model cards, metrics, sources, and presentation notes. |
| artifacts/ and research-data/ | Local ignored directories for weights, reports, figures and research data. |
| [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) | Milestones, completed checks and deployment gates. |

[Architecture notes](docs/ARCHITECTURE.md) · [Dataset card](docs/DATASET_CARD.md) · [Model card](docs/MODEL_CARD.md) · [Presentation guide](docs/PRESENTATION_GUIDE.md)
