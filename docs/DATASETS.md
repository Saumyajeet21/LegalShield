# LegalShield research datasets

| Dataset | Source and rights | Measured or reported scope |
| --- | --- | --- |
| ContractNLI | [Official release and terms](https://stanfordnlp.github.io/contract-nli/) and [Koreeda and Manning, 2021](https://aclanthology.org/2021.findings-emnlp.164/). CC BY 4.0 subject to the release terms. | Verified the 2021-10-05 official archive: 607 contracts and 10,319 pairs, with 423/61/123 official train/development/test contracts. Six near-duplicate training contracts were excluded. Eligible splits are 7,089 train, 629 validation, 408 calibration and 2,091 test pairs. |
| CUAD | [Official archive](https://github.com/The-Atticus-Project/cuad/blob/67faa0e6023b04fcaae6cc09497ab00e5d63a2a2/data.zip), [CC BY 4.0](https://www.atticusprojectai.org/legal/), [Hendrycks et al., 2021](https://arxiv.org/abs/2103.06268). | Imported all 510 contracts and 20,910 clause/question rows from CUADv1.json. This includes questions with no labeled clause. Archive SHA-256 f8161d18bea4e9c05e78fa6dda61c19c846fb8087ea969c172753bc2f45b999a. |
| LegalBench | [Official repo](https://github.com/HazyResearch/legalbench), [task accommodations and rights](https://hazyresearch.stanford.edu/legalbench/tasks/), [distribution](https://huggingface.co/datasets/nguha/legalbench), [Guha et al., 2023](https://arxiv.org/abs/2308.11462). | At Git revision b46bf4ffae90524b2b72aaa30e7745fe9db64481, verified 8, 8 and 6 training rows for contract_qa, contract_nli_notice_on_compelled_disclosure and cuad_competitive_restriction_exception. At author distribution revision daec8237410aa23e3faf4bc41ad8b3a7e1696826, verified 80, 142 and 220 corresponding test rows. Separate the derived ContractNLI/CUAD tasks when checking test independence. Each chosen task lists CC BY 4.0. |
| LegalBench-RAG | [Release](https://github.com/zeroentropy-ai/legalbenchrag), [paper](https://arxiv.org/abs/2408.10343). Source-specific rights. | Authors report 6,889 queries/714 docs in v1 and 6,858 pairs in the revised abstract. Use the imported archive denominator. Retrieval offsets, precision and coverage are the target. Overlaps with ContractNLI and CUAD must be reported. |
| BIPIA | [Microsoft repo](https://github.com/microsoft/BIPIA), [paper](https://arxiv.org/abs/2312.14197). MIT software plus separate task data terms. | External literature reference. The importer accepts only a reviewed clean derivative and no such input is currently imported. |
| LAVEN | Authored fictional fixtures, redistribution review pending. | Generated 800 candidate cases in 100 template families. Human review is pending. These cases are development material. |

Run research.datasets with download, the HTTPS release URL and pinned
checksum for permitted archives. Use the dataset-specific importer on the
approved, extracted local inputs. It writes JSONL and a checksum/count
manifest. For ContractNLI use the directory containing train.json, dev.json and
test.json. For CUAD use the archive's individual native split file with its
split label, or CUADv1.json with partition full. For LegalBench use the
tab-separated task file and supply --task and --partition. For LegalBench-RAG
supply each benchmark JSON and --corpus path. Keep the corpus raw for offset
checks. Use research.datasets audit to find cross-partition source and text
overlap in inputs with family/partition fields. Exact and shingle-based audits
are heuristics for content similarity. Source lineage review still determines
independence. A separate grouped subset of official development is reserved
for calibration with --calibration-fraction. Never train on official test.
