# Fictional LAVEN pilot card

Run python -m research.pilot artifacts/data/laven-pilot.jsonl.
Default seed 13 generates 100 families, 800 records and both JSONL and CSV.
Families are assigned 70/10/10/10 across training, validation, calibration and
test. Repeated templates limit generalization. This is a template-controlled
feasibility set with deterministic reference answers and pending human review.

Identity and origin fields include sample_id, scenario_family_id, document_id,
source_dataset, source_url and dataset_license. Original/variant text fields
carry document_version, modification_type, authority_status and authority_evidence.
Each annotation carries question, expected_answer, expected_evidence,
expected_answerability, expected_system_action, legal_conditions,
legal_exceptions, verification_label, injection_label, suspicious_text_span,
span_start and span_end. Spans use zero-based, end-exclusive Python Unicode
code points. synthetic_data_indicator is always true. annotation_source,
human_review_status and dataset_partition describe eligibility.

Eight variants include an intact original, a reviewed amended version fixture,
an unauthorized change, a conflict, missing evidence, an irrelevant addition,
an approved exception case and an inert marker. The marker contains no
behavioral instructions. It supports integrity controls and is not an attack
success dataset. Distinguish fixture-level authority statuses from the
review process for the dataset. Have two independent reviewers label the
answers, version applicability, conditions and spans, then adjudicate
disagreements before setting approved. Pending cases are excluded by the
live pilot runner.


Use research.scenario_paraphrases for bounded Qwen wording augmentation of
training-only questions. Its estimate mode makes no calls. Any generated
paraphrase inherits its family/partition, remains in pending review and needs
an independent check that its deterministic reference still applies.
