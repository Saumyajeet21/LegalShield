# Experiment protocol

Before evaluating, freeze eligible human-reviewed family IDs, data/checkpoint
hashes, candidate budget, model revisions, and scoring code. Use the same
evidence track across classifiers. Retain separate development, calibration
and test families. Score by answer action and typed answer where applicable.
Compare M7 and M3 with conditional change sensitivity, invariance and
coverage. Use family bootstrap for paired correctness. Compare reference and
variant jointly and keep ablations separate from deployment settings.

Run python -m research.run_pilot DATA.jsonl OUTPUT.jsonl --systems M1 M2 M3
to estimate scope without spending. Add --execute --max-calls 12 only for a
reviewed bounded pilot, with configured Groq access. The runner uses fictional
single-instrument inputs and a simplified one-excerpt offline RAG fixture.
Its M3 uses the existing LAVEN detection, retention and grounding functions.
Tag these offline approximations as such. The native system comparison still
needs a staged Supabase run. For enhanced profiles supply --model-path and
--calibration. The output records the configured model, action, typed answer,
latency and logical call count. Each logical completion can make up to three
transport attempts.

Score with python -m research.experiments GOLD.jsonl PREDICTIONS.jsonl REPORT.json.
Use the plots module to export PNG/SVG/PDF. The default scoring mode excludes
unreviewed references. The illustrative full matrix budget in the approved
plan was around 3,600 logical completions for 40 question/variant instances,
or US$9.36 under the stated Qwen rates and token assumptions. Measure tokens
and failure rates in the pilot, then confirm an updated ceiling with the user
before any expensive run. No paid experiment has been executed in this branch.



## Staging acceptance and four-system research pilot

Use a separate Supabase project. In its SQL Editor, apply 001 only when
the baseline tables do not exist, then apply 002 once to add legal version
metadata. The reviewed SQL enables RLS on version, document, chat and
experiment tables. It makes the originals bucket private and places
verification records in owner-scoped chat-turn results. Research
experiments require the researcher role. This is a source inspection.
Live access and storage were not exercised in this pass.

Set staging URLs/keys and configure the existing two fixture identities
expected by scripts/verify_live.py. Run it from backend and inspect both
the normal and upgraded analyst flows. Check that a separate owner cannot
read the original bytes, legal version metadata, saved verification or
reports, and that an ordinary user is denied researcher endpoints. Confirm
the researcher sees aggregate plots and can save a synthetic experiment.
Enable the optional verifier only with a deployable Colab checkpoint.

After independent review and model integration, an eight-instance pilot
with M1, M2, M3 and M7 is estimated at 104 logical completions at averages
of 1, 1, 6 and 5 per instance. Cap it at 120 logical completions. At 1,500
input and 350 output tokens per call, that cap budgets 180,000 input and
42,000 output tokens, or about US$0.31 using illustrative US$0.80 and US$4.00
per million tokens. Reprice using the actual Groq model and request settings
before launch. The transport may retry failed attempts. Budget approval and
independent review are still required for paid runs.

Use the existing scorer for answer correctness, abstention, version
robustness, coverage and latency. Record claim and citation observations
in evidence records, with sensitivity/invariance eligibility set before
running. Keep the existing preserved-span and localization evaluation on
approved defensive fixtures. The inert fictional pilot does not supply
an attack success rate.
