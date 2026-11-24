# LegalShield-RAG

LegalShield AI is a document-grounded legal research workspace. Its experimental LAVEN pipeline combines instruction-target analysis with exact quote localization, selective isolation, citation support checks, and comparison against an explicitly trusted reference corpus.

## Docker-free setup

Use Python 3.11+ and Node.js 22.16+ or 24 LTS. You need a Supabase project with email/password Auth and a Groq API key with access to your Qwen model. Prefer a fresh Supabase project for acceptance testing.

1. Run `supabase/001_legalshield.sql` once in your Supabase SQL Editor. It creates the schema, vector search, RLS policies, and private originals bucket. Create confirmed email/password users from the Auth dashboard. Public registration is unnecessary.
2. Copy `backend/.env.example` to `backend/.env` and `frontend/.env.local.example` to `frontend/.env.local`. Set the same Supabase URL and publishable key in both. Set the backend-only service role key, Groq API key, and your existing Qwen model ID in `GROQ_MODEL`. Check the configured model in the account's Groq model list if the app reports that it is unavailable. Keep the backend env file private.
3. Start FastAPI from the `backend` directory.

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade "pip>=26.2" "setuptools>=84.1"
   # Linux CPU install. macOS users can install directly from requirements.txt.
   python -m pip install 'torch>=2.6,<3' --index-url https://download.pytorch.org/whl/cpu
   python -m pip install -r requirements.txt
   python -m uvicorn legalshield.api:app --host 127.0.0.1 --port 8000
   ```

   On Windows PowerShell activate with `.venv\Scripts\Activate.ps1` and use double quotes around the Torch version range. The embedding model downloads public weights on first use, or you can set `EMBEDDING_MODEL` to a local 384-dimensional Sentence Transformers model directory. Reindex documents if you change it. A model change of the same dimension still requires reindexing every document before comparing vectors.

4. Start Next.js in a second terminal from `frontend`.

   ```bash
   npm ci
   npm run dev
   ```

5. Open `http://localhost:3000`, sign in, and choose **Check connections** from the profile menu. Upload a text-based PDF, DOCX, or UTF-8 TXT. Select the documents in the assistant and ask a legal question. Open a source from a citation or security finding to review the original page.

## Document handling

Text-based PDFs support red overlays at mapped text positions, including rotated pages. DOCX, TXT, and findings without complete PDF geometry use the extracted text view. Image-only scans require OCR before upload. The uploader accepts files up to 20 MiB, PDFs up to 300 pages, and up to 200,000 extracted characters.

Original files remain byte-for-byte intact in private storage while a separate index holds retained excerpts. The reviewer can see excerpts removed by the detector and passages quarantined for review. Each substantive claim needs a literal citation and a model support check. Authority verification is reported as unavailable when the trusted corpus lacks a usable source. The citation check is model-assisted and should be reviewed by a legal professional.

Groq receives extracted text during security scanning and relevant excerpts during answer generation and verification. Use the application with material you are authorized to send to your Groq account and according to that account's data controls. Originals are served only after checking document ownership. Metadata changes and index replacement are backend operations. Deleting a document removes its original, index, experiments, and saved answers based on it. The audit trail retains the document ID and lifecycle events.

## Research laboratory

Assign the role by merging `{"legalshield_role":"researcher"}` into the dedicated user's Auth **app metadata** with the Supabase admin API or SQL Editor, then sign out and back in to refresh the session. For example, run this with the intended user's ID.

```sql
update auth.users
set raw_app_meta_data = coalesce(raw_app_meta_data, '{}'::jsonb) || '{"legalshield_role":"researcher"}'::jsonb
where id = '00000000-0000-0000-0000-000000000000';
```

Upload an experiment document marked synthetic and open **Research laboratory** from the profile menu. Gold span coordinates are zero-based and end-exclusive, counted in Unicode code points in the extracted page. Supply a unique inert marker present in the synthetic test passage and absent from the question. Keep development and test documents separate. To create a small labeled corpus with independent splits and clean controls, run `python scripts/generate_corpus.py corpus` from the project root using the backend environment. The manifest provides the form values for each document. A trusted reference belongs to the researcher's own workspace and should come from an authoritative source.

Paired runs retrieve the top eight chunks by cosine similarity on original text and use the same chunk IDs, model, temperature, and output format in the original and protected runs. The protected run uses retained excerpts from those chunks. Both use the same system prompt. The application records marker observation, verified citation support, detection localization, evidence retention, and latency. Reference phrase matching is a proxy for answer correctness. An observed marker is evidence for the specified synthetic outcome only. Export the run data as NDJSON, then aggregate test runs with `python scripts/summarize_runs.py export.ndjson`.

## Verification

In `backend`, run `python -m pytest -q`. In `frontend`, run `npm run build` and `npm run typecheck`.

For live acceptance, start FastAPI with the configured services. From `backend`, set `VERIFY_OWNER_EMAIL`, `VERIFY_OWNER_PASSWORD`, `VERIFY_OUTSIDER_EMAIL`, and `VERIFY_OUTSIDER_PASSWORD` in your shell for two disposable accounts. The first needs the researcher role and the second needs the ordinary role. Run `python ../scripts/verify_live.py`. It creates and deletes synthetic PDFs, checks the model and vector workflow, verifies access isolation, checks exact PDF coordinates and legal evidence, and saves and reads a research experiment during the run. Inspect the visible overlay in the browser as well.
