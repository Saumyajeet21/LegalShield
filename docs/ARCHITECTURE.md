# LegalShield architecture

The preserved baseline is Git commit 06a50f97e973baf2755e34905adf7fda1442d71e.
Next.js authenticates with Supabase and calls FastAPI. Upload extracts text from
PDF, DOCX and TXT, preserves original bytes and their SHA-256 in private
storage, and indexes retained excerpts after LAVEN screening. PDF.js shows
source coordinates when available. Normalized 384-dimensional Sentence
Transformers embeddings are searched by Supabase pgvector. Groq Qwen drafts
and verifies cited claims, and checks referenced authority against trusted
workspace excerpts. The owner-scoped API persists turns. Research experiments
require the Supabase Auth app-metadata researcher role and use synthetic
documents.

The baseline uses index revisions, which change on reindexing. Legal lineage
is explicit research metadata in the upgrade. The baseline citation normalizer
also applied normalized character indices to original text. That mismatch
needs explicit raw offsets when validating evidence. Use the recorded commit
for exact baseline comparisons.



The optional enhanced path resolves applicable owner-scoped full-document
versions first, retrieves retained excerpts, validates the structured Qwen
candidate, checks original quotation coordinates and runs an independent
DeBERTa verifier. The typed Z3 compiler checks only encoded conditions and
facts. Bounded sampling and correction can abstain. Evidence records live
with the saved turn and support authenticated PDF export. Research
aggregates are separate approved artifacts behind the researcher role.
