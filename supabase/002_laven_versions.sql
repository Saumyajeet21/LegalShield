begin;

-- A reviewed full instrument version. Overlapping dates are resolved in the
-- application by explicit precedence. Concurrent conflicting priorities abstain.
create table public.legal_document_versions (
  owner uuid not null,
  document_id uuid not null,
  family_id uuid not null,
  version_label text not null check (char_length(version_label) between 1 and 80),
  sha256 text not null check (sha256 ~ '^[0-9a-f]{64}$'),
  authority_status text not null check (authority_status in ('verified','unverified','revoked')),
  source_reference text not null check (char_length(source_reference) between 1 and 500),
  effective_from date not null,
  effective_to date,
  precedence integer not null check (precedence between 0 and 10000),
  created_at timestamptz not null default now(),
  foreign key (document_id, owner) references public.documents(id,owner) on delete cascade,
  primary key (owner, document_id),
  unique (owner, family_id, version_label),
  check (effective_to is null or effective_to > effective_from)
);
create index legal_document_versions_lookup on public.legal_document_versions(owner,family_id,effective_from);
alter table public.legal_document_versions enable row level security;
revoke all on public.legal_document_versions from public, anon, authenticated;
grant select on public.legal_document_versions to authenticated;
grant all on public.legal_document_versions to service_role;
create policy read_own_versions on public.legal_document_versions for select to authenticated
  using ((select auth.uid()) = owner);

commit;
