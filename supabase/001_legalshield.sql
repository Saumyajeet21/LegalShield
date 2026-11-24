begin;
create schema if not exists extensions;
create extension if not exists vector with schema extensions;
grant usage on schema extensions to authenticated, service_role;
set local search_path = public, extensions;

create table public.documents (
  id uuid primary key default gen_random_uuid(),
  owner uuid not null references auth.users on delete cascade,
  name text not null,
  media_type text not null,
  category text not null,
  object_path text not null unique,
  sha256 text not null,
  page_count integer not null check (page_count > 0),
  finding_count integer not null default 0,
  status text not null check (status in ('processing','indexed','deleting')),
  synthetic boolean not null default false,
  trusted boolean not null default false,
  revision integer not null default 0,
  created_at timestamptz not null default now(),
  unique(id, owner)
);
create index on public.documents(owner, created_at desc);

create table public.document_pages (
  document_id uuid not null,
  owner uuid not null,
  page integer not null check (page > 0),
  text text not null,
  sanitized_text text not null,
  primary key(document_id, page),
  foreign key(document_id,owner) references public.documents(id,owner) on delete cascade
);
create table public.document_chunks (
  id bigint generated always as identity primary key,
  document_id uuid not null,
  owner uuid not null,
  page integer not null,
  clause text,
  start_offset integer not null,
  end_offset integer not null check (end_offset > start_offset),
  parts jsonb not null,
  clean_text text not null,
  removed integer not null,
  embedding extensions.vector(384),
  foreign key(document_id,owner) references public.documents(id,owner) on delete cascade,
  foreign key(document_id,page) references public.document_pages(document_id,page) on delete cascade
);
create index on public.document_chunks(owner,document_id);
create index on public.document_chunks using hnsw (embedding vector_cosine_ops) where embedding is not null;

create table public.security_findings (
  id uuid primary key default gen_random_uuid(),
  document_id uuid not null,
  owner uuid not null,
  page integer not null,
  start_offset integer not null,
  end_offset integer not null check (end_offset > start_offset),
  quote text not null,
  boxes jsonb not null,
  category text not null,
  severity text not null check (severity in ('low','medium','high','critical')),
  reason text not null,
  action text not null check (action in ('isolated','quarantined','allowed')),
  foreign key(document_id,owner) references public.documents(id,owner) on delete cascade,
  foreign key(document_id,page) references public.document_pages(document_id,page) on delete cascade
);
create index on public.security_findings(owner,document_id,page);

create table public.conversations (
  id uuid primary key default gen_random_uuid(),
  owner uuid not null references auth.users on delete cascade,
  title text not null,
  created_at timestamptz not null default now(),
  unique(id,owner)
);
create index on public.conversations(owner, created_at desc);

create table public.chat_turns (
  id bigint generated always as identity primary key,
  conversation_id uuid not null,
  owner uuid not null,
  question text not null,
  result jsonb not null,
  created_at timestamptz not null default now(),
  foreign key(conversation_id,owner) references public.conversations(id,owner) on delete cascade
);
create index on public.chat_turns(owner,conversation_id,id);

create table public.audit_events (
  id bigint generated always as identity primary key,
  owner uuid not null references auth.users on delete cascade,
  document_id uuid,
  event text not null,
  details jsonb not null default '{}',
  created_at timestamptz not null default now()
);
create table public.research_experiments (
  id uuid primary key default gen_random_uuid(),
  owner uuid not null references auth.users on delete cascade,
  document_id uuid not null,
  report jsonb not null,
  created_at timestamptz not null default now(),
  foreign key(document_id,owner) references public.documents(id,owner) on delete cascade
);

do $$
declare relation text;
begin
  foreach relation in array array['documents','document_pages','document_chunks','security_findings','conversations','chat_turns','audit_events','research_experiments'] loop
    execute format('alter table public.%I enable row level security', relation);
    execute format('revoke all on public.%I from public, anon, authenticated', relation);
    execute format('grant select on public.%I to authenticated', relation);
    execute format('grant all on public.%I to service_role', relation);
    execute format('create policy read_own on public.%I for select to authenticated using ((select auth.uid()) = owner%s)', relation,
      case when relation = 'research_experiments' then ' and (select auth.jwt())->''app_metadata''->>''legalshield_role'' = ''researcher''' else '' end);
  end loop;
end $$;

create function public.legalshield_replace_index(
  p_owner uuid, p_document uuid, p_revision integer, p_pages jsonb, p_chunks jsonb, p_findings jsonb
) returns boolean language plpgsql security invoker set search_path = public, extensions, pg_temp as $$
declare old_status text;
begin
  select status into old_status from documents where id = p_document and owner = p_owner
    and revision = p_revision and status in ('processing', 'indexed') for update;
  if not found then return false; end if;
  delete from document_pages where owner = p_owner and document_id = p_document;
  insert into document_pages(document_id,owner,page,text,sanitized_text)
    select p_document,p_owner,r.page,r.text,r.sanitized_text
    from jsonb_to_recordset(p_pages) r(page integer,text text,sanitized_text text);
  insert into document_chunks(document_id,owner,page,clause,start_offset,end_offset,parts,clean_text,removed,embedding)
    select p_document,p_owner,r.page,r.clause,r.start_offset,r.end_offset,r.parts,r.clean_text,r.removed,r.embedding::text::extensions.vector(384)
    from jsonb_to_recordset(p_chunks) r(page integer,clause text,start_offset integer,end_offset integer,parts jsonb,
      clean_text text,removed integer,embedding jsonb);
  insert into security_findings(document_id,owner,page,start_offset,end_offset,quote,boxes,category,severity,reason,action)
    select p_document,p_owner,r.page,r.start_offset,r.end_offset,r.quote,r.boxes,r.category,r.severity,r.reason,r.action
    from jsonb_to_recordset(p_findings) r(page integer,start_offset integer,end_offset integer,quote text,boxes jsonb,
      category text,severity text,reason text,action text);
  update documents set status = 'indexed',revision = revision + 1,finding_count = jsonb_array_length(p_findings)
    where id = p_document and owner = p_owner;
  insert into audit_events(owner,document_id,event,details)
    values(p_owner,p_document,case when old_status = 'indexed' then 'reindexed' else 'indexed' end,
      jsonb_build_object('finding_count',jsonb_array_length(p_findings),'revision',p_revision + 1));
  return true;
end $$;

create function public.legalshield_match(
  p_embedding extensions.vector(384), p_documents uuid[], p_limit integer, p_trusted boolean default false
) returns table(id bigint,document_id uuid,name text,page integer,clause text,start_offset integer,
  end_offset integer,parts jsonb,similarity double precision,revision integer)
language sql stable security invoker set search_path = public, extensions, pg_temp as $$
  select c.id, c.document_id, d.name, c.page, c.clause, c.start_offset, c.end_offset, c.parts,
    1 - (c.embedding <=> p_embedding), d.revision
  from document_chunks c join documents d on d.id = c.document_id and d.owner = c.owner
  where c.owner = (select auth.uid()) and c.embedding is not null and d.status = 'indexed'
    and case when p_trusted then d.trusted else d.id = any(p_documents) end
  order by c.embedding <=> p_embedding
  limit least(greatest(p_limit,1),24)
$$;

revoke execute on function public.legalshield_replace_index(uuid,uuid,integer,jsonb,jsonb,jsonb) from public, anon, authenticated;
grant execute on function public.legalshield_replace_index(uuid,uuid,integer,jsonb,jsonb,jsonb) to service_role;
revoke execute on function public.legalshield_match(extensions.vector,uuid[],integer,boolean) from public, anon;
grant execute on function public.legalshield_match(extensions.vector,uuid[],integer,boolean) to authenticated;
insert into storage.buckets(id,name,public,file_size_limit)
  values ('legalshield-originals','legalshield-originals',false,20971520)
  on conflict (id) do update set public = false, file_size_limit = excluded.file_size_limit;
create policy legalshield_private_originals on storage.objects as restrictive for all to anon, authenticated
  using (bucket_id <> 'legalshield-originals') with check (bucket_id <> 'legalshield-originals');
create function public.legalshield_save_turn(p_owner uuid,p_conversation uuid,p_create boolean,p_question text,p_result jsonb)
returns boolean language plpgsql security invoker set search_path = public, pg_temp as $$
declare source_id uuid;
begin
  for source_id in
    select value::uuid from jsonb_array_elements_text(p_result->'document_ids')
    union
    select (source #>> '{}')::uuid from jsonb_path_query(p_result, '$.claims[*].authority.source.document_id') as refs(source)
    order by 1
  loop
    perform 1 from documents where id = source_id and owner = p_owner and status = 'indexed' for update;
    if not found then return false; end if;
  end loop;
  if p_create then
    insert into conversations(id,owner,title) values(p_conversation,p_owner,left(p_question,100));
  end if;
  insert into chat_turns(conversation_id,owner,question,result) values(p_conversation,p_owner,p_question,p_result);
  return true;
end $$;
create function public.legalshield_delete_document(p_owner uuid,p_document uuid)
returns void language plpgsql security invoker set search_path = public, pg_temp as $$
begin
  delete from documents where owner = p_owner and id = p_document and status = 'deleting';
  if found then
    delete from chat_turns where owner = p_owner and
      ((result->'document_ids') ? p_document::text or jsonb_path_exists(result, '$.claims[*].authority.source ? (@.document_id == $doc)',
        jsonb_build_object('doc', p_document::text)));
    delete from conversations c where owner = p_owner and not exists
      (select 1 from chat_turns t where t.conversation_id = c.id and t.owner = c.owner);
    insert into audit_events(owner,document_id,event) values(p_owner,p_document,'deleted');
  end if;
end $$;
revoke execute on function public.legalshield_save_turn(uuid,uuid,boolean,text,jsonb) from public, anon, authenticated;
revoke execute on function public.legalshield_delete_document(uuid,uuid) from public, anon, authenticated;
grant execute on function public.legalshield_save_turn(uuid,uuid,boolean,text,jsonb) to service_role;
grant execute on function public.legalshield_delete_document(uuid,uuid) to service_role;
grant usage,select on sequence public.chat_turns_id_seq, public.audit_events_id_seq, public.document_chunks_id_seq to service_role;
commit;
