"use client";

import { useEffect, useRef, useState, type FormEvent } from "react";
import type { Session } from "@supabase/supabase-js";
import { motion } from "motion/react";
import Markdown from "react-markdown";
import { ArrowDownToLine, ArrowRight, ArrowUp, Bell, BookOpen, Check, ChevronDown, CircleAlert, Copy, FileText, FlaskConical, FolderClosed, Landmark, LogOut, Menu, MessageSquare, Paperclip, Plus, RefreshCw, Search, Settings, ShieldAlert, ShieldCheck, UploadCloud, X } from "lucide-react";
import { original, request, supabase, upload, type Answer, type Conversation, type Doc, type Finding, type Target, type Turn } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Viewer, downloadBlob } from "@/components/viewer";
import { Research } from "@/components/research";

type View = "assistant" | "documents" | "alerts" | "research";
const NAV = [{id: "assistant", label: "AI Legal Assistant", icon: MessageSquare}, {id: "documents", label: "My Documents", icon: FolderClosed}, {id: "alerts", label: "Security Alerts", icon: ShieldAlert}] as const;
const SUGGESTIONS = ["What are the obligations of the parties in this contract?", "Summarize the important clauses of this agreement.", "What compliance requirements apply under this regulation?", "Are there any conflicting or unusual provisions?"];
const CATEGORIES = ["Contract", "Legislation", "Court judgment", "Regulation", "Legal notice", "Other"];

function Brand() {
  return <div className="brand"><span className="brand-icon"><ShieldCheck size={24} strokeWidth={1.6}/></span><span>LegalShield <b>AI</b></span></div>;
}

function Illustration() {
  return <svg aria-hidden="true" className="court" width="142" height="86" viewBox="0 0 142 86" fill="none">
    <path d="M9 73h124M21 67h100M30 61h82M31 31l40-20 40 20H31Zm7 6h66M43 39v20m13-20v20m15-20v20m15-20v20m13-20v20" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
    <circle cx="71" cy="23" r="3" stroke="currentColor" strokeWidth="1.3"/>
    <path d="M18 39v10m-5-5h10m99-24v12m-6-6h12M30 5v8m-4-4h8" stroke="currentColor" strokeWidth="1.1" opacity=".6"/>
  </svg>;
}

type AuthView = "signin" | "signup" | "forgot" | "forgot-sent";

function SignIn() {
  const [view, setView] = useState<AuthView>("signin");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  function switchView(next: AuthView) { setView(next); setError(""); }

  async function login(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!supabase) return;
    const fields = new FormData(event.currentTarget);
    setBusy(true); setError("");
    try {
      const result = await supabase.auth.signInWithPassword({email: String(fields.get("email")), password: String(fields.get("password"))});
      if (result.error) setError(result.error.message);
    } catch { setError("Supabase Auth could not be reached."); }
    finally { setBusy(false); }
  }

  async function signup(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!supabase) return;
    const fields = new FormData(event.currentTarget);
    const password = String(fields.get("password"));
    const confirm = String(fields.get("confirm"));
    if (password !== confirm) { setError("Passwords do not match."); return; }
    if (password.length < 8) { setError("Password must be at least 8 characters."); return; }
    setBusy(true); setError("");
    try {
      const result = await supabase.auth.signUp({email: String(fields.get("email")), password});
      if (result.error) setError(result.error.message);
      else if (result.data.session) { /* auto-confirmed, onAuthStateChange handles redirect */ }
      else setError("Check your email for a confirmation link before signing in.");
    } catch { setError("Supabase Auth could not be reached."); }
    finally { setBusy(false); }
  }

  async function forgot(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!supabase) return;
    const fields = new FormData(event.currentTarget);
    setBusy(true); setError("");
    try {
      const result = await supabase.auth.resetPasswordForEmail(String(fields.get("email")), {
        redirectTo: typeof window !== "undefined" ? window.location.origin : undefined,
      });
      if (result.error) setError(result.error.message);
      else switchView("forgot-sent");
    } catch { setError("Supabase Auth could not be reached."); }
    finally { setBusy(false); }
  }

  const visual = <div className="login-visual">
    <Brand/>
    <div><Illustration/><h1>Legal research<br/>with evidence in view.</h1><p>Review your documents, trace legal answers to their sources, and inspect suspicious instructions in context.</p></div>
  </div>;

  if (view === "forgot-sent") return <main className="login">{visual}
    <div className="login-content"><div className="form-stack login-form">
      <h2>Check your email</h2>
      <p>If an account exists for that address, a password reset link is on its way. Check your spam folder if it doesn't arrive within a minute.</p>
      <Button variant="outline" onClick={() => switchView("signin")}>Back to sign in</Button>
    </div></div>
  </main>;

  if (view === "forgot") return <main className="login">{visual}
    <div className="login-content"><form className="form-stack login-form" onSubmit={forgot}>
      <h2>Reset password</h2><p>Enter your email and we'll send a reset link.</p>
      {!supabase && <div className="notice warning" role="status"><strong>Workspace setup required</strong><p>Configure the Supabase URL and public key in the frontend environment file, then restart the app.</p></div>}
      {error && <p role="alert" className="notice error">{error}</p>}
      <label>Email address<input name="email" type="email" autoComplete="username" required disabled={!supabase}/></label>
      <Button type="submit" disabled={!supabase || busy}>{busy ? "Sending…" : "Send reset link"}<ArrowRight size={17}/></Button>
      <p className="auth-switch">Remembered it? <button type="button" className="text-link" onClick={() => switchView("signin")}>Sign in</button></p>
    </form></div>
  </main>;

  if (view === "signup") return <main className="login">{visual}
    <div className="login-content"><form className="form-stack login-form" onSubmit={signup}>
      <h2>Create account</h2><p>Set up your private legal workspace.</p>
      {!supabase && <div className="notice warning" role="status"><strong>Workspace setup required</strong><p>Configure the Supabase URL and public key in the frontend environment file, then restart the app.</p></div>}
      {error && <p role="alert" className="notice error">{error}</p>}
      <label>Email address<input name="email" type="email" autoComplete="username" required disabled={!supabase}/></label>
      <label>Password<input name="password" type="password" autoComplete="new-password" required disabled={!supabase}/><small>At least 8 characters.</small></label>
      <label>Confirm password<input name="confirm" type="password" autoComplete="new-password" required disabled={!supabase}/></label>
      <Button type="submit" disabled={!supabase || busy}>{busy ? "Creating account…" : "Create account"}<ArrowRight size={17}/></Button>
      <p className="auth-switch">Already have an account? <button type="button" className="text-link" onClick={() => switchView("signin")}>Sign in</button></p>
    </form></div>
  </main>;

  return <main className="login">{visual}
    <div className="login-content"><form className="form-stack login-form" onSubmit={login}><h2>Sign in</h2><p>Access your private legal workspace.</p>
      {!supabase && <div className="notice warning" role="status"><strong>Workspace setup required</strong><p>Configure the Supabase URL and public key in the frontend environment file, then restart the app.</p></div>}
      {error && <p role="alert" className="notice error">{error}</p>}
      <label>Email address<input name="email" type="email" autoComplete="username" required disabled={!supabase}/></label>
      <label>
        <span className="label-row">Password <button type="button" className="text-link forgot-link" onClick={() => switchView("forgot")} disabled={!supabase}>Forgot password?</button></span>
        <input name="password" type="password" autoComplete="current-password" required disabled={!supabase}/>
      </label>
      <Button type="submit" disabled={!supabase || busy}>{busy ? "Signing in…" : "Sign in"}<ArrowRight size={17}/></Button>
      <p className="auth-switch">Don't have an account? <button type="button" className="text-link" onClick={() => switchView("signup")}>Create one</button></p>
    </form></div>
  </main>;
}

function FindingCard({item, onOpen}: {item: Finding; onOpen: () => void}) {
  return <article className="finding">
    <div className="finding-title"><ShieldAlert size={19}/><strong>{item.category.replaceAll("_", " ")}</strong><span className={`severity ${item.severity}`}>{item.severity}</span></div>
    <small>{item.name || "Source document"} · Page {item.page} · {item.action === "quarantined" ? "Passage quarantined" : item.action === "isolated" ? "Instruction isolated" : "Allowed after review"}</small>
    {item.quote && <blockquote>{item.quote}</blockquote>}
    <p>{item.reason}</p>
    <Button variant="outline" onClick={onOpen}>Open source <ArrowRight size={14}/></Button>
  </article>;
}

function AnswerCard({answer, open}: {answer: Answer; open: (target: Target) => void}) {
  const [copied, setCopied] = useState(false);
  const [copyError, setCopyError] = useState(false);
  return <div className="answer-layout">
    <section className="answer-card">
      <header><div className="row"><ShieldCheck size={19}/><h3>Legal answer</h3></div>
        {answer.claims.length > 0 && <Button variant="ghost" aria-label={copied ? "Copied answer" : "Copy answer"} onClick={async () => {
          try { await navigator.clipboard.writeText(answer.claims.map(claim => claim.text).join("\n\n")); setCopied(true); setCopyError(false); }
          catch { setCopyError(true); }
        }}>{copied ? <Check size={16}/> : <Copy size={16}/>}</Button>}</header>
      {copyError && <p className="muted" role="alert">Clipboard access is unavailable.</p>}
      {answer.message && <p>{answer.message}</p>}
      {answer.claims.map((claim, index) => <div key={index} className="claim">
        <Markdown skipHtml allowedElements={["p", "strong", "em", "ul", "ol", "li", "code", "blockquote", "br"]} unwrapDisallowed>{claim.text}</Markdown>
        <div className="chips">{claim.citations.map((citation, i) => <button type="button" className="citation" key={i}
          onClick={() => open({documentId: citation.document_id, page: citation.page, citation})}><BookOpen size={13}/> {citation.name} · p. {citation.page}{citation.clause && ` · ${citation.clause}`}</button>)}</div>
        {claim.authority && <p className="authority">Authority check: {claim.authority.status === "supported" ? "Supported by the trusted corpus" : "Unable to verify"}. {claim.authority.references.join(", ")}
          {claim.authority.source && <button className="text-link" type="button" onClick={() => open({documentId: claim.authority!.source!.document_id, page: claim.authority!.source!.page})}>Open trusted source</button>}</p>}
      </div>)}
    </section>
    {answer.alerts.length > 0 && <section className="question-alerts"><h3><ShieldAlert size={18}/> Security warning</h3>
      <p>{answer.claims.length ? "The answer was checked against retained evidence after the flagged material was isolated." : "No supported answer was returned after applying document and question checks."}</p>
      {answer.alerts.map(item => <FindingCard key={item.id} item={item} onOpen={() => open({documentId: item.document_id, page: item.page, findingId: item.id})}/>)}
    </section>}
  </div>;
}

function UploadDialog({session, researcher, initial, close, complete}: {session: Session; researcher: boolean; initial?: File; close: () => void; complete: (doc: Doc) => void}) {
  const dialog = useRef<HTMLDialogElement>(null);
  const controller = useRef<AbortController | null>(null);
  const [file, setFile] = useState(initial);
  const [progress, setProgress] = useState<number | null>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    dialog.current?.showModal();
    return () => controller.current?.abort();
  }, []);
  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!file || !/\.(pdf|docx|txt)$/i.test(file.name) || file.size === 0 || file.size > 20 * 1024 * 1024) {
      setError("Choose a nonempty PDF, DOCX, or TXT file of up to 20 MiB."); return;
    }
    const data = new FormData(event.currentTarget);
    data.set("file", file);
    if (data.has("trusted") && data.has("synthetic")) {setError("Choose either synthetic or trusted reference."); return;}
    setError(""); setProgress(0);
    const operation = new AbortController();
    controller.current = operation;
    try { complete(await upload(session, data, setProgress, operation.signal)); }
    catch (failure) { if (!operation.signal.aborted) setError(failure instanceof Error ? failure.message : "Upload failed."); }
    finally { if (!operation.signal.aborted) setProgress(null); }
  }
  return <dialog ref={dialog} className="small-dialog" aria-labelledby="upload-heading" onCancel={event => {if (progress !== null) event.preventDefault(); else close();}}>
    <form className="form-stack" onSubmit={submit}>
      <div className="page-heading"><h2 id="upload-heading">Upload document</h2><Button variant="ghost" aria-label="Close upload" onClick={close} disabled={progress !== null}><X size={20}/></Button></div>
      <label className="drop-zone" onDragOver={event => event.preventDefault()} onDrop={event => {event.preventDefault(); if (progress === null) setFile(event.dataTransfer.files[0]);}}>
        <UploadCloud size={24}/><strong>{file?.name || "Choose a file or drag it here"}</strong><small>PDF, DOCX or TXT · Up to 20 MiB</small>
        <input className="file-input" type="file" accept=".pdf,.docx,.txt" aria-label="Select a file" disabled={progress !== null} onChange={event => setFile(event.target.files?.[0])}/>
      </label>
      <label>Category<select name="category" defaultValue="Other" disabled={progress !== null}>{CATEGORIES.map(category => <option key={category}>{category}</option>)}</select></label>
      {researcher && <><label className="check"><input name="synthetic" type="checkbox" value="true" disabled={progress !== null}/> Synthetic research document</label><label className="check"><input name="trusted" type="checkbox" value="true" disabled={progress !== null}/> Explicitly trusted legal reference</label></>}
      {error && <p role="alert" className="notice error">{error}</p>}
      {progress !== null && <div role="status"><progress value={progress < 100 ? progress : undefined} max={100}/><p className="muted">{progress >= 100 ? "Scanning and indexing document…" : `Uploading ${progress}%`}</p></div>}
      <Button type="submit" disabled={!file || progress !== null}><UploadCloud size={17}/> Upload and scan</Button>
    </form>
  </dialog>;
}

function Workspace({session}: {session: Session}) {
  const [view, setView] = useState<View>("assistant");
  const [docs, setDocs] = useState<Doc[]>([]);
  const [alerts, setAlerts] = useState<Finding[]>([]);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const [conversation, setConversation] = useState<string>();
  const [turns, setTurns] = useState<Turn[]>([]);
  const [question, setQuestion] = useState("");
  const [busy, setBusy] = useState(false);
  const [loading, setLoading] = useState(true);
  const [researcher, setResearcher] = useState(false);
  const [error, setError] = useState("");
  const [search, setSearch] = useState("");
  const [drawer, setDrawer] = useState(false);
  const [settings, setSettings] = useState(false);
  const [connection, setConnection] = useState("");
  const [uploadOpen, setUploadOpen] = useState(false);
  const [initialFile, setInitialFile] = useState<File>();
  const [target, setTarget] = useState<Target>();
  const [pending, setPending] = useState<string>();
  const [deleteDoc, setDeleteDoc] = useState<Doc>();
  const deletion = useRef<HTMLDialogElement>(null);
  const composer = useRef<HTMLTextAreaElement>(null);
  const end = useRef<HTMLDivElement>(null);

  async function refresh() {
    try {
      const [documents, findings, history, role] = await Promise.all([
        request<Doc[]>(session, "/documents"), request<Finding[]>(session, "/alerts"),
        request<Conversation[]>(session, "/conversations"), request<{researcher: boolean}>(session, "/session")
      ]);
      setDocs(documents); setAlerts(findings); setConversations(history); setResearcher(role.researcher);
      const active = documents.filter(doc => doc.status === "indexed").map(doc => doc.id);
      setSelected(previous => previous.length ? previous.filter(id => active.includes(id)) : active.length === 1 ? active : []);
    } catch (failure) { setError(failure instanceof Error ? failure.message : "Could not load the workspace."); }
    finally { setLoading(false); }
  }

  useEffect(() => { void refresh(); }, [session]);
  useEffect(() => { if (turns.length && view === "assistant") end.current?.scrollIntoView({behavior: "smooth", block: "end"}); }, [turns, view]);
  useEffect(() => { if (deleteDoc) deletion.current?.showModal(); }, [deleteDoc]);

  function navigate(next: View) { setView(next); setDrawer(false); setSettings(false); setError(""); }
  function newChat() {
    if (busy) return;
    setConversation(undefined); setTurns([]); setQuestion(""); navigate("assistant");
  }
  async function openConversation(id: string) {
    if (busy) return;
    setBusy(true); navigate("assistant");
    try { setTurns(await request<Turn[]>(session, `/conversations/${id}`)); setConversation(id); }
    catch (failure) { setError(failure instanceof Error ? failure.message : "Could not load the conversation."); }
    finally { setBusy(false); }
  }
  async function ask(event: FormEvent) {
    event.preventDefault();
    const text = question.trim();
    if (!text || !selected.length || busy) return;
    setError(""); setBusy(true); setQuestion(""); setTurns(previous => [...previous, {question: text}]);
    try {
      const answer = await request<Answer>(session, "/ask", {method: "POST",
        body: JSON.stringify({question: text, document_ids: selected, conversation_id: conversation})});
      setTurns(previous => [...previous.slice(0, -1), {question: text, result: answer}]);
      setConversation(answer.conversation_id);
      request<Conversation[]>(session, "/conversations").then(setConversations).catch(() => undefined);
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "Could not answer the question.");
      setTurns(previous => previous.slice(0, -1)); setQuestion(text);
    } finally { setBusy(false); }
  }
  async function changeDocument(doc: Doc, action: "delete" | "reindex" | "download") {
    setPending(doc.id); setError("");
    try {
      if (action === "download") {
        const data = await original(session, doc.id);
        downloadBlob(new Blob([data], {type: doc.media_type}), doc.name);
      } else {
        await request(session, `/documents/${doc.id}${action === "reindex" ? "/reindex" : ""}`, {method: action === "delete" ? "DELETE" : "POST"});
        if (action === "delete") {setDeleteDoc(undefined); setTurns([]); setConversation(undefined);}
        await refresh();
      }
    } catch (failure) {setError(failure instanceof Error ? failure.message : "Document operation failed.");}
    finally {setPending(undefined);}
  }
  function openUpload(file?: File) { setInitialFile(file); setUploadOpen(true); }
  function completion(doc: Doc) {
    setUploadOpen(false);
    setSelected(previous => [...new Set([...previous, doc.id])].slice(0, 20));
    void refresh();
  }
  const active = docs.filter(doc => doc.status === "indexed");
  const results = docs.filter(doc => `${doc.name} ${doc.category}`.toLowerCase().includes(search.toLowerCase()));
  const visible = alerts.filter(item => selected.includes(item.document_id));

  return <div className="workspace">
    {drawer && <button type="button" className="sidebar-backdrop" onClick={() => setDrawer(false)} aria-label="Close navigation"/>}
    <aside className={`sidebar ${drawer ? "open" : ""}`}>
      <Brand/>
      <Button className="new-chat" onClick={newChat} disabled={busy}><Plus size={18}/> New conversation</Button>
      <nav aria-label="Workspace">
        {NAV.map(({id, label, icon: Icon}) => <button key={id} type="button" className={view === id ? "nav-active" : ""}
          aria-current={view === id ? "page" : undefined} onClick={() => navigate(id)}><Icon size={18}/>{label}{id === "alerts" && alerts.length > 0 && <span className="nav-count">{alerts.length}</span>}</button>)}
      </nav>
      <div className="history"><h2>Recent conversations</h2>
        {conversations.length === 0 && <p>No conversations yet</p>}
        {conversations.map(chat => <button type="button" key={chat.id} title={chat.title} aria-current={conversation === chat.id ? "true" : undefined} onClick={() => openConversation(chat.id)} disabled={busy}>{chat.title}</button>)}
      </div>
      <div className="account">
        {settings && <div className="account-menu">
          {researcher && <button type="button" onClick={() => navigate("research")}><FlaskConical size={16}/> Research laboratory</button>}
          <button type="button" onClick={async () => {
            setConnection("Checking connections…");
            try { const result = await request<{model: string}>(session, "/connection"); setConnection(`Connected to Groq (${result.model}) and Supabase.`); }
            catch (failure) { setConnection(failure instanceof Error ? failure.message : "Connection check failed."); }
          }}><RefreshCw size={16}/> Check connections</button>
          {connection && <p role="status">{connection}</p>}
          <button type="button" onClick={() => void supabase?.auth.signOut()}><LogOut size={16}/> Sign out</button>
        </div>}
        <button type="button" className="profile" aria-expanded={settings} onClick={() => setSettings(!settings)}><span className="avatar">{(session.user.email || "L")[0].toUpperCase()}</span><span>{session.user.email || "My workspace"}</span><Settings size={16}/></button>
      </div>
    </aside>
    <main className="main">
      <header className="topbar">
        <div className="row"><Button variant="ghost" className="menu-toggle" aria-label="Open navigation" onClick={() => setDrawer(!drawer)}><Menu size={22}/></Button><span className="top-title">{NAV.find(x => x.id === view)?.label || "Research laboratory"}</span></div>
        <Button variant="ghost" className="top-alert" onClick={() => navigate("alerts")} aria-label="Open security alerts"><Bell size={18}/>{alerts.length > 0 && <span/>}</Button>
      </header>
      {error && <div className="global-error"><p className="notice error" role="alert"><CircleAlert size={17}/>{error} <button type="button" className="text-link" onClick={() => setError("")}>Dismiss</button></p></div>}
      {loading && <div className="loading-bar" role="status" aria-label="Loading workspace"/>}
      {view === "assistant" && <div className="assistant">
        <div className={`chat-scroll ${turns.length ? "" : "empty-chat"}`}>
          {turns.length === 0 && <motion.section className="hero" initial={{opacity: 0, y: 10}} animate={{opacity: 1, y: 0}} transition={{duration: 0.25}}>
            <Illustration/><h1>Your Secure AI Legal Research Assistant</h1><p>Ask questions about your legal documents and trace each supported answer to its source.</p>
            <div className="suggestions">{SUGGESTIONS.map((text, index) => <button key={text} type="button" onClick={() => {setQuestion(text); composer.current?.focus();}}>
              {index % 2 === 0 ? <BookOpen size={17}/> : <Landmark size={17}/>}<span>{text}</span><ArrowRight size={14}/></button>)}</div>
          </motion.section>}
          {turns.map((turn, i) => <div className="turn" key={turn.id || i}><p className="question">{turn.question}</p>
            {turn.result ? <AnswerCard answer={turn.result} open={setTarget}/> : <div role="status" className="thinking"><span/> Retrieving evidence and checking citations…</div>}</div>)}
          <div ref={end}/>
        </div>
        <div className="composer-wrap">
          <div className="composer-meta">
            <details className="select-docs"><summary><FolderClosed size={16}/>{selected.length ? `${selected.length} selected` : "Select documents"}<ChevronDown size={14}/></summary>
              <div className="select-menu">{active.length ? active.map(doc => <label key={doc.id} className="check">
                <input type="checkbox" checked={selected.includes(doc.id)} disabled={selected.length >= 20 && !selected.includes(doc.id)}
                  onChange={event => setSelected(previous => event.target.checked ? [...previous, doc.id] : previous.filter(id => id !== doc.id))}/><span>{doc.name}</span></label>) : <p>Upload a document to start.</p>}</div>
            </details>
            {selected.length > 0 && <span className={visible.length ? "scan-warn" : "scan-ok"}>{visible.length ? <ShieldAlert size={15}/> : <ShieldCheck size={15}/>}
              {visible.length ? `${visible.length} flagged ${visible.length === 1 ? "span" : "spans"}` : "No flagged spans"}</span>}
          </div>
          <form className="composer" onSubmit={ask}>
            <Button variant="ghost" aria-label="Attach document" title="Attach document" onClick={() => openUpload()}><Paperclip size={20}/></Button>
            <textarea ref={composer} value={question} onChange={event => setQuestion(event.target.value)} maxLength={2000} rows={1} aria-label="Legal research question" placeholder="Ask a question about your documents…"
              onKeyDown={event => {if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {event.preventDefault(); event.currentTarget.form?.requestSubmit();}}}/>
            <Button type="submit" aria-label="Send question" disabled={!question.trim() || !selected.length || busy}><ArrowUp size={20}/></Button>
          </form>
          <p className="legal-note">LegalShield AI is a research aid. Review the sources and exercise professional legal judgment.</p>
        </div>
      </div>}
      {view === "documents" && <section className="content-page">
        <div className="page-heading"><div><h1>My documents</h1><p>Manage source material and review its scan results.</p></div><Button onClick={() => openUpload()}><UploadCloud size={17}/> Upload document</Button></div>
        <label className="search"><Search size={18}/><input placeholder="Search documents…" aria-label="Search documents" value={search} onChange={event => setSearch(event.target.value)}/></label>
        {!results.length && <div className="empty-panel" onDragOver={event => event.preventDefault()} onDrop={event => {event.preventDefault(); if (event.dataTransfer.files.length) openUpload(event.dataTransfer.files[0]);}}><FolderClosed size={30}/>
          <h2>{search ? "No matching documents" : "No documents yet"}</h2><p>{search ? "Try a different name or category." : "Drop a PDF, DOCX or TXT file here, or choose one to upload."}</p>
          {!search && <Button variant="outline" onClick={() => openUpload()}>Choose a file</Button>}</div>}
        {results.map(doc => <article className="doc-card" key={doc.id}>
          <div className="head-label"><span className="file-icon"><FileText size={20}/></span><div><h2>{doc.name}</h2><p>{doc.category} · {doc.page_count} {doc.media_type === "application/pdf" ? (doc.page_count === 1 ? "page" : "pages") : "extracted view"}{doc.synthetic ? " · Synthetic" : doc.trusted ? " · Trusted reference" : ""}</p></div></div>
          <span className={doc.finding_count ? "scan-warn" : "scan-ok"}>{doc.status !== "indexed" ? doc.status : doc.finding_count ? `${doc.finding_count} flagged` : "Scan completed"}</span>
          <div className="doc-actions">
            <Button variant="outline" disabled={doc.status !== "indexed"} onClick={() => setTarget({documentId: doc.id, page: 1})}>Preview</Button>
            <Button variant="ghost" disabled={pending === doc.id || doc.status !== "indexed"} aria-label="Download original" onClick={() => changeDocument(doc, "download")}><ArrowDownToLine size={17}/></Button>
            <Button variant="ghost" disabled={pending === doc.id || doc.status !== "indexed"} title="Reindex" aria-label="Reindex document" onClick={() => changeDocument(doc, "reindex")}><RefreshCw size={16} className={pending === doc.id ? "spin" : ""}/></Button>
            <Button variant="ghost" disabled={pending === doc.id} onClick={() => setDeleteDoc(doc)}>Delete</Button>
          </div>
        </article>)}
      </section>}
      {view === "alerts" && <section className="content-page"><div className="page-heading"><div><h1>Security alerts</h1><p>Inspect flagged passages in their original context.</p></div></div>
        {!alerts.length && <div className="empty-panel"><ShieldCheck size={31}/><h2>No flagged passages</h2><p>Findings from indexed documents will appear here.</p></div>}
        {alerts.map(item => <FindingCard item={item} key={item.id} onOpen={() => setTarget({documentId: item.document_id, page: item.page, findingId: item.id})}/>)}
      </section>}
      {view === "research" && researcher && <Research session={session} docs={docs}/>}
    </main>
    {uploadOpen && <UploadDialog session={session} researcher={researcher} initial={initialFile} close={() => setUploadOpen(false)} complete={completion}/>}
    {target && <Viewer key={`${target.documentId}-${target.page}-${target.findingId || target.citation?.start_offset || ""}`} session={session} target={target} onClose={() => setTarget(undefined)}/>}
    {deleteDoc && <dialog ref={deletion} className="small-dialog" onCancel={event => {if (pending) event.preventDefault(); else setDeleteDoc(undefined);}} aria-labelledby="delete-heading">
      <div className="form-stack"><h2 id="delete-heading">Delete document?</h2><p>Delete {deleteDoc.name}, its stored index, and conversation answers based on it.</p>
        {error && <p role="alert" className="notice error">{error}</p>}
        <div className="row"><Button variant="outline" disabled={Boolean(pending)} onClick={() => setDeleteDoc(undefined)}>Cancel</Button><Button variant="danger" disabled={Boolean(pending)} onClick={() => changeDocument(deleteDoc, "delete")}>{pending ? "Deleting…" : "Delete document"}</Button></div></div>
    </dialog>}
  </div>;
}

export default function Home() {
  const [session, setSession] = useState<Session | null>();
  useEffect(() => {
    if (!supabase) {setSession(null); return;}
    const {data: {subscription}} = supabase.auth.onAuthStateChange((_event, current) => setSession(current));
    return () => subscription.unsubscribe();
  }, []);
  if (session === undefined) return <main className="boot" role="status"><Brand/><p>Opening your workspace…</p></main>;
  return session ? <Workspace key={session.user.id} session={session}/> : <SignIn/>;
}
