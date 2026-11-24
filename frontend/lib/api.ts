import { createClient } from "@supabase/supabase-js";
import type { Session } from "@supabase/supabase-js";

const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
const key = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;
const recoveryKey = "legalshield:password-recovery";
const callback = typeof window === "undefined" ? null : new URLSearchParams(window.location.hash.slice(1));
let recoveryLink = callback?.get("type") === "recovery" || callback?.get("error_code") === "otp_expired";
let recoveryPending = Boolean(callback?.get("type") === "recovery" && callback.get("access_token") && callback.get("refresh_token"));
let recoveryUser: string | null = null;
export const supabase = url && key ? createClient(url, key) : null;
if (typeof window !== "undefined") {
  try {
    if (recoveryLink) window.sessionStorage.removeItem(recoveryKey);
    else recoveryUser = window.sessionStorage.getItem(recoveryKey);
  } catch { /* Storage can be disabled. */ }
  // Capture recovery before React subscribes. Supabase may clear the callback hash on initialization.
  supabase?.auth.onAuthStateChange((event, session) => {
    if (event === "PASSWORD_RECOVERY" && session) {
      recoveryPending = false;
      recoveryUser = session.user.id;
      try { window.sessionStorage.setItem(recoveryKey, recoveryUser); } catch { /* Use in-memory state. */ }
    }
  });
}

export function passwordRecoveryMode(session: Session | null) {
  if (recoveryUser && recoveryUser === session?.user.id) return "verified";
  if (recoveryPending) return "pending";
  return recoveryLink || recoveryUser ? "unavailable" : null;
}

export function failPasswordRecovery() {
  recoveryPending = false;
}

export function clearPasswordRecovery() {
  recoveryUser = null;
  recoveryLink = false;
  recoveryPending = false;
  try { window.sessionStorage.removeItem(recoveryKey); } catch { /* Storage can be disabled. */ }
  const hash = new URLSearchParams(window.location.hash.slice(1));
  if (hash.get("type") === "recovery" || hash.get("error_code") === "otp_expired") {
    window.history.replaceState(window.history.state, "", window.location.pathname + window.location.search);
  }
}
const apiBase = (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000").replace(/\/$/, "");

export type Doc = {id: string; name: string; media_type: string; category: string; page_count: number; finding_count: number;
  status: "processing" | "indexed" | "deleting"; synthetic: boolean; trusted: boolean; created_at: string};
export type Finding = {id: string; document_id: string; name?: string; page: number; quote?: string; category: string;
  severity: string; reason: string; action: string; boxes: [number, number, number, number][]; start_offset: number; end_offset: number};
export type Citation = {document_id: string; name: string; page: number; clause: string | null; quote: string; start_offset: number; end_offset: number; legal_version?: string | null};
export type Claim = {text: string; citations: Citation[]; authority: null | {references: string[]; status: string; source?: {document_id: string; page: number; quote: string}}};
type VerificationRecord = {question_id: string; status: string; as_of: string | null; semantic_entropy: number | null;
  authority_issues: string[];
  evidence_graph?: {nodes: {id: string; kind: "claim" | "citation" | "rule"; source?: Citation}[];
    edges: {from: string; to: string; relation: string}[]};
  attempts: {claims?: {text: string; status: string; unresolved_conditions: string[]; citations: Citation[]}[]}[]};
export type Answer = {verification?: VerificationRecord; response_latency_ms?: number; claims: Claim[]; message: string | null; alerts: Finding[]; conversation_id?: string; question_flagged: boolean;
  comparison?: Record<"baseline" | "verified", {claims: number; latency_ms: number}>;
  retrieval: {id: number; document_id: string; page: number; similarity: number}[]};
export type Turn = {id?: number; question: string; result?: Answer};
export type Conversation = {id: string; title: string};
export type Preview = {document: Doc; page: {page: number; text: string}; findings: Finding[]};
export type Target = {documentId: string; page: number; findingId?: string; citation?: Citation};

async function token(session: Session) {
  const current = await supabase?.auth.getSession();
  // Prefer the live session from the client (it may hold a refreshed token).
  if (current?.data.session?.user.id === session.user.id) return current.data.session.access_token;
  // Fall back to the session passed in (e.g. right after sign-in before the
  // client's internal store has propagated the new session).
  if (current?.data.session == null && session.access_token) return session.access_token;
  throw new Error("Your session ended. Sign in again.");
}

function detail(value: unknown) {
  if (typeof value === "string") return value;
  if (Array.isArray(value)) return value.map(item => item?.msg).filter(Boolean).join(". ") || "Check the supplied fields.";
  return "The request could not be completed.";
}

async function responseError(response: Response) {
  const body = await response.json().catch(() => null);
  return new Error(detail(body?.detail) + (body ? "" : ` (HTTP ${response.status})`));
}

export async function request<T>(session: Session, path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(apiBase + path, { ...init, cache: "no-store",
    headers: { Authorization: `Bearer ${await token(session)}`,
      ...(init?.body ? { "Content-Type": "application/json" } : {}), ...init?.headers } });
  if (!response.ok) throw await responseError(response);
  return response.status === 204 ? undefined as T : response.json();
}

export async function original(session: Session, id: string, signal?: AbortSignal) {
  const response = await fetch(`${apiBase}/documents/${id}/original`, {
    signal, cache: "no-store", headers: { Authorization: `Bearer ${await token(session)}` }
  });
  if (!response.ok) throw await responseError(response);
  return response.arrayBuffer();
}

export async function upload(session: Session, data: FormData, progress: (amount: number) => void, signal?: AbortSignal) {
  const bearer = await token(session);
  return new Promise<Doc>((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    if (signal?.aborted) return reject(new Error("Upload canceled."));
    const cancel = () => xhr.abort();
    signal?.addEventListener("abort", cancel, { once: true });
    xhr.addEventListener("loadend", () => signal?.removeEventListener("abort", cancel));
    xhr.open("POST", apiBase + "/documents");
    xhr.setRequestHeader("Authorization", `Bearer ${bearer}`);
    xhr.responseType = "json";
    xhr.upload.addEventListener("progress", event => {
      if (event.lengthComputable && event.total > 0) progress(Math.min(99, Math.round(event.loaded / event.total * 100)));
    });
    xhr.upload.addEventListener("load", () => progress(100));
    xhr.onload = () => xhr.status === 201 ? resolve(xhr.response as Doc) : reject(new Error(detail(xhr.response?.detail)));
    xhr.onerror = () => reject(new Error("The backend could not be reached."));
    xhr.onabort = () => reject(new Error("Upload canceled."));
    xhr.send(data);
  });
}


export async function artifact(session: Session, path: string, init?: RequestInit) {
  const response = await fetch(apiBase + path, {...init, cache: "no-store",
    headers: {Authorization: "Bearer " + await token(session),
      ...(init?.body ? {"Content-Type": "application/json"} : {}), ...init?.headers}});
  if (!response.ok) throw await responseError(response);
  return response.blob();
}
