import { createClient } from "@supabase/supabase-js";
import type { Session } from "@supabase/supabase-js";

const url = process.env.NEXT_PUBLIC_SUPABASE_URL;
const key = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;
export const supabase = url && key ? createClient(url, key) : null;
const apiBase = (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000").replace(/\/$/, "");

export type Doc = {id: string; name: string; media_type: string; category: string; page_count: number; finding_count: number;
  status: "processing" | "indexed" | "deleting"; synthetic: boolean; trusted: boolean; created_at: string};
export type Finding = {id: string; document_id: string; name?: string; page: number; quote?: string; category: string;
  severity: string; reason: string; action: string; boxes: [number, number, number, number][]; start_offset: number; end_offset: number};
export type Citation = {document_id: string; name: string; page: number; clause: string | null; quote: string; start_offset: number; end_offset: number};
export type Claim = {text: string; citations: Citation[]; authority: null | {references: string[]; status: string; source?: {document_id: string; page: number; quote: string}}};
export type Answer = {claims: Claim[]; message: string | null; alerts: Finding[]; conversation_id?: string; question_flagged: boolean;
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
      if (event.lengthComputable) progress(Math.round(event.loaded / event.total * 100));
    });
    xhr.onload = () => xhr.status === 201 ? resolve(xhr.response as Doc) : reject(new Error(detail(xhr.response?.detail)));
    xhr.onerror = () => reject(new Error("The backend could not be reached."));
    xhr.onabort = () => reject(new Error("Upload canceled."));
    xhr.send(data);
  });
}
