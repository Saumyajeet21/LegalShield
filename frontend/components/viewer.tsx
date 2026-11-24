"use client";
import dynamic from "next/dynamic";
import { useEffect, useRef, useState, type ReactNode } from "react";
import type { Session } from "@supabase/supabase-js";
import { ChevronLeft, ChevronRight, Download, FileText, X } from "lucide-react";
import { original, request, type Preview, type Target } from "@/lib/api";
import { Button } from "@/components/ui/button";

const PdfCanvas = dynamic(() => import("./pdf-canvas"), {ssr: false});

export function downloadBlob(data: Blob, name: string) {
  const url = URL.createObjectURL(data);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = name;
  anchor.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function Viewer({session, target, onClose}: {session: Session; target: Target; onClose: () => void}) {
  const dialog = useRef<HTMLDialogElement>(null);
  const [page, setPage] = useState(target.page);
  const [data, setData] = useState<Preview>();
  const [loading, setLoading] = useState(true);
  const [active, setActive] = useState(target.findingId);
  const [error, setError] = useState("");

  useEffect(() => {
    dialog.current?.showModal();
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setError("");
    request<Preview>(session, `/documents/${target.documentId}/preview?page=${page}`, {signal: controller.signal})
      .then(value => { if (!controller.signal.aborted) setData(value); })
      .catch(failure => { if (!controller.signal.aborted) setError(failure.message); })
      .finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, [session, target.documentId, page]);

  async function download() {
    if (!data) return;
    try {
      const bytes = await original(session, target.documentId);
      downloadBlob(new Blob([bytes], {type: data.document.media_type}), data.document.name);
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "Unable to download the document.");
    }
  }

  const hits = data?.page.page === page ? data.findings : [];
  const selected = hits.find(item => item.id === active);
  const citation = target.citation?.page === page ? target.citation : undefined;
  const text = data?.page.page === page ? Array.from(data.page.text) : [];
  const marks = hits.map(item => ({start: item.start_offset, end: item.end_offset, key: item.id, className: "text-highlight"}));
  if (citation && !marks.some(mark => mark.start < citation.end_offset && mark.end > citation.start_offset)) {
    marks.push({start: citation.start_offset, end: citation.end_offset, key: "citation", className: "citation-highlight"});
  }
  let cursor = 0;
  const snippets: ReactNode[] = [];
  for (const mark of marks.sort((a, b) => a.start - b.start)) {
    snippets.push(text.slice(cursor, mark.start).join(""), <mark key={mark.key} className={mark.className}>{text.slice(mark.start, mark.end).join("")}</mark>);
    cursor = mark.end;
  }
  snippets.push(text.slice(cursor).join(""));

  return <dialog ref={dialog} className="viewer-dialog" onCancel={onClose} aria-labelledby="viewer-title">
    <header className="viewer-head">
      <div className="head-label"><span className="file-icon"><FileText size={18}/></span><div><h2 id="viewer-title">{data?.document.name || "Document preview"}</h2><span>Page {page}{data && ` of ${data.document.page_count}`}</span></div></div>
      <div className="row">
        <Button variant="ghost" aria-label="Download original" title="Download original" disabled={!data} onClick={download}><Download size={18}/></Button>
        <Button variant="ghost" aria-label="Close preview" onClick={onClose}><X size={20}/></Button>
      </div>
    </header>
    {error && <p className="notice error" role="alert">{error}</p>}
    <div className="viewer-tools">
      <Button variant="outline" disabled={page <= 1 || loading} aria-label="Previous page" onClick={() => {setPage(page - 1); setActive(undefined);}}><ChevronLeft size={17}/></Button>
      <span>Page {page}</span>
      <Button variant="outline" disabled={!data || page >= data.document.page_count || loading} aria-label="Next page" onClick={() => {setPage(page + 1); setActive(undefined);}}><ChevronRight size={17}/></Button>
    </div>
    <div className="viewer-body">
      {data?.document.media_type === "application/pdf" && <PdfCanvas session={session} documentId={target.documentId} page={page} findings={hits} activeId={active} onSelect={item => setActive(item.id)}/>}
      <aside className="viewer-details">
        {selected && <div className="notice warning"><strong>{selected.category.replaceAll("_", " ")}</strong><p>{selected.reason}</p><blockquote>{selected.quote}</blockquote></div>}
        {citation && <div className="notice verified"><strong>Cited passage</strong><blockquote>{citation.quote}</blockquote></div>}
        {loading ? <p role="status">Loading extracted text…</p> : <><h3>Extracted text</h3>
          {hits.some(item => !item.boxes.length) && <p className="muted">Exact page coordinates are unavailable for some findings. Their text is highlighted below.</p>}
          <pre className="extracted">{text.length ? snippets : "No selectable text on this page."}</pre></>}
      </aside>
    </div>
  </dialog>;
}
