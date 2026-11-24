"use client";

import { useEffect, useRef, useState } from "react";
import type { PDFDocumentProxy } from "pdfjs-dist";
import type { Session } from "@supabase/supabase-js";
import { original, type Finding } from "@/lib/api";

export default function PdfCanvas({session, documentId, page, findings, activeId, onSelect}:
  {session: Session; documentId: string; page: number; findings: Finding[]; activeId?: string; onSelect: (finding: Finding) => void}) {
  const box = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const [pdf, setPdf] = useState<PDFDocumentProxy | null>(null);
  const [width, setWidth] = useState(0);
  const [aspect, setAspect] = useState(0);
  const [error, setError] = useState("");
  const [rendered, setRendered] = useState(0);
  const generation = useRef(Promise.resolve());

  useEffect(() => {
    if (!box.current) return;
    const observer = new ResizeObserver(([entry]) => setWidth(Math.floor(entry.contentRect.width)));
    observer.observe(box.current);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    let task: ReturnType<typeof import("pdfjs-dist")["getDocument"]> | undefined;
    setPdf(null);
    setAspect(0);
    setError("");
    setRendered(0);
    async function load() {
      try {
        const [pdfjs, data] = await Promise.all([import("pdfjs-dist"), original(session, documentId, controller.signal)]);
        if (controller.signal.aborted) return;
        pdfjs.GlobalWorkerOptions.workerSrc = "/pdf-worker.mjs";
        task = pdfjs.getDocument({data, enableXfa: false});
        const opened = await task.promise;
        if (!controller.signal.aborted) setPdf(opened);
      } catch (failure) {
        if (!controller.signal.aborted) setError(failure instanceof Error ? failure.message : "The PDF could not be opened.");
      }
    }
    void load();
    return () => {
      controller.abort();
      void task?.destroy().catch(() => undefined);
    };
  }, [documentId, session]);

  useEffect(() => {
    if (!pdf || !width) return;
    let canceled = false;
    let job: ReturnType<Awaited<ReturnType<PDFDocumentProxy["getPage"]>>["render"]> | undefined;
    async function render() {
      try {
        const source = await pdf!.getPage(page);
        if (canceled || !canvas.current) return;
        const normal = source.getViewport({scale: 1});
        const viewport = source.getViewport({scale: width / normal.width});
        const scale = Math.min(window.devicePixelRatio || 1, 2);
        canvas.current.width = Math.ceil(viewport.width * scale);
        canvas.current.height = Math.ceil(viewport.height * scale);
        setAspect(normal.width / normal.height);
        job = source.render({canvas: canvas.current, viewport, transform: scale === 1 ? undefined : [scale, 0, 0, scale, 0, 0]});
        await job.promise;
        if (!canceled) setRendered(page);
      } catch (failure) {
        if (!canceled) setError(failure instanceof Error ? failure.message : "The PDF page could not be rendered.");
      }
    }
    generation.current = generation.current.then(render);
    return () => {
      canceled = true;
      job?.cancel();
    };
  }, [pdf, page, width]);

  return <div ref={box} className="pdf-frame">
    {error ? <p className="notice error" role="alert">{error} The extracted text is available below.</p> : rendered !== page && <p className="pdf-loading" role="status">Rendering page {page}…</p>}
    <div className="pdf-sheet" style={{aspectRatio: aspect || undefined, visibility: !error && rendered === page ? "visible" : "hidden"}}>
      <canvas ref={canvas} aria-label={`Original PDF, page ${page}`}/>
      {rendered === page && findings.flatMap(item => item.boxes.map(([left, top, right, bottom], index) =>
        <button key={`${item.id}-${index}`} type="button" title={item.reason}
          aria-label={`${item.category.replaceAll("_", " ")}: ${item.quote}`}
          className={`pdf-highlight ${activeId === item.id ? "selected" : ""}`}
          style={{left: `${left * 100}%`, top: `${top * 100}%`, width: `${(right - left) * 100}%`, height: `${(bottom - top) * 100}%`}}
          onClick={() => onSelect(item)}/>))}
    </div>
  </div>;
}
