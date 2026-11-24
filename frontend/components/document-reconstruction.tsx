"use client";

import { useState, type FormEvent } from "react";
import type { Session } from "@supabase/supabase-js";
import { request, type Doc } from "@/lib/api";
import { Button } from "./ui/button";
import { MetricBars, MetricCards, unavailable } from "./metrics";

type ImageMetrics = {mse: number; mae: number; psnr: number | null; ssim: number; cer: number | null; wer: number | null};
type Output = {original: string; input: string | null; input_metrics: ImageMetrics | null; controlled: boolean; source_kind: string;
  results: {model: "ae" | "vae"; image: string; inference_ms: number; parameters: number; metrics: ImageMetrics | null}[]};
const name = {ae: "Denoising autoencoder (DAE)", vae: "Variational autoencoder"};
const imageUrl = (base64: string) => "data:image/png;base64," + base64;
const post = (value: unknown): RequestInit => ({method: "POST", body: JSON.stringify(value)});
const metricNames = [["MSE", "mse"], ["MAE", "mae"], ["PSNR (dB)", "psnr"], ["SSIM", "ssim"], ["CER", "cer"], ["WER", "wer"]] as const;

function ImagePanel({label, image, downloadable = false}: {label: string; image: string; downloadable?: boolean}) {
  return <figure className="reconstruction-image">
    <figcaption>{label}</figcaption>
    <img src={imageUrl(image)} alt={label}/>
    {downloadable && <a className="text-link" href={imageUrl(image)} download={label.toLowerCase().replaceAll(" ", "-") + ".png"}>Download image</a>}
  </figure>;
}

function ReconstructionImages({result}: {result: Output}) {
  const [zoom, setZoom] = useState(100);
  return <>
    <label>Image zoom · {zoom}%<input type="range" min={100} max={250} step={25} value={zoom} onChange={event => setZoom(Number(event.target.value))}/></label>
    <div className="reconstruction-scroll"><div className="reconstruction-grid" style={{width: zoom + "%"}}>
      <ImagePanel label={result.source_kind} image={result.original}/>
      {result.input && <ImagePanel label="Controlled degraded input" image={result.input}/>}
      {result.results.map(item => <ImagePanel key={item.model} label={name[item.model] + " reconstruction"} image={item.image} downloadable/>)}
    </div></div>
  </>;
}

export function DocumentRestoration({session, documents}: {session: Session; documents: Doc[]}) {
  const [file, setFile] = useState<File>();
  const [page, setPage] = useState(1);
  const [result, setResult] = useState<Output & {page_count: number}>();
  const [pageCount, setPageCount] = useState<number>();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [benchmarkId, setBenchmarkId] = useState("");
  const indexed = documents.filter(document => document.status === "indexed");
  const benchmarkDocument = indexed.find(document => document.id === benchmarkId);
  const isPdf = file?.type === "application/pdf" || /\.pdf$/i.test(file?.name ?? "");

  async function restore(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!file) return;
    if (file.size > 20 * 1024 * 1024) {
      setError("Choose a file of up to 20 MiB.");
      return;
    }
    const params = new URLSearchParams({page: String(isPdf ? page : 1), model: String(new FormData(event.currentTarget).get("model"))});
    setBusy(true);
    setError("");
    setResult(undefined);
    try {
      const output = await request<Output & {page_count: number}>(session, "/reconstruction/upload?" + params, {
        method: "POST", body: file, headers: {"Content-Type": file.type || "application/octet-stream"},
      });
      setResult(output);
      setPageCount(output.page_count);
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "Restoration failed.");
    } finally {
      setBusy(false);
    }
  }

  return <section className="content-page analytics-page analytics-stack">
    <div className="page-heading"><div><h1>Document Restoration</h1><p>Restore an already blurry scan and compare the result with the uploaded page.</p></div></div>
    <form className="card form-stack analytics-block" onSubmit={restore}>
      <label>Upload a PDF, PNG or JPEG<input type="file" accept=".pdf,.png,.jpg,.jpeg" required disabled={busy}
        onChange={event => {
          setFile(event.target.files?.[0]); setPage(1); setPageCount(undefined); setResult(undefined); setError("");
        }}/></label>
      <div className="form-grid">
        {isPdf && <label>PDF page<input type="number" min={1} max={Math.min(300, pageCount ?? 300)} value={page}
          required disabled={busy} onChange={event => setPage(Number(event.target.value))}/></label>}
        <label>Restoration model<select name="model" disabled={busy}>
          <option value="ae">Denoising autoencoder (default)</option><option value="both">Compare DAE and VAE</option><option value="vae">Variational autoencoder</option>
        </select></label>
      </div>
      <Button type="submit" disabled={!file || busy}>{busy ? "Restoring page…" : "Restore page"}</Button>
      {error && <p className="notice error" role="alert">{error}</p>}
    </form>
    {result && <section className="card analytics-block">
      <h2>Before and after</h2>
      <p className="muted">The decoder produces a grayscale reconstruction. Compare letters, numbers, and dates with the uploaded source before relying on them.</p>
      <ReconstructionImages result={{...result, source_kind: "Before · " + result.source_kind}}/>
      <MetricCards items={result.results.map(item => ({label: name[item.model] + " processing time", value: item.inference_ms + " ms"}))}/>
      <p className="muted">MSE, PSNR and SSIM · Not Evaluated. These metrics require a clean reference image.</p>
    </section>}
    <details className="card analytics-detail">
      <summary>Controlled benchmark and VAE explorer</summary>
      {indexed.length ? <>
        <label>Indexed document<select value={benchmarkDocument?.id ?? ""} onChange={event => setBenchmarkId(event.target.value)}>
          <option value="">Select a document</option>
          {indexed.map(document => <option key={document.id} value={document.id}>{document.name}</option>)}
        </select></label>
        {benchmarkDocument && <DocumentReconstruction key={benchmarkDocument.id} document={benchmarkDocument} session={session}/>}
      </> : <p className="muted">Upload and index a document to run a controlled benchmark.</p>}
    </details>
  </section>;
}

function DocumentReconstruction({document, session}: {document: Doc; session: Session}) {
  const [status, setStatus] = useState<{available: boolean; message?: string}>();
  const [visible, setVisible] = useState(false);
  const [result, setResult] = useState<Output>();
  const [sample, setSample] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  if (document.status !== "indexed") return null;

  async function open() {
    if (visible) {setVisible(false); return;}
    setVisible(true); setError("");
    try {setStatus(await request(session, "/reconstruction/status"));}
    catch (failure) {setError((failure as Error).message);}
  }

  async function run(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    setBusy(true); setError("");
    try {
      setResult(await request(session, "/documents/" + document.id + "/reconstruction", post({
        page: Number(form.get("page")), model: form.get("model"), degradation: form.get("degradation"),
        controlled: form.has("controlled"), severity: Number(form.get("severity")),
      })));
    } catch (failure) {setError((failure as Error).message);}
    finally {setBusy(false);}
  }

  async function generateSample(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    setBusy(true); setError("");
    try {
      const generated = await request<{image: string}>(session, "/reconstruction/latent", post({
        seed: Number(form.get("seed")), coordinates: [Number(form.get("latent1")), Number(form.get("latent2"))],
      }));
      setSample(generated.image);
    } catch (failure) {setError((failure as Error).message);}
    finally {setBusy(false);}
  }

  return <div className="operation-metrics">
    <Button variant="outline" aria-expanded={visible} onClick={() => void open()}>Open controlled reconstruction</Button>
    {visible && <section className="card analytics-block" aria-label="Controlled reconstruction">
      <h3>Document reconstruction</h3>
      <p className="muted">AI-assisted images are separate from the original. Review reconstructed text before relying on it.</p>
      {error && <p className="notice error" role="alert">{error}</p>}
      {status?.available === false && <>
        <p role="status">{status.message}</p>
        <MetricCards items={[{label: "MSE", value: null}, {label: "PSNR", value: null}, {label: "SSIM", value: null}]}/>
      </>}
      {!status && !error && <p role="status">Checking model availability…</p>}
      {status?.available && <>
        <form className="form-stack" onSubmit={run}>
          <div className="form-grid">
            <label>Page<input name="page" type="number" defaultValue={1} min={1} max={document.page_count} required/></label>
            <label>Model<select name="model"><option value="ae">Denoising autoencoder (default)</option><option value="both">Compare DAE and VAE</option><option value="vae">Variational autoencoder</option></select></label>
            <label>Controlled degradation<select name="degradation"><option value="gaussian">Gaussian noise</option><option value="blur">Blur</option><option value="contrast">Contrast loss</option><option value="speckle">Speckle</option><option value="compression">JPEG compression</option></select></label>
            <label>Degradation strength<input name="severity" type="range" min={0} max={1} step={.05} defaultValue={.5}/></label>
          </div>
          <label className="check"><input type="checkbox" name="controlled" defaultChecked/>Create a degraded copy and measure against the source render</label>
          <Button type="submit" disabled={busy}>Reconstruct page</Button>
        </form>
        {result && <>
          <ReconstructionImages result={result}/>
          {!result.controlled && <p className="muted">Reference-based image quality · {unavailable}. Real input has no clean reference.</p>}
          <div className="metrics-grid">
            {metricNames.map(([label, key]) => {
              const rows = [...(result.input_metrics ? [{label: "Degraded input", value: result.input_metrics[key]}] : []),
                ...result.results.map(item => ({label: name[item.model], value: item.metrics?.[key] ?? null}))];
              return <MetricBars key={label} title={label} rows={rows} maximum={Math.max(.001, ...rows.map(row => row.value ?? 0))}
                format={value => value.toFixed(key === "psnr" ? 2 : 4)}/>;
            })}
            <MetricBars title="Reconstruction latency" rows={result.results.map(item => ({label: name[item.model], value: item.inference_ms}))}
              maximum={Math.max(1, ...result.results.map(item => item.inference_ms))} format={value => value + " ms"}/>
            <MetricBars title="Model parameters" rows={result.results.map(item => ({label: name[item.model], value: item.parameters}))}
              maximum={Math.max(1, ...result.results.map(item => item.parameters))} format={value => value.toLocaleString()}/>
          </div>
        </>}
        <details className="analytics-detail"><summary>VAE latent-space explorer</summary>
          <form className="form-stack" onSubmit={generateSample}>
            <p className="muted">Generate a new synthetic document-layout patch.</p>
            <div className="form-grid">
              <label>Seed<input type="number" name="seed" min={0} max={1000000000} defaultValue={2026} required/></label>
              <label>First coordinate<input type="range" name="latent1" min={-4} max={4} step={.25} defaultValue={0}/></label>
              <label>Second coordinate<input type="range" name="latent2" min={-4} max={4} step={.25} defaultValue={0}/></label>
            </div>
            <Button type="submit" disabled={busy}>Generate sample</Button>
          </form>
          {sample && <div className="latent-sample"><ImagePanel label="New synthetic VAE sample" image={sample} downloadable/></div>}
        </details>
      </>}
    </section>}
  </div>;
}
