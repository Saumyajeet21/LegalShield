"use client";

import { useState, type FormEvent } from "react";
import type { Session } from "@supabase/supabase-js";
import { artifact, request, type Claim, type Doc, type Target } from "@/lib/api";
import { Button } from "./ui/button";
import { downloadBlob } from "./viewer";
import { MetricBars, MetricCards, unavailable } from "./metrics";

const draftTypes = [["nda", "Non-disclosure agreement"], ["supplier", "Supplier agreement"], ["employment", "Employment agreement"],
  ["notice", "Legal notice"], ["amendment", "Contract amendment"], ["custom", "Custom document"], ["clause", "Alternative clause"]];
const requirementFields = ["Parties", "Dates", "Obligations", "Payment terms", "Termination conditions"];
type Props = {session: Session; docs: Doc[]; open: (target: Target) => void};
type Check = {label: string; value?: string; matched: boolean};
type Clause = {title: string; text: string};
type Draft = {title: string; clauses: Clause[]; source_citations: Claim["citations"]; metrics: {
  requirements: Check[]; headings: Check[]; clauses: number; words: number; generation_ms: number; tokens: number | null;
  temperature: number; top_p: number; verification: {status: string; symbolic?: string}[] | null;
}};
type DraftResult = {drafts: Draft[]; semantic_similarity: number[][] | null};
const jsonExport = (value: unknown, name: string) => downloadBlob(new Blob([JSON.stringify(value, null, 2)], {type: "application/json"}), name + ".json");
const post = (data: unknown): RequestInit => ({method: "POST", body: JSON.stringify(data)});
const rate = (checks: Check[]) => checks.length ? checks.filter(check => check.matched).length / checks.length : null;
const duration = (value: number) => value.toLocaleString() + " ms";
const max = (values: (number | null)[]) => Math.max(1, ...values.map(value => value ?? 0));

function Sources({docs}: {docs: Doc[]}) {
  return <fieldset className="workflow-sources">
    <legend>Source documents (optional)</legend>
    {!docs.some(doc => doc.status === "indexed") && <p className="muted">Upload a document in My Documents to use source evidence.</p>}
    {docs.filter(doc => doc.status === "indexed").map(doc => <label className="check" key={doc.id}>
      <input type="checkbox" name="sources" value={doc.id}/>{doc.name}
    </label>)}
  </fieldset>;
}

function Citations({citations, open}: {citations: Claim["citations"]; open: Props["open"]}) {
  return <ul className="workflow-citations">{citations.map((source, index) => <li key={index}>
    <button type="button" className="text-link" onClick={() => open({documentId: source.document_id, page: source.page, citation: source})}>
      {source.name} · page {source.page}
    </button>
  </li>)}</ul>;
}

export function GenerationStudio({session, docs, open}: Props) {
  const [result, setResult] = useState<DraftResult>();
  const [edited, setEdited] = useState<Record<string, string>>({});
  const [config, setConfig] = useState<Record<string, unknown>>();
  const [draftCount, setDraftCount] = useState(1);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  async function generate(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    const requirements = Object.fromEntries(requirementFields.map(label => [label, String(form.get(label) ?? "").trim()]).filter(([, value]) => value));
    const settings = {kind: form.get("kind"), requirements, count: draftCount, instructions: form.get("instructions"),
      document_ids: form.getAll("sources"), max_tokens: Number(form.get("tokens")), temperature: Number(form.get("temperature")),
      top_p: Number(form.get("top_p")), comparison: form.get("compare") && draftCount > 1 ? {
        temperature: Number(form.get("compare_temp")), top_p: Number(form.get("compare_top_p"))} : null};
    setBusy(true); setError("");
    try {
      const response = await request<DraftResult>(session, "/generation/drafts", post(settings));
      setResult(response); setEdited({}); setConfig(settings);
    } catch (failure) {setError((failure as Error).message);}
    finally {setBusy(false);}
  }

  async function revise(draft: number, index: number, clause: Clause) {
    const id = draft + "-" + index;
    setBusy(true); setError("");
    try {
      const response = await request<DraftResult>(session, "/generation/drafts",
        post({...config, count: 1, comparison: null, revise_clause: clause.title, previous_text: edited[id] ?? clause.text}));
      setEdited(old => ({...old, [id]: response.drafts[0].clauses[0].text}));
    } catch (failure) {setError((failure as Error).message);}
    finally {setBusy(false);}
  }

  async function save(draft: Draft, index: number, kind: "pdf" | "docx") {
    const text = draft.clauses.map((clause, position) => clause.title + "\n" + (edited[index + "-" + position] ?? clause.text)).join("\n\n");
    try {
      downloadBlob(await artifact(session, "/generation/export/" + kind, post({title: draft.title, text})), "laven-draft." + kind);
    } catch (failure) {setError((failure as Error).message);}
  }

  const series = (metric: (draft: Draft["metrics"]) => number | null) => (result?.drafts ?? []).map(({metrics}, index) =>
    ({label: "Draft " + (index + 1) + " · " + metrics.temperature + " temperature · " + metrics.top_p + " top-p", value: metric(metrics)}));
  const scores = [
    ["Requirement values matched", series(metrics => rate(metrics.requirements))],
    ["Required headings present", series(metrics => rate(metrics.headings))],
    ["Verified source support", series(metrics => metrics.verification?.length ?
      metrics.verification.filter(row => row.status === "supported").length / metrics.verification.length : null)],
    ["Explicit logical checks satisfied", series(metrics => {
      const evaluated = metrics.verification?.filter(row => row.symbolic && row.symbolic !== "not_applicable");
      return evaluated?.length ? evaluated.filter(row => row.symbolic === "supported").length / evaluated.length : null;
    })],
  ] as const;
  const comparisons = [
    ["Supported source claims", series(metrics => metrics.verification?.filter(row => row.status === "supported").length ?? null), String],
    ["Unsupported or unresolved source claims", series(metrics => metrics.verification?.filter(row => !["supported", "contradicted", "authority_contradiction"].includes(row.status)).length ?? null), String],
    ["Contradictions", series(metrics => metrics.verification?.filter(row => ["contradicted", "authority_contradiction"].includes(row.status)).length ?? null), String],
    ["Generation latency", series(metrics => metrics.generation_ms), duration],
    ["Reported generation tokens", series(metrics => metrics.tokens), String],
  ] as const;
  return <section className="content-page analytics-page analytics-stack">
    <div className="page-heading"><div><h1>Legal Generation Studio</h1><p>Generate, edit and compare AI drafts for legal review.</p></div></div>
    <form className="card form-stack analytics-block" onSubmit={generate}>
      <div className="form-grid">
        <label>Document type<select name="kind">{draftTypes.map(([value, name]) => <option key={value} value={value}>{name}</option>)}</select></label>
        <label>Drafts<select value={draftCount} onChange={event => setDraftCount(Number(event.target.value))}>{[1, 2, 3].map(value => <option key={value}>{value}</option>)}</select></label>
        {requirementFields.map(label => <label key={label}>{label}<input name={label} maxLength={300}/></label>)}
      </div>
      <label>Additional instructions<textarea name="instructions" rows={3} maxLength={2500}/></label>
      <Sources docs={docs}/>
      <details className="analytics-detail"><summary>Generation settings</summary>
        <div className="form-grid">
          <label>Temperature<input name="temperature" type="number" min={0} max={1.5} step={.05} defaultValue={.35} required/></label>
          <label>Top-p<input name="top_p" type="number" min={.05} max={1} step={.05} defaultValue={1} required/></label>
          <label>Max output tokens<input name="tokens" type="number" min={512} max={3600} defaultValue={2200} required/></label>
        </div>
        {draftCount > 1 && <div className="form-grid">
          <label className="check"><input type="checkbox" name="compare"/>Use alternate settings for later drafts</label>
          <label>Alternate temperature<input name="compare_temp" type="number" min={0} max={1.5} step={.05} defaultValue={.65}/></label>
          <label>Alternate top-p<input name="compare_top_p" type="number" min={.05} max={1} step={.05} defaultValue={.9}/></label>
        </div>}
      </details>
      <p className="muted">{draftCount} Qwen drafting {draftCount === 1 ? "request" : "requests"}. Source verification can make additional calls.</p>
      <Button type="submit" disabled={busy}>{busy ? "Generating draft…" : "Generate drafts"}</Button>
    </form>
    {error && <p className="notice error" role="alert">{error}</p>}
    {result && <>
      <section className="card analytics-block"><h2>Draft comparison</h2>
        <p className="muted">Scores count explicit value and heading matches on the generated drafts. Review applicability, completeness and enforceability separately.</p>
        <div className="metrics-grid">{scores.map(([title, rows]) => <MetricBars key={title} title={title} rows={rows}/>)}
          {comparisons.map(([title, rows, format]) => <MetricBars key={title} title={title} rows={rows} format={format} maximum={max(rows.map(row => row.value))}/>)}</div>
        <h3>Draft semantic similarity</h3>
        {result.semantic_similarity ? <div className="table-wrap"><table className="heatmap"><thead><tr><th>MiniLM cosine</th>
          {result.drafts.map((_, i) => <th key={i}>Draft {i + 1}</th>)}</tr></thead>
          <tbody>{result.semantic_similarity.map((row, i) => <tr key={i}><th scope="row">Draft {i + 1}</th>
            {row.map((value, j) => <td key={j} style={{backgroundColor: "rgba(8,122,112," + Math.max(0, value) * .2 + ")"}}>{value.toFixed(3)}</td>)}</tr>)}</tbody></table></div> : <p>{unavailable}</p>}
        <Button variant="outline" onClick={() => jsonExport({generated: result, edited: Object.keys(edited).length > 0}, "laven-draft-evaluation")}>Download evaluation JSON</Button>
      </section>
      {result.drafts.map((draft, index) => <article className="card analytics-block" key={index}>
        <h2>Draft {index + 1} · {draft.title}</h2>
        <p className="notice">AI-generated draft. Review all terms and sources before use.</p>
        <MetricCards items={[{label: "Clauses", value: draft.metrics.clauses}, {label: "Words", value: draft.metrics.words},
          {label: "Generation time", value: duration(draft.metrics.generation_ms)}, {label: "Reported tokens", value: draft.metrics.tokens}]}/>
        <details><summary>Requirements and headings</summary><ul>{[...draft.metrics.requirements, ...draft.metrics.headings].map((check, i) =>
          <li key={i}>{check.label}{check.value ? " · " + check.value : ""} · {check.matched ? "Matched" : "Missing"}</li>)}</ul></details>
        <Citations citations={draft.source_citations} open={open}/>
        {draft.clauses.map((clause, position) => <div className="form-stack" key={position}>
          <label>{clause.title}<textarea rows={4} maxLength={4000} value={edited[index + "-" + position] ?? clause.text}
            onChange={event => setEdited(old => ({...old, [index + "-" + position]: event.target.value}))}/></label>
          <Button variant="outline" disabled={busy} onClick={() => void revise(index, position, clause)}>Regenerate clause</Button>
        </div>)}
        {Object.keys(edited).some(key => key.startsWith(index + "-")) && <p className="muted">Evaluation reflects the original generated draft. This edited version can be exported.</p>}
        <div className="row"><Button variant="outline" onClick={() => void save(draft, index, "pdf")}>Download PDF</Button>
          <Button variant="outline" onClick={() => void save(draft, index, "docx")}>Download DOCX</Button></div>
      </article>)}
    </>}
  </section>;
}
