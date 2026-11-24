"use client";
import { useEffect, useState, type FormEvent } from "react";
import type { Session } from "@supabase/supabase-js";
import { Download, FlaskConical } from "lucide-react";
import { request, type Doc } from "@/lib/api";
import { Button } from "./ui/button";
import { downloadBlob } from "./viewer";

type Metric = { precision: number | null; recall: number | null; character_f1: number | null; exact_span_f1: number | null; false_positive_clause_rate: number | null };
type Run = { id: string; created_at: string; report: { question: string; statuses: string[]; metrics: Record<string, Metric>; latency_ms: number;
  raw_marker_observed: boolean; protected_marker_observed: boolean; legitimate_character_retention: number | null;
  citation_supported_claim_fraction: number | null; answer_phrase_match: boolean | null; raw_output: unknown; protected_output: unknown; grounded_claims: unknown; } };
const percent = (value: number | null) => value === null ? "—" : `${(value * 100).toFixed(1)}%`;

export function Research({session, docs}: {session: Session; docs: Doc[]}) {
  const [runs, setRuns] = useState<Run[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const synthetic = docs.filter(doc => doc.synthetic && doc.status === "indexed");
  useEffect(() => {
    request<Run[]>(session, "/research").then(setRuns).catch(failure => setError(failure.message));
  }, [session]);

  async function run(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const data = Object.fromEntries(new FormData(event.currentTarget));
    setBusy(true); setError("");
    try {
      const expected = JSON.parse(String(data.expected).trim() || "[]");
      if (!Array.isArray(expected)) throw new Error("Gold spans must be a JSON array.");
      const result = await request<Run>(session, "/research", {method: "POST", body: JSON.stringify({...data, expected})});
      setRuns(previous => [result, ...previous]);
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "The experiment failed.");
    } finally { setBusy(false); }
  }

  return <section className="content-page">
    <div className="page-heading"><div><h1>Research laboratory</h1><p>Compare detectors and paired RAG runs on synthetic legal documents.</p></div>
      <Button variant="outline" disabled={!runs.length} onClick={() => downloadBlob(new Blob([runs.map(run => JSON.stringify(run)).join("\n") + "\n"], {type: "application/x-ndjson"}), "legalshield-experiments.ndjson")}><Download size={16}/> Export runs</Button></div>
    {error && <p className="notice error" role="alert">{error}</p>}
    <form className="card form-stack" onSubmit={run}>
      <h2><FlaskConical size={20}/> New experiment</h2>
      {!synthetic.length && <p className="muted">Upload a document marked synthetic through My Documents to begin.</p>}
      <label>Synthetic document<select name="document_id" required><option value="">Select document</option>{synthetic.map(doc => <option value={doc.id} key={doc.id}>{doc.name}</option>)}</select></label>
      <label>Question<input name="question" required minLength={3} maxLength={2000}/></label>
      <div className="form-grid">
        <label>Synthetic attack marker<input name="attack_marker" minLength={8} maxLength={100} pattern="[A-Za-z0-9_\-]+" required/></label>
        <label>Dataset split<select name="split"><option value="test">Test</option><option value="development">Development</option></select></label>
      </div>
      <label>Gold spans <small>JSON array of page, start_offset and end_offset. Offsets are zero-based.</small><textarea name="expected" rows={3} placeholder='[{"page":1,"start_offset":26,"end_offset":68}]'/></label>
      <label>Expected answer phrase (optional)<input name="expected_answer_phrase" maxLength={300}/></label>
      <Button type="submit" disabled={busy || !synthetic.length}>{busy ? "Running paired experiment…" : "Run experiment"}</Button>
    </form>
    {runs.map(({id, report, created_at}) => <section className="card run-card" key={id}>
      <h2>{report.question}</h2><p className="muted">{new Date(created_at).toLocaleString()} · {report.latency_ms} ms</p>
      <div className="chips">{report.statuses.map(status => <span className="tag" key={status}>{status}</span>)}</div>
      <div className="table-wrap"><table><thead><tr><th>Detector</th><th>Precision</th><th>Recall</th><th>Character F1</th><th>Exact-span F1</th><th>Clause false positives</th></tr></thead>
      <tbody>{Object.entries(report.metrics).map(([name, row]) => <tr key={name}><td>{name.replaceAll("_", " ")}</td><td>{percent(row.precision)}</td><td>{percent(row.recall)}</td><td>{percent(row.character_f1)}</td><td>{percent(row.exact_span_f1)}</td><td>{percent(row.false_positive_clause_rate)}</td></tr>)}</tbody></table></div>
      <p className="muted">Marker observed in raw and protected outputs: {report.raw_marker_observed ? "yes" : "no"} / {report.protected_marker_observed ? "yes" : "no"}. Evidence retention: {percent(report.legitimate_character_retention)}.</p>
      <details><summary>Compare generated outputs</summary><pre>{JSON.stringify({raw: report.raw_output, protected: report.protected_output, grounded: report.grounded_claims}, null, 2)}</pre></details>
    </section>)}
  </section>;
}
