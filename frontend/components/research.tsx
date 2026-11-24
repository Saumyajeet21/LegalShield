"use client";
import { useEffect, useState } from "react";
import type { Session } from "@supabase/supabase-js";
import { Download } from "lucide-react";
import { request } from "@/lib/api";
import { Button } from "./ui/button";
import { downloadBlob } from "./viewer";
import { ResearchArtifacts } from "./research-artifacts";

type Metric = { precision: number | null; recall: number | null; character_f1: number | null; exact_span_f1: number | null; false_positive_clause_rate: number | null };
type Run = { id: string; created_at: string; report: { question: string; statuses: string[]; metrics: Record<string, Metric>; latency_ms: number;
  raw_marker_observed: boolean; protected_marker_observed: boolean; legitimate_character_retention: number | null;
  citation_supported_claim_fraction: number | null; answer_phrase_match: boolean | null; raw_output: unknown; protected_output: unknown; grounded_claims: unknown; } };
const percent = (value: number | null) => value === null ? "—" : `${(value * 100).toFixed(1)}%`;

export function Research({session}: {session: Session}) {
  const [runs, setRuns] = useState<Run[]>([]);
  const [error, setError] = useState("");
  useEffect(() => {
    request<Run[]>(session, "/research").then(setRuns).catch(failure => setError(failure.message));
  }, [session]);

  return <section className="content-page analytics-page">
    <div className="page-heading"><div><h1>AI Performance &amp; Analytics</h1><p>Official held-out results, training curves, and measured AI performance.</p></div>
      <Button variant="outline" disabled={!runs.length} onClick={() => downloadBlob(new Blob([runs.map(run => JSON.stringify(run)).join("\n") + "\n"], {type: "application/x-ndjson"}), "legalshield-experiments.ndjson")}><Download size={16}/> Export runs</Button></div>
    <ResearchArtifacts session={session}/>
    {error && <p className="notice error" role="alert">{error}</p>}
    <details className="card analytics-detail"><summary>Saved detector and paired answer runs ({runs.length})</summary>
    {!runs.length && <p className="muted">Not Evaluated</p>}
    {runs.map(({id, report, created_at}) => <section className="card run-card" key={id}>
      <h2>{report.question}</h2><p className="muted">{new Date(created_at).toLocaleString()} · {report.latency_ms} ms</p>
      <div className="chips">{report.statuses.map(status => <span className="tag" key={status}>{status}</span>)}</div>
      <div className="table-wrap"><table><thead><tr><th>Detector</th><th>Precision</th><th>Recall</th><th>Character F1</th><th>Exact-span F1</th><th>Clause false positives</th></tr></thead>
      <tbody>{Object.entries(report.metrics).map(([name, row]) => <tr key={name}><td>{name.replaceAll("_", " ")}</td><td>{percent(row.precision)}</td><td>{percent(row.recall)}</td><td>{percent(row.character_f1)}</td><td>{percent(row.exact_span_f1)}</td><td>{percent(row.false_positive_clause_rate)}</td></tr>)}</tbody></table></div>
      <p className="muted">Marker observed in raw and protected outputs: {report.raw_marker_observed ? "yes" : "no"} / {report.protected_marker_observed ? "yes" : "no"}. Evidence retention: {percent(report.legitimate_character_retention)}.</p>
      <details><summary>Compare generated outputs</summary><pre>{JSON.stringify({raw: report.raw_output, protected: report.protected_output, grounded: report.grounded_claims}, null, 2)}</pre></details>
    </section>)}
    </details>
  </section>;
}
