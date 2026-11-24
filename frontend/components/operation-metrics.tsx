"use client";

import { useState } from "react";
import type { Session } from "@supabase/supabase-js";
import { request, type Answer, type Doc } from "@/lib/api";
import { MetricBars, MetricCards, formatPercent } from "./metrics";

type Scan = {spans: number; affected_pages: number; isolated: number; quarantined: number; retention: number | null; confidence: number | null};

export function ScanResults({document, session}: {document: Doc; session: Session}) {
  const [scan, setScan] = useState<Scan>();
  const [error, setError] = useState("");
  if (!document.finding_count || document.status !== "indexed") return null;
  return <details className="operation-metrics" onToggle={event => {
    if (event.currentTarget.open) request<Scan>(session, "/documents/" + document.id + "/scan-metrics")
      .then(value => {setScan(value); setError("");}).catch(failure => setError(failure.message));
  }}>
    <summary>Detection results · {document.finding_count} suspicious spans</summary>
    {error ? <p role="alert">{error}</p> : scan ? <>
      <MetricCards items={[{label: "Suspicious spans", value: scan.spans}, {label: "Affected pages", value: scan.affected_pages},
        {label: "Evidence retention", value: scan.retention == null ? null : formatPercent(scan.retention)},
        {label: "Detector confidence", value: scan.confidence == null ? null : formatPercent(scan.confidence)}]}/>
      <div className="metrics-grid">
        <MetricBars title="Action by span" rows={[{label: "Isolated", value: scan.isolated}, {label: "Quarantined", value: scan.quarantined}]}
          maximum={Math.max(1, scan.spans)} format={String}/>
        <MetricBars title="Candidate source text coverage" rows={[{label: "Retained in retrieval index", value: scan.retention},
          {label: "Excluded from retrieval index", value: scan.retention == null ? null : 1 - scan.retention}]}/>
      </div>
    </> : <p role="status">Loading scan results…</p>}
  </details>;
}

export function AnswerMetrics({answer}: {answer: Answer}) {
  const record = answer.verification;
  const checks = record?.attempts.at(-1)?.claims;
  const contradicted = checks?.filter(claim => ["contradicted", "authority_contradiction"].includes(claim.status)).length;
  const supported = checks?.filter(claim => claim.status === "supported").length;
  const unsupported = checks && supported != null && contradicted != null ? checks.length - supported - contradicted : null;
  const comparison = answer.comparison ? Object.entries(answer.comparison).map(([key, row]) =>
    ({label: key === "baseline" ? "Baseline" : "Independent verification", ...row})) : [];
  return <section className="operation-metrics" aria-label="Answer performance">
    <MetricCards items={[
      {label: "Published claims", value: answer.claims.length},
      {label: "Response latency", value: answer.response_latency_ms == null ? null : answer.response_latency_ms.toLocaleString() + " ms"},
      {label: "Candidate uncertainty", value: record?.semantic_entropy == null ? null : record.semantic_entropy.toFixed(2) + " bits"},
    ]}/>
    <details><summary>Claim verification metrics</summary>
      <MetricBars title="Final evaluated draft" maximum={Math.max(1, checks?.length ?? 0)} format={String}
        rows={[{label: "Supported", value: supported ?? null}, {label: "Other rejected or unresolved", value: unsupported},
          {label: "Contradicted", value: contradicted ?? null}]}/>
      {comparison.length ? <div className="metrics-grid">
        <MetricBars title="Paired run · Published claims" rows={comparison.map(row => ({label: row.label, value: row.claims}))}
          maximum={Math.max(1, ...comparison.map(row => row.claims))} format={String}/>
        <MetricBars title="Paired run · Generation and checking time" rows={comparison.map(row => ({label: row.label, value: row.latency_ms}))}
          maximum={Math.max(1, ...comparison.map(row => row.latency_ms))} format={value => value.toLocaleString() + " ms"}/>
      </div> : <p className="muted">Paired baseline and independent verification comparison · Not Evaluated</p>}
    </details>
  </section>;
}
