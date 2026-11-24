"use client";

import { useState } from "react";
import type { Session } from "@supabase/supabase-js";
import { artifact, type Answer, type Target } from "@/lib/api";
import { downloadBlob } from "./viewer";

export function VerificationDetails({answer, conversationId, session, open}: {
  answer: Answer; conversationId?: string; session: Session; open: (target: Target) => void;
}) {
  const [error, setError] = useState("");
  const record = answer.verification;
  if (!record) return null;
  const questionId = record.question_id;
  const conversation = answer.conversation_id ?? conversationId;
  function exportJSON() {
    downloadBlob(new Blob([JSON.stringify(record, null, 2)], {type: "application/json"}), "laven-verification.json");
  }
  async function exportPDF() {
    if (!conversation) return;
    try {
      const pdf = await artifact(session, "/conversations/" + conversation + "/verification/" + questionId + ".pdf");
      downloadBlob(pdf, "laven-verification.pdf");
      setError("");
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "PDF export failed.");
    }
  }
  return <details className="card">
    <summary>Verification · {record.status.replaceAll("_", " ")}</summary>
    {record.as_of && <p className="muted">Applicable as of {record.as_of}</p>}
    {record.semantic_entropy != null && <p className="muted">Candidate uncertainty {record.semantic_entropy.toFixed(2)} bits</p>}
    {record.authority_issues?.map(issue => <p key={issue}>{issue}</p>)}
    {record.attempts.at(-1)?.claims?.map((claim, index) => <section key={index}>
      <p>{claim.text} · {claim.status.replaceAll("_", " ")}</p>
      {claim.unresolved_conditions.map(condition => <p className="muted" key={condition}>{condition}</p>)}
      <ul>
        {claim.citations.map((source, i) => <li key={i}><button type="button" className="text-link"
          onClick={() => open({documentId: source.document_id, page: source.page, citation: source})}>
          {source.name} · page {source.page}{source.legal_version ? " · " + source.legal_version : ""}
        </button></li>)}
      </ul>
    </section>)}
    {record.evidence_graph?.edges.length ? <details>
      <summary>Evidence links</summary>
      <ul>{record.evidence_graph.edges.map((edge, i) => {
        const source = record.evidence_graph?.nodes.find(node => node.id === edge.to)?.source;
        return <li key={i}>{edge.from} {edge.relation} {source ? <button type="button" className="text-link"
          onClick={() => open({documentId: source.document_id, page: source.page, citation: source})}>
          {source.name} · page {source.page}
        </button> : edge.to}</li>;
      })}</ul>
    </details> : null}
    <div className="chips">
      <button className="text-link" type="button" onClick={exportJSON}>Download verification JSON</button>
      {conversation && <button className="text-link" type="button" onClick={exportPDF}>Download PDF</button>}
    </div>
    {error && <p role="alert">{error}</p>}
  </details>;
}
