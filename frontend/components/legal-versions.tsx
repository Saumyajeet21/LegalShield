"use client";

import { useState } from "react";
import type { Session } from "@supabase/supabase-js";
import { request, type Doc } from "@/lib/api";

type LegalVersion = {
  document_id: string;
  family_id: string;
  version_label: string;
  effective_from: string;
  effective_to: string | null;
  authority_status: "verified" | "unverified" | "revoked";
};

type VersionChanges = {added_passages: number; removed_passages: number; newly_observed_findings: unknown[]};

export function LegalVersions({document, session, researcher}: {document: Doc; session: Session; researcher: boolean}) {
  const [versions, setVersions] = useState<LegalVersion[]>();
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [changes, setChanges] = useState<VersionChanges>();
  const [comparison, setComparison] = useState("");
  const path = "/documents/" + document.id;
  async function loadVersions() {
    setError("");
    try {
      setVersions(await request<LegalVersion[]>(session, path + "/legal-versions"));
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "Version history unavailable.");
    }
  }

  return <details onToggle={event => {if (event.currentTarget.open && !versions) void loadVersions();}}>
    <summary>Legal versions</summary>
    {error && <p role="alert">{error}</p>}
    {versions?.length ? <>
      <p className="muted">Family {versions[0].family_id}</p>
      <ul>{versions.map(version => <li key={version.document_id}>
        {version.version_label} · {version.authority_status} · {version.effective_from}
        {version.effective_to ? " to " + version.effective_to + " (exclusive)" : ""}
      </li>)}</ul>
    </> : versions && <p className="muted">No version registered.</p>}
    {versions && versions.length > 1 && <div className="form-stack">
      <label>Compare this document with
        <select value={comparison} onChange={event => {setComparison(event.target.value); setChanges(undefined);}}>
          <option value="">Select a version</option>
          {versions.filter(version => version.document_id !== document.id).map(version =>
            <option key={version.document_id} value={version.document_id}>{version.version_label}</option>)}
        </select>
      </label>
      <button type="button" disabled={!comparison || busy} onClick={async () => {
        setBusy(true);
        setError("");
        try {
          setChanges(await request<VersionChanges>(session, path + "/legal-change?before=" + comparison));
        } catch (failure) {
          setError(failure instanceof Error ? failure.message : "Version comparison unavailable.");
        } finally {
          setBusy(false);
        }
      }}>Compare versions</button>
      {changes && <p className="muted">{changes.added_passages} passages added · {changes.removed_passages} passages removed
        {" · "}{changes.newly_observed_findings.length} newly observed findings</p>}
    </div>}
    {versions?.length === 0 && researcher && <form className="form-stack" onSubmit={async event => {
      event.preventDefault();
      const form = new FormData(event.currentTarget);
      setBusy(true);
      setError("");
      try {
        await request<LegalVersion>(session, path + "/legal-version", {
          method: "POST", body: JSON.stringify({
            family_id: form.get("family_id") || crypto.randomUUID(),
            version_label: form.get("version_label"),
            authority_status: form.get("authority_status"),
            source_reference: form.get("source_reference"),
            effective_from: form.get("effective_from"),
            effective_to: form.get("effective_to") || null,
            precedence: Number(form.get("precedence")),
          }),
        });
        await loadVersions();
      } catch (failure) {
        setError(failure instanceof Error ? failure.message : "Registration unavailable.");
      } finally {
        setBusy(false);
      }
    }}>
      <label>Family ID, or blank for a new family<input name="family_id"
        pattern="[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}"/></label>
      <label>Version label<input name="version_label" required maxLength={80}/></label>
      <label>Source reference<input name="source_reference" required maxLength={500}/></label>
      <label>Authority review<select name="authority_status">
        <option value="unverified">Unverified</option>
        {(document.trusted || document.synthetic) && <option value="verified">Verified by reviewer</option>}
      </select></label>
      <label>Effective from<input name="effective_from" type="date" required/></label>
      <label>Effective to, exclusive<input name="effective_to" type="date"/></label>
      <label>Priority<input name="precedence" type="number" min={0} max={10000} defaultValue={0} required/></label>
      <button type="submit" disabled={busy}>Register reviewed metadata</button>
    </form>}
  </details>;
}
