"use client";

import { useEffect, useState } from "react";
import type { Session } from "@supabase/supabase-js";
import { artifact, request } from "@/lib/api";
import { downloadBlob } from "./viewer";
import { MetricBars, MetricCards, formatPercent, unavailable } from "./metrics";

type Figure = {id: number; points: {label: string; value: number; x?: number | null}[]};
type Summary = {version: string; figures: Figure[]};
const measures = [["Accuracy", "accuracy"], ["Macro precision", "macro_precision"], ["Macro recall", "macro_recall"], ["Macro F1-score", "macro_f1"]] as const;
type Model = {id: string; name: string; test_examples: number} & Record<typeof measures[number][1], number>;
type Comparison = {models: Model[]; charts: string[]};
type ReconstructionSummary = {test_count: number; test_source: string; primary_model?: string; metrics: Record<string, Record<string, number | null>>;
  charts: {name: string; title: string}[]};
const plottedModels = [["deberta-legal", "Fine-tuned legal DeBERTa"], ["bilstm", "BiLSTM"], ["tfidf", "TF-IDF"], ["general", "Frozen DeBERTa"]];

function PublishedPlot({name, title, session, reconstruction = false}: {name: string; title: string; session: Session; reconstruction?: boolean}) {
  const [url, setUrl] = useState<string | null>();
  useEffect(() => {
    let active = true;
    let image: string | undefined;
    artifact(session, (reconstruction ? "/research/reconstruction/charts/" : "/research/charts/") + name).then(blob => {
      if (active) {image = URL.createObjectURL(blob); setUrl(image);}
    }).catch(() => {if (active) setUrl(null);});
    return () => {active = false; if (image) URL.revokeObjectURL(image);};
  }, [session, name, reconstruction]);
  return <figure className="published-plot">
    <figcaption>{title}</figcaption>
    {url ? <a href={url} download={name + ".png"} title="Download chart as PNG"><img src={url} alt={title}/></a> :
      <p className="muted">{url === null ? unavailable : "Loading chart…"}</p>}
  </figure>;
}

const groups = [
  ["Dataset Explorer", 1, ["Source distribution", "Class distribution", "Split distribution", "Text length", "Public and synthetic datasets"]],
  ["Model Training", 6, ["BiLSTM accuracy", "BiLSTM loss", "DeBERTa accuracy", "DeBERTa loss", "Learning rate", "Validation macro F1", "Training time"]],
  ["Classification Evaluation", 13, ["Model accuracy", "Precision, recall and F1", "Per-class F1", "Confusion matrices", "ROC", "Precision–recall"]],
  ["Generative AI Evaluation", 19, ["Answer correctness", "Claim grounding", "Unsupported claims", "Citation precision", "Before and after regeneration", "Regeneration success and failure"]],
  ["Prompt Injection Research", 25, ["Controlled answer deviation", "Injection detection F1", "False positives and false negatives", "Evidence retention"]],
  ["Authority and Version Analysis", 29, ["Authorized amendment accuracy", "Unauthorized change resistance", "Change sensitivity and invariance", "Version robustness"]],
  ["Counterfactual Experiments", 33, ["Counterfactual category performance"]],
  ["Semantic Entropy", 34, ["Semantic entropy distribution", "Entropy by correctness"]],
  ["Uncertainty Calibration", 36, ["Reliability", "Risk and coverage", "Uncertainty AUROC"]],
  ["Symbolic Reasoning", 39, ["Logical verification accuracy", "Rule and exception extraction"]],
  ["Baseline Comparisons", 41, ["System comparisons"]],
  ["Ablation Studies", 42, ["Ablation comparisons"]],
  ["Failure Analysis", 43, ["Paired confidence intervals", "Latency", "Tokens and API cost"]],
] as const;

function Curves({points, name}: {points: Figure["points"]; name: string}) {
  const colors = ["#176388", "#b55435", "#24734e", "#784e91", "#94721c", "#455b72"];
  const grouped = new Map<string, Figure["points"]>();
  points.forEach(point => grouped.set(point.label, [...(grouped.get(point.label) ?? []), point]));
  return <div>
    <svg viewBox="0 0 420 245" width="100%" role="img" aria-label={name}>
      <path d="M40 15V212H408" stroke="currentColor" fill="none"/>
      {Array.from(grouped.values()).map((series, index) => <polyline key={index} fill="none"
        stroke={colors[index % colors.length]} strokeWidth="2"
        strokeDasharray={index >= colors.length ? "5 3" : undefined}
        points={series.map(point => [40 + 365 * (point.x ?? 0), 212 - 193 * point.value].join(",")).join(" ")}/>)}
      <text x="38" y="233" fontSize="12" fill="currentColor">0</text>
      <text x="400" y="233" fontSize="12" fill="currentColor">1</text>
      <text x="22" y="20" fontSize="12" fill="currentColor">1</text>
      <text x="170" y="233" fontSize="12" fill="currentColor">{({"ROC": "False-positive rate", "Reliability": "Confidence", "Risk and coverage": "Coverage"} as Record<string, string>)[name] ?? "Recall"}</text>
    </svg>
    <ul>{Array.from(grouped.keys()).map((label, index) =>
      <li key={label}><span style={{color: colors[index % colors.length]}}>●</span> {label}</li>)}</ul>
  </div>;
}

export function ResearchArtifacts({session}: {session: Session}) {
  const [summary, setSummary] = useState<Summary | null>();
  const [comparison, setComparison] = useState<Comparison>();
  const [reconstruction, setReconstruction] = useState<ReconstructionSummary | null>();
  const [reconstructionOpen, setReconstructionOpen] = useState(false);
  const [open, setOpen] = useState<string[]>([]);
  const [error, setError] = useState("");
  useEffect(() => {
    const controller = new AbortController();
    request<Summary | null>(session, "/research/artifacts", {signal: controller.signal})
      .then(setSummary)
      .catch(failure => {if (!controller.signal.aborted) setError(failure.message);});
    request<Comparison>(session, "/research/comparison", {signal: controller.signal})
      .then(setComparison)
      .catch(failure => {if (!controller.signal.aborted) setError(failure.message);});
    request<ReconstructionSummary | null>(session, "/research/reconstruction", {signal: controller.signal})
      .then(setReconstruction)
      .catch(() => {if (!controller.signal.aborted) setReconstruction(null);});
    return () => controller.abort();
  }, [session]);

  function exportSummary() {
    if (summary) downloadBlob(new Blob([JSON.stringify(summary, null, 2)], {type: "application/json"}), "laven-research.json");
  }

  const models = comparison?.models ?? [];
  const legal = models.find(model => model.id === "deberta-legal");
  const plot = (name: string, title: string) => comparison?.charts.includes(name) ? <PublishedPlot key={name} name={name} title={title} session={session}/> : null;
  const details = [
    ["Confusion matrices", plottedModels.map(([id, label]) => [id + "-confusion_matrix", label + " confusion matrix"])],
    ["Training, inference and per-class scores", [["comparison-class-f1", "Per-class F1"], ["comparison-inference", "Inference latency"], ["comparison-training", "Training duration"], ["bilstm-accuracy", "BiLSTM accuracy"], ["bilstm-loss", "BiLSTM loss"]]],
    ["Calibration and uncertainty", plottedModels.flatMap(([id, label]) => [[id + "-test_reliability", label + " reliability"], [id + "-test_risk_coverage", label + " risk and coverage"]])],
    ["ROC and precision–recall", plottedModels.flatMap(([id, label]) => [[id + "-roc", label + " ROC"], [id + "-precision_recall", label + " precision–recall"]])],
  ] as const;
  return <div className="analytics-stack">
    {error && <p role="alert">{error}</p>}
    <section className="card analytics-block">
      <h2>ContractNLI model performance</h2>
      <p className="muted">{legal ? "Fine-tuned legal DeBERTa · " + legal.test_examples.toLocaleString() + " official held-out document–hypothesis pairs." : comparison ? unavailable : "Loading measured results…"}</p>
      {legal && <MetricCards items={measures.map(([label, key]) => ({label, value: formatPercent(legal[key])}))}/>}
      {models.length > 0 && <div className="metrics-grid">{measures.map(([title, key]) => <MetricBars key={key} title={title}
        rows={models.map(model => ({label: model.name, value: model[key], highlight: model.id === "deberta-legal"}))}/>)}</div>}
    </section>
    <section className="card analytics-block"><h2>Model comparison</h2>
      <div className="analytics-chart-grid">
        {plot("comparison-accuracy", "ContractNLI accuracy comparison")}
        {plot("comparison-prf", "ContractNLI macro precision, recall and F1")}
        {plot("comparison-overview", "Four official test metrics by model")}
        {plot("comparison-tradeoff", "Macro F1 and measured inference latency")}
      </div>
      {comparison && !comparison.charts.includes("comparison-accuracy") && <p>{unavailable}</p>}
    </section>
    <section className="card analytics-block"><h2>Fine-tuned DeBERTa training</h2>
      <div className="analytics-chart-grid">
        {plot("deberta-legal-accuracy", "Training and validation accuracy")}
        {plot("deberta-legal-loss", "Training and validation loss")}
      </div>
      {comparison && !comparison.charts.includes("deberta-legal-loss") && <p>{unavailable}</p>}
    </section>
    {details.map(([title, charts]) => <details className="card analytics-detail" key={title}
      onToggle={event => {const shown = event.currentTarget.open; setOpen(previous => shown ? [...new Set([...previous, title])] : previous.filter(item => item !== title));}}>
      <summary>{title}</summary>
      {open.includes(title) && <div className="analytics-chart-grid">{charts.map(([name, label]) => plot(name, label))}</div>}
      {comparison && !charts.some(([name]) => comparison.charts.includes(name)) && <p>{unavailable}</p>}
    </details>)}
    <details className="card analytics-detail" onToggle={event => setReconstructionOpen(event.currentTarget.open)}>
      <summary>Document reconstruction benchmarks</summary>
      {reconstruction ? <>
        <p className="muted">{reconstruction.test_count.toLocaleString()} held-out {reconstruction.test_source}.</p>
        <div className="table-wrap"><table><thead><tr><th>Model</th><th>MSE</th><th>PSNR</th><th>SSIM</th><th>CER</th><th>WER</th></tr></thead>
          <tbody>{Object.entries(reconstruction.metrics).map(([model, row]) => <tr key={model}>
            <th scope="row">{{input: "Degraded input", ae: reconstruction.primary_model ?? "Autoencoder", vae: "VAE"}[model] ?? model}</th>
            {["mse", "psnr", "ssim", "cer", "wer"].map(key => <td key={key}>{row[key]?.toFixed(4) ?? unavailable}</td>)}
          </tr>)}</tbody></table></div>
        {reconstructionOpen && <div className="analytics-chart-grid">{reconstruction.charts.map(({name, title}) =>
          <PublishedPlot name={name} title={title} session={session} key={name} reconstruction/>)}</div>}
      </> : <p className="muted">{reconstruction === undefined ? "Checking measured results…" : unavailable}</p>}
    </details>
    {groups.filter(([, first]) => first !== 13).map(([title, first, figures]) => <details className="card analytics-detail" key={title}>
      <summary>{title}</summary>
      <div className="analytics-chart-grid">
      {first === 1 && plot("dataset-classes", "ContractNLI class distribution")}
      {figures.map((name, index) => {
        const id = first + index;
        const data = summary?.figures.find(figure => figure.id === id);
        const maximum = data ? Math.max(...data.points.map(point => Math.abs(point.value)), .0001) : 1;
        return <section className="table-wrap" key={name}>
          <h3>{name}</h3>
          {data?.points[0]?.x != null ? <Curves points={data.points} name={name}/> : data ? <table><tbody>{data.points.map(({label, value}, i) => <tr key={i}>
            <th scope="row">{label}</th><td><meter aria-label={label} min={-maximum} max={maximum} value={value}/></td>
            <td>{new Intl.NumberFormat(undefined, {maximumFractionDigits: 5}).format(value)}</td>
          </tr>)}</tbody></table> : <p className="muted">{unavailable}</p>}
        </section>;
      })}
      </div>
    </details>)}
    <details><summary>Dataset and paper references</summary>
      <ul>
        <li><a href="https://stanfordnlp.github.io/contract-nli/" target="_blank" rel="noreferrer">ContractNLI</a></li>
        <li><a href="https://github.com/The-Atticus-Project/cuad" target="_blank" rel="noreferrer">CUAD</a></li>
        <li><a href="https://hazyresearch.stanford.edu/legalbench/" target="_blank" rel="noreferrer">LegalBench</a></li>
        <li><a href="https://arxiv.org/abs/2408.10343" target="_blank" rel="noreferrer">LegalBench-RAG</a></li>
      </ul>
    </details>
    {summary && <button className="text-link" onClick={exportSummary} type="button">Export measured research summary JSON</button>}
  </div>;
}
