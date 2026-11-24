import type { ReactNode } from "react";

export const unavailable = "Not Evaluated";
export const formatPercent = (value: number) => `${(value * 100).toFixed(2)}%`;

export function MetricCards({items}: {items: {label: string; value: ReactNode}[]}) {
  return <dl className="metric-cards">{items.map(({label, value}) => <div className="metric-card" key={label}>
    <dt>{label}</dt><dd>{value ?? unavailable}</dd>
  </div>)}</dl>;
}

export function MetricBars({title, rows, maximum = 1, format = formatPercent}: {
  title: string; rows: {label: string; value: number | null; highlight?: boolean}[];
  maximum?: number; format?: (value: number) => string;
}) {
  return <section className="metric-chart" aria-label={title}>
    <h3>{title}</h3>
    {rows.map(({label, value, highlight}) => <div className={`metric-row${highlight ? " metric-highlight" : ""}`} key={label}>
      <div><span>{label}</span><strong>{value == null ? unavailable : format(value)}</strong></div>
      {value != null && <meter min={0} max={maximum} value={Math.max(0, Math.min(maximum, value))} aria-label={label}/>}
    </div>)}
  </section>;
}
