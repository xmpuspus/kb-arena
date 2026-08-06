import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";

const root = resolve(import.meta.dirname, "..");
const sourcePath = resolve(root, "results/run_855aac4e/retriever_lab.json");
const outputPath = resolve(root, "tmp/benchmark-evidence.html");
const run = JSON.parse(readFileSync(sourcePath, "utf8"));
const corpus = "aws-compute";
const labels = {
  contextual_vector: "Contextual Vector",
  naive_vector: "Naive Vector",
  raptor: "RAPTOR",
  bm25: "BM25",
  hybrid: "Hybrid",
  pageindex: "PageIndex",
};
const rows = Object.entries(run.corpora[corpus])
  .filter(([name, metrics]) => name in labels && metrics.mean_recall_at_k > 0)
  .sort((a, b) => b[1].mean_recall_at_k - a[1].mean_recall_at_k);
const maximum = Math.max(...rows.map(([, metrics]) => metrics.mean_recall_at_k));
const questions = run.corpora[corpus].naive_vector.questions;

const rowMarkup = rows
  .map(([name, metrics], index) => {
    const width = (metrics.mean_recall_at_k / maximum) * 100;
    const color = ["#167d73", "#315ca8", "#7f5aa2", "#c45731", "#9a7221", "#55606f"][index];
    return `<tr>
      <th>${labels[name]}</th>
      <td class="bar-cell"><span class="bar" style="width:${width.toFixed(2)}%;background:${color}"></span></td>
      <td>${metrics.mean_recall_at_k.toFixed(3)}</td>
      <td>${metrics.mean_mrr.toFixed(3)}</td>
      <td>${metrics.mean_ndcg_at_k.toFixed(3)}</td>
    </tr>`;
  })
  .join("\n");

const html = `<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>KB Arena historical retrieval evidence</title>
<style>
  * { box-sizing: border-box; }
  html, body { margin: 0; width: 1200px; height: 675px; background: #f7f8fa; color: #1d2733; }
  body { font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; padding: 54px 64px; }
  header { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 34px; }
  .brand { color: #167d73; font-size: 18px; font-weight: 750; margin-bottom: 10px; }
  h1 { font-size: 36px; line-height: 1.15; margin: 0 0 12px; letter-spacing: 0; }
  .subtitle { color: #596676; font-size: 17px; }
  .run { border-left: 4px solid #c45731; padding: 8px 0 8px 16px; color: #465362; font-size: 15px; line-height: 1.6; }
  table { width: 100%; border-collapse: collapse; table-layout: fixed; background: #fff; border: 1px solid #d9dee5; }
  thead { background: #eef1f4; color: #465362; }
  th, td { border-bottom: 1px solid #e2e6eb; padding: 13px 15px; text-align: right; font-variant-numeric: tabular-nums; }
  thead th { font-size: 13px; text-transform: uppercase; font-weight: 700; }
  tbody th { text-align: left; font-size: 15px; width: 190px; }
  .bar-head, .bar-cell { width: 430px; text-align: left; }
  .bar { display: block; height: 15px; border-radius: 3px; }
  .note { margin-top: 20px; display: grid; grid-template-columns: 1fr 1fr; gap: 28px; color: #596676; font-size: 14px; line-height: 1.5; }
  .note strong { color: #1d2733; }
  .source { text-align: right; }
</style>
<body>
  <header>
    <div>
      <div class="brand">KB Arena</div>
      <h1>Historical retrieval sample</h1>
      <div class="subtitle">Recall, reciprocal rank, and ranking quality from the same corpus and qrels</div>
    </div>
    <div class="run"><strong>Run ${run.run_id}</strong><br>${questions} questions | top-k ${run.top_k}<br>${run.timestamp.slice(0, 10)}</div>
  </header>
  <table>
    <thead><tr><th>Strategy</th><th class="bar-head">Recall@${run.top_k}</th><th>Recall</th><th>MRR</th><th>NDCG</th></tr></thead>
    <tbody>${rowMarkup}</tbody>
  </table>
  <div class="note">
    <div><strong>Interpretation limit:</strong> the three-document corpus and incomplete chunk labels make this a report example, not a universal ranking.</div>
    <div class="source"><strong>Source:</strong> results/run_855aac4e/retriever_lab.json</div>
  </div>
</body>
</html>`;

mkdirSync(dirname(outputPath), { recursive: true });
writeFileSync(outputPath, html);
console.log(outputPath);
