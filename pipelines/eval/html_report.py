"""Generate a self-contained HTML report from eval/metrics.json + failures.parquet.

Thumbnails are presigned R2 URLs (valid for 24h). FN highlighted red, FP yellow.
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
from pathlib import Path

import pandas as pd
from PIL import Image

from ..common import REPO_ROOT, get_logger
from ..collection import r2_client

log = get_logger(__name__)


_HTML_TPL = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"/><title>Tzniut Eval Report</title>
<style>
  :root {{ --bg:#0a0d11; --panel:#11161d; --line:#1f2630; --text:#e6e9ee; --muted:#8a94a4;
           --reject:#ef4444; --warn:#f59e0b; --accept:#2fbf71; --accent:#4e9cff; }}
  body {{ background:var(--bg); color:var(--text); font-family:-apple-system,Inter,sans-serif; margin:0; }}
  header {{ padding:24px; border-bottom:1px solid var(--line); background:var(--panel); }}
  h1 {{ margin:0 0 8px; font-size:22px; }}
  .summary {{ display:grid; grid-template-columns:repeat(4,1fr); gap:16px; padding:24px; }}
  .stat {{ background:var(--panel); border:1px solid var(--line); padding:16px; border-radius:10px; }}
  .stat .v {{ font-size:28px; font-weight:600; font-family:ui-monospace,monospace; }}
  .stat .l {{ font-size:11px; color:var(--muted); text-transform:uppercase; letter-spacing:0.08em; }}
  .stat.bad .v {{ color:var(--reject); }}
  .stat.warn .v {{ color:var(--warn); }}
  .stat.good .v {{ color:var(--accept); }}
  section {{ padding:0 24px 32px; }}
  h2 {{ margin:24px 0 12px; font-size:14px; letter-spacing:0.08em; color:var(--muted); text-transform:uppercase; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(160px,1fr)); gap:8px; }}
  .card {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:6px; }}
  .card.fn {{ border-color:var(--reject); box-shadow:0 0 0 1px rgba(239,68,68,0.3); }}
  .card.fp {{ border-color:var(--warn); }}
  .card img {{ width:100%; height:160px; object-fit:cover; border-radius:6px; display:block; }}
  .card .row {{ font-family:ui-monospace,monospace; font-size:11px; color:var(--muted); padding:4px 4px 0; display:flex; justify-content:space-between; }}
  .card .row b {{ color:var(--text); }}
  .badge.fn {{ color:var(--reject); }}
  .badge.fp {{ color:var(--warn); }}
  table {{ width:100%; border-collapse:collapse; font-family:ui-monospace,monospace; font-size:12px; }}
  th, td {{ padding:6px 10px; border-bottom:1px solid var(--line); text-align:left; }}
  th {{ color:var(--muted); font-weight:500; font-size:11px; text-transform:uppercase; }}
</style></head><body>
<header>
  <h1>Tzniut Classifier Eval Report</h1>
  <div style="color:var(--muted)">checkpoint metrics + failure inspection</div>
</header>
<div class="summary">
  <div class="stat {recall_class}"><div class="l">recall · NOT_ACCEPTABLE</div><div class="v">{recall:.3f}</div></div>
  <div class="stat {prec_class}"><div class="l">precision · NOT_ACCEPTABLE</div><div class="v">{precision:.3f}</div></div>
  <div class="stat"><div class="l">F2</div><div class="v">{f2:.3f}</div></div>
  <div class="stat {fnr_class}"><div class="l">false negative rate</div><div class="v">{fnr:.3f}</div></div>
</div>
<section>
  <h2>Confusion</h2>
  <table>
    <tr><th></th><th>predicted ACCEPT</th><th>predicted BLOCK</th></tr>
    <tr><th>true ACCEPT</th><td>{tn}</td><td class="badge fp">{fp} (FP)</td></tr>
    <tr><th>true BLOCK</th><td class="badge fn">{fn} (FN)</td><td>{tp}</td></tr>
  </table>
</section>
<section>
  <h2>Per-attribute accuracy</h2>
  <table>{per_head_rows}</table>
</section>
<section>
  <h2>False negatives — model said ALLOW, truth was BLOCK ({n_fn} shown)</h2>
  <div class="grid">{fn_cards}</div>
</section>
<section>
  <h2>False positives — model said BLOCK, truth was ALLOW ({n_fp} shown)</h2>
  <div class="grid">{fp_cards}</div>
</section>
</body></html>
"""


def _thumb_b64(r2_key: str, max_side: int = 200) -> str:
    """Fetch image, downscale to a thumbnail, return base64 WebP data URL."""
    try:
        raw = r2_client.download_bytes(r2_key)
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        img.thumbnail((max_side, max_side), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="WEBP", quality=75)
        return "data:image/webp;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return ""


def _card(row: pd.Series, kind: str) -> str:
    url = _thumb_b64(row["r2_key"]) if row.get("r2_key") else ""
    return f"""
<div class="card {kind}">
  <img src="{html.escape(url)}" loading="lazy" alt="" />
  <div class="row"><b>{html.escape(str(row['image_id'])[:14])}</b><span>{row['score']:.2f}</span></div>
</div>"""


def render(output: str | None = None, max_cards: int = 120) -> Path:
    eval_dir = REPO_ROOT / "models" / "eval"
    metrics = json.loads((eval_dir / "metrics.json").read_text())
    failures = pd.read_parquet(eval_dir / "failures.parquet")

    fns = failures[(failures["true_block"]) & (~failures["predicted_block"])].sort_values("score", ascending=False)
    fps = failures[(~failures["true_block"]) & (failures["predicted_block"])].sort_values("score", ascending=False)

    fn_html = "".join(_card(r, "fn") for _, r in fns.head(max_cards).iterrows())
    fp_html = "".join(_card(r, "fp") for _, r in fps.head(max_cards).iterrows())

    per_head_rows = "".join(
        f"<tr><td>{html.escape(h)}</td><td>{v:.3f}</td></tr>"
        for h, v in metrics["per_head_accuracy"].items()
    )
    per_head_rows = "<tr><th>head</th><th>accuracy</th></tr>" + per_head_rows

    rec = metrics["block_recall_not_acceptable"]
    prec = metrics["block_precision"]
    fnr = metrics["false_negative_rate"]
    page = _HTML_TPL.format(
        recall=rec,
        recall_class="good" if rec >= 0.95 else ("warn" if rec >= 0.85 else "bad"),
        precision=prec,
        prec_class="good" if prec >= 0.85 else ("warn" if prec >= 0.7 else "bad"),
        f2=metrics["block_f2"],
        fnr=fnr,
        fnr_class="good" if fnr <= 0.05 else ("warn" if fnr <= 0.15 else "bad"),
        tp=metrics["confusion"]["tp"],
        fp=metrics["confusion"]["fp"],
        fn=metrics["confusion"]["fn"],
        tn=metrics["confusion"]["tn"],
        per_head_rows=per_head_rows,
        n_fn=min(max_cards, len(fns)),
        n_fp=min(max_cards, len(fps)),
        fn_cards=fn_html,
        fp_cards=fp_html,
    )

    out = Path(output) if output else eval_dir / "report.html"
    out.write_text(page, encoding="utf-8")
    log.info("report → %s", out)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", default=None)
    p.add_argument("--max-cards", type=int, default=120)
    a = p.parse_args()
    render(a.output, a.max_cards)
