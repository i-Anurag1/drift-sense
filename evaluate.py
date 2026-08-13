import argparse
import json
import os
import time
import numpy as np
import cv2

from drift_sense.matcher import locate_reference

HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Drift-Sense Evaluation Report</title>
<style>
  body {{ font-family: -apple-system, 'Segoe UI', Roboto, sans-serif; background: #f7f8fa; color: #1c1f26; margin: 0; padding: 40px; }}
  h1 {{ font-size: 26px; margin-bottom: 4px; }}
  .subtitle {{ color: #6b7280; margin-bottom: 32px; }}
  .stats {{ display: flex; gap: 16px; margin-bottom: 32px; flex-wrap: wrap; }}
  .card {{ background: #ffffff; border: 1px solid #e5e7eb; border-radius: 12px; padding: 18px 22px; min-width: 150px; box-shadow: 0 1px 2px rgba(0,0,0,0.04); }}
  .card .label {{ font-size: 12px; text-transform: uppercase; letter-spacing: 0.04em; color: #6b7280; }}
  .card .value {{ font-size: 24px; font-weight: 600; margin-top: 4px; color: #111827; }}
  table {{ width: 100%; border-collapse: collapse; background: #fff; border-radius: 12px; overflow: hidden; box-shadow: 0 1px 2px rgba(0,0,0,0.04); }}
  th, td {{ padding: 10px 14px; text-align: left; font-size: 13px; border-bottom: 1px solid #eef0f3; }}
  th {{ background: #f1f3f6; color: #374151; font-weight: 600; }}
  tr:hover {{ background: #fafbfc; }}
  .good {{ color: #16a34a; font-weight: 600; }}
  .warn {{ color: #d97706; font-weight: 600; }}
  .bad {{ color: #dc2626; font-weight: 600; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 11px; font-weight: 600; }}
  .badge.dram {{ background: #dbeafe; color: #1d4ed8; }}
  .badge.finfet {{ background: #ede9fe; color: #6d28d9; }}
  .flag {{ background: #fef3c7; color: #92400e; padding: 2px 8px; border-radius: 999px; font-size: 11px; }}
</style>
</head>
<body>
  <h1>Drift-Sense Self-Evaluation Report</h1>
  <div class="subtitle">Generated {timestamp} &middot; {n} image pairs</div>
  <div class="stats">
    <div class="card"><div class="label">Mean Error</div><div class="value">{mean_err:.2f} px</div></div>
    <div class="card"><div class="label">Median Error</div><div class="value">{median_err:.2f} px</div></div>
    <div class="card"><div class="label">P90 Error</div><div class="value">{p90_err:.2f} px</div></div>
    <div class="card"><div class="label">Within 15px</div><div class="value">{within15:.0f}%</div></div>
    <div class="card"><div class="label">Ambiguous Sites</div><div class="value">{n_ambiguous}</div></div>
    <div class="card"><div class="label">Avg Runtime</div><div class="value">{avg_time:.2f}s</div></div>
  </div>
  <table>
    <tr><th>#</th><th>Style</th><th>GT (x, y)</th><th>Pred (x, y)</th><th>Error (px)</th><th>Score</th><th>Ambiguous</th></tr>
    {rows}
  </table>
</body>
</html>
"""


def error_class(err):
    if err < 15:
        return "good"
    if err < 40:
        return "warn"
    return "bad"


def main():
    parser = argparse.ArgumentParser(description="Self-evaluation on the generated dataset")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--output", default="./results/report.html")
    args = parser.parse_args()

    with open(os.path.join(args.data_dir, "ground_truth.json")) as f:
        records = json.load(f)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    errors, rows, times = [], [], []
    n_ambiguous = 0

    for i, r in enumerate(records):
        ref = cv2.imread(os.path.join(args.data_dir, r["reference_path"]), cv2.IMREAD_GRAYSCALE)
        search = cv2.imread(os.path.join(args.data_dir, r["search_path"]), cv2.IMREAD_GRAYSCALE)

        t0 = time.time()
        result = locate_reference(ref, search)
        elapsed = time.time() - t0
        times.append(elapsed)

        err = float(np.hypot(result.x - r["gt_x"], result.y - r["gt_y"]))
        errors.append(err)
        if result.ambiguous:
            n_ambiguous += 1

        rows.append(
            f"<tr><td>{i}</td><td><span class='badge {r['style']}'>{r['style']}</span></td>"
            f"<td>({r['gt_x']:.1f}, {r['gt_y']:.1f})</td>"
            f"<td>({result.x:.1f}, {result.y:.1f})</td>"
            f"<td class='{error_class(err)}'>{err:.2f}</td>"
            f"<td>{result.score:.3f}</td>"
            f"<td>{'<span class=\"flag\">ambiguous</span>' if result.ambiguous else '-'}</td></tr>"
        )
        print(f"[{i + 1}/{len(records)}] {r['style']:6s} err={err:6.2f}px "
              f"score={result.score:.3f} ambiguous={result.ambiguous}")

    errors = np.array(errors)
    html = HTML_TEMPLATE.format(
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        n=len(records),
        mean_err=errors.mean(),
        median_err=np.median(errors),
        p90_err=np.percentile(errors, 90),
        within15=100.0 * np.mean(errors < 15),
        n_ambiguous=n_ambiguous,
        avg_time=np.mean(times),
        rows="\n".join(rows),
    )

    with open(args.output, "w") as f:
        f.write(html)

    print(f"\nMean error: {errors.mean():.2f}px | Median: {np.median(errors):.2f}px | "
          f"P90: {np.percentile(errors, 90):.2f}px")
    print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
