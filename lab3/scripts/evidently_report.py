"""Генерує HTML-звіт про drift (власна реалізація без evidently)."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy import stats

ROOT     = Path(__file__).resolve().parent.parent
ref_data = joblib.load(ROOT / "reference_stats.joblib")
ref_df   = pd.DataFrame(ref_data["X"], columns=ref_data["feature_names"])

# Імітуємо drifted live-вибірку
rng     = np.random.default_rng(0)
current = ref_df.copy().sample(n=200, random_state=0, replace=True)
current["petal_length"] = current["petal_length"] + 1.5

# KS-тест для кожної ознаки
results = []
for col in ref_data["feature_names"]:
    ks_stat, p_value = stats.ks_2samp(ref_df[col], current[col])
    results.append({
        "feature":        col,
        "ref_mean":       round(ref_df[col].mean(), 4),
        "cur_mean":       round(current[col].mean(), 4),
        "ref_std":        round(ref_df[col].std(), 4),
        "cur_std":        round(current[col].std(), 4),
        "ks_statistic":   round(ks_stat, 4),
        "p_value":        round(p_value, 6),
        "drift_detected": p_value < 0.05,
    })

def make_histogram_data(series, bins=20):
    counts, edges = np.histogram(series, bins=bins)
    centers = [(edges[i] + edges[i+1]) / 2 for i in range(len(edges)-1)]
    return centers, counts.tolist()

feature_charts = ""
for r in results:
    col   = r["feature"]
    rx, rc = make_histogram_data(ref_df[col])
    cx, cc = make_histogram_data(current[col])
    drift_badge = (
        '<span style="color:#e74c3c;font-weight:bold">⚠ DRIFT</span>'
        if r["drift_detected"]
        else '<span style="color:#27ae60;font-weight:bold">✓ OK</span>'
    )
    feature_charts += f"""
    <div class="feature-card {'drift' if r['drift_detected'] else 'ok'}">
        <h3>{col} {drift_badge}</h3>
        <div class="stats-row">
            <div class="stat-box">
                <b>Reference</b><br>
                Mean: {r['ref_mean']}<br>
                Std:  {r['ref_std']}
            </div>
            <div class="stat-box">
                <b>Current</b><br>
                Mean: {r['cur_mean']}<br>
                Std:  {r['cur_std']}
            </div>
            <div class="stat-box {'alert' if r['drift_detected'] else ''}">
                <b>KS Test</b><br>
                Statistic: {r['ks_statistic']}<br>
                p-value: {r['p_value']}
            </div>
        </div>
        <canvas id="chart_{col}" width="600" height="200"></canvas>
        <script>
        (function(){{
            var ctx = document.getElementById('chart_{col}').getContext('2d');
            new Chart(ctx, {{
                type: 'bar',
                data: {{
                    labels: {rx},
                    datasets: [
                        {{
                            label: 'Reference',
                            data: {rc},
                            backgroundColor: 'rgba(52,152,219,0.5)',
                            borderColor: 'rgba(52,152,219,1)',
                            borderWidth: 1
                        }},
                        {{
                            label: 'Current',
                            data: {cc},
                            backgroundColor: 'rgba(231,76,60,0.4)',
                            borderColor: 'rgba(231,76,60,1)',
                            borderWidth: 1
                        }}
                    ]
                }},
                options: {{
                    responsive: false,
                    plugins: {{ title: {{ display: false }} }},
                    scales: {{ y: {{ beginAtZero: true }} }}
                }}
            }});
        }})();
        </script>
    </div>
    """

drifted     = [r["feature"] for r in results if r["drift_detected"]]
total       = len(results)
n_drifted   = len(drifted)
overall_badge = (
    '<span class="badge-red">DRIFT DETECTED</span>'
    if n_drifted > 0
    else '<span class="badge-green">NO DRIFT</span>'
)

html = f"""<!DOCTYPE html>
<html lang="uk">
<head>
<meta charset="UTF-8">
<title>Data Drift Report — Iris ML API</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
<style>
  body {{ font-family: Arial, sans-serif; background: #f5f6fa; margin: 0; padding: 20px; }}
  .container {{ max-width: 900px; margin: auto; }}
  h1 {{ color: #2c3e50; }}
  .summary {{ background: white; border-radius: 8px; padding: 20px;
              margin-bottom: 24px; box-shadow: 0 2px 6px rgba(0,0,0,0.08); }}
  .feature-card {{ background: white; border-radius: 8px; padding: 20px;
                   margin-bottom: 20px; box-shadow: 0 2px 6px rgba(0,0,0,0.08); }}
  .feature-card.drift {{ border-left: 5px solid #e74c3c; }}
  .feature-card.ok    {{ border-left: 5px solid #27ae60; }}
  .stats-row {{ display: flex; gap: 16px; margin: 12px 0; }}
  .stat-box {{ background: #f8f9fa; border-radius: 6px; padding: 12px;
               flex: 1; font-size: 14px; }}
  .stat-box.alert {{ background: #fff5f5; border: 1px solid #e74c3c; }}
  .badge-red   {{ background:#e74c3c; color:white; padding:4px 12px;
                  border-radius:4px; font-size:16px; font-weight:bold; }}
  .badge-green {{ background:#27ae60; color:white; padding:4px 12px;
                  border-radius:4px; font-size:16px; font-weight:bold; }}
  table {{ width:100%; border-collapse:collapse; font-size:14px; }}
  th, td {{ padding:10px; border-bottom:1px solid #eee; text-align:left; }}
  th {{ background:#f0f3f7; }}
</style>
</head>
<body>
<div class="container">
  <h1>📊 Data Drift Report — Iris ML API</h1>

  <div class="summary">
    <h2>Загальний результат: {overall_badge}</h2>
    <p>Перевірено ознак: <b>{total}</b> &nbsp;|&nbsp;
       Виявлено drift: <b>{n_drifted}</b> &nbsp;|&nbsp;
       Поріг значущості α = 0.05</p>
    <table>
      <tr><th>Ознака</th><th>KS Statistic</th><th>p-value</th><th>Результат</th></tr>
      {''.join(f"""<tr>
        <td>{r['feature']}</td>
        <td>{r['ks_statistic']}</td>
        <td>{r['p_value']}</td>
        <td>{'⚠ Drift' if r['drift_detected'] else '✓ OK'}</td>
      </tr>""" for r in results)}
    </table>
  </div>

  <h2>Розподіли ознак (Reference vs Current)</h2>
  {feature_charts}
</div>
</body>
</html>
"""

out = ROOT / "drift_report.html"
out.write_text(html, encoding="utf-8")
print(f" Report saved to {out}")