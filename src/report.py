"""Generate a readable HTML summary from experiment CSV logs."""

from __future__ import annotations

import argparse
from html import escape
from pathlib import Path

import pandas as pd


def make_report_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a compact leaderboard view for manual browsing in the HTML report."""
    columns = ["model", "feature_set", "phase", "fold", "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "low_f1", "medium_f1", "high_f1"]
    for column in columns:
        if column not in frame.columns:
            frame = frame.copy()
            frame[column] = "" if column in {"phase", "fold", "feature_set"} else float("nan")
    return (
        frame.loc[:, columns]
        .sort_values(["macro_f1", "balanced_accuracy"], ascending=False)
        .reset_index(drop=True)
    )


def render_html(frame: pd.DataFrame, title: str = "Model experiment report") -> str:
    """Render a plain HTML leaderboard and parameter table."""
    leaderboard = make_report_table(frame)
    rows = []
    for _, row in leaderboard.iterrows():
        rows.append(
            "<tr>"
            f"<td>{escape(str(row['model']))}</td>"
            f"<td>{escape(str(row['feature_set']))}</td>"
            f"<td>{escape(str(row['phase']))}</td>"
            f"<td>{escape(str(row['fold']) if row['fold'] is not None and not pd.isna(row['fold']) else '')}</td>"
            f"<td>{row['accuracy']:.4f}</td>"
            f"<td>{row['balanced_accuracy']:.4f}</td>"
            f"<td>{row['macro_f1']:.4f}</td>"
            f"<td>{row['weighted_f1']:.4f}</td>"
            f"<td>{row['low_f1']:.4f}</td>"
            f"<td>{row['medium_f1']:.4f}</td>"
            f"<td>{row['high_f1']:.4f}</td>"
            "</tr>"
        )
    table_html = "\n".join(rows)
    return f"""
    <!doctype html>
    <html>
      <head>
        <meta charset="utf-8">
        <title>{escape(title)}</title>
        <style>
          body {{ font-family: Arial, sans-serif; margin: 2rem; color: #1f2933; }}
          table {{ border-collapse: collapse; width: 100%; margin-top: 1rem; }}
          th, td {{ border: 1px solid #d9e2ec; padding: 0.5rem; text-align: left; font-size: 0.9rem; }}
          th {{ background: #f1f5f9; }}
          h1 {{ margin-bottom: 0.5rem; }}
        </style>
      </head>
      <body>
        <h1>{escape(title)}</h1>
        <table>
          <thead>
            <tr>
              <th>Model</th>
              <th>Feature set</th>
              <th>Phase</th>
              <th>Fold</th>
              <th>Accuracy</th>
              <th>Balanced acc.</th>
              <th>Macro F1</th>
              <th>Weighted F1</th>
              <th>Low F1</th>
              <th>Medium F1</th>
              <th>High F1</th>
            </tr>
          </thead>
          <tbody>
            {table_html}
          </tbody>
        </table>
      </body>
    </html>
    """


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a readable HTML summary from an experiment CSV.")
    parser.add_argument("--input", type=Path, default=Path("outputs/experiments.csv"), help="Experiment CSV path")
    parser.add_argument("--output", type=Path, default=Path("outputs/experiment_report.html"), help="HTML report output")
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    html = render_html(frame, title=f"Experiment report: {args.input.stem}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    print(f"Saved HTML report to {args.output}")


if __name__ == "__main__":
    main()
