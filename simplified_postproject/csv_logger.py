"""Lightweight CSV/XLSX evaluation logger for optimization runs.

Writes one row per evaluation with:
  eval_no, gen, particle, score, nrmse, ssim, emd, ncc, gmsd, lpips,
  <all 24 parameter values>, timestamp

Usage:
    csv_log = EvalCSVLogger(output_dir, params_list)
    csv_log.log(eval_no, gen, pid, score, metrics_dict, x_full)
    csv_log.close()
"""
from __future__ import annotations

import csv
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

METRIC_COLS = ["nrmse", "ssim", "emd", "ncc", "gmsd", "lpips", "brightness"]


class EvalCSVLogger:
    """Append-mode CSV logger for each optimization evaluation."""

    def __init__(
        self,
        output_dir: str | Path,
        params: list,  # config.PARAMS list
        n_frames: int = 0,
    ) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = output_dir / f"eval_history_{stamp}.csv"
        self.xlsx_path = output_dir / f"eval_history_{stamp}.xlsx"
        self.param_names = [p.name for p in params]
        self._n_frames = n_frames
        self._records: list[dict] = []

        # Per-frame render param columns: f0_sun_energy, f0_film_exp, ...
        render_cols = []
        for i in range(n_frames):
            render_cols += [f"f{i}_sun_energy", f"f{i}_film_exp"]

        # Build header
        self._header = (
            ["eval_no", "gen", "particle", "score"]
            + METRIC_COLS
            + self.param_names
            + render_cols
            + ["timestamp"]
        )

        self._file = open(self.csv_path, "w", newline="", buffering=1)
        self._writer = csv.DictWriter(self._file, fieldnames=self._header)
        self._writer.writeheader()

        logger.info("CSV logger opened: %s", self.csv_path)

    # --------------------------------------------------------------------- #

    def log(
        self,
        eval_no: int,
        gen: int,
        particle: int,
        score: float,
        metrics: dict,
        x_full: list[float],
        render_info: list[dict] | None = None,
    ) -> None:
        """Append one evaluation row.

        render_info: list of per-frame dicts with 'sun_energy' and 'film_exp'
        (from scene_mgr.last_render_info).
        """
        row: dict = {
            "eval_no": eval_no,
            "gen": gen,
            "particle": particle,
            "score": f"{score:.6f}",
        }
        for m in METRIC_COLS:
            row[m] = f"{metrics.get(m, float('nan')):.6f}"
        for name, val in zip(self.param_names, x_full):
            row[name] = f"{val:.6f}"
        for i in range(self._n_frames):
            info = (render_info or [])[i] if render_info and i < len(render_info) else {}
            row[f"f{i}_sun_energy"] = f"{info.get('sun_energy', float('nan')):.8g}"
            row[f"f{i}_film_exp"] = f"{info.get('film_exp', float('nan')):.6f}"
        row["timestamp"] = datetime.now().isoformat(timespec="seconds")

        self._writer.writerow(row)
        self._records.append(row)

    # --------------------------------------------------------------------- #

    def close(self) -> None:
        """Flush CSV and optionally write XLSX."""
        self._file.close()
        logger.info("CSV saved: %s (%d rows)", self.csv_path, len(self._records))

        # XLSX (best-effort)
        try:
            import pandas as pd

            df = pd.DataFrame(self._records)
            df.to_excel(self.xlsx_path, index=False, engine="openpyxl")
            logger.info("XLSX saved: %s", self.xlsx_path)
        except Exception as exc:
            logger.warning("XLSX save skipped: %s", exc)
