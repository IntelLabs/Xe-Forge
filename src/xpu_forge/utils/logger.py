"""
Logger
"""

import csv
from datetime import datetime
from pathlib import Path
from typing import Any

from xpu_forge.models import OptimizationResult


class ResultLogger:
    """Logs optimization results to files"""

    def __init__(self, log_dir: str = "./outputs/logs", kernel_dir: str = "./outputs/kernels"):
        self.log_dir = Path(log_dir)
        self.kernel_dir = Path(kernel_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.kernel_dir.mkdir(parents=True, exist_ok=True)

        self.csv_path = self.log_dir / f"results_{datetime.now().strftime('%Y%m%d')}.csv"
        self._initialize_csv()

    def _initialize_csv(self):
        """Initialize CSV file with headers"""
        if not self.csv_path.exists():
            headers = [
                "timestamp",
                "kernel_name",
                "success",
                "stages_applied",
                "original_tflops",
                "optimized_tflops",
                "speedup",
                "issues_found",
            ]
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()

    def log_result(self, result: OptimizationResult):
        """Log optimization result"""
        row = {
            "timestamp": result.timestamp.isoformat(),
            "kernel_name": result.kernel_name,
            "success": result.success,
            "stages_applied": ",".join([s.stage.value for s in result.stages_applied]),
            "original_tflops": result.original_tflops or 0,
            "optimized_tflops": result.optimized_tflops or 0,
            "speedup": result.total_speedup or 0,
            "issues_found": len(result.analysis.detected_issues) if result.analysis else 0,
        }

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writerow(row)

    def get_summary(self) -> dict[str, Any]:
        """Get summary statistics"""
        if not self.csv_path.exists():
            return {}

        with open(self.csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            return {}

        total = len(rows)
        successful = sum(1 for r in rows if r["success"] == "True")
        speedups = [float(r["speedup"]) for r in rows if float(r["speedup"]) > 0]

        return {
            "total_optimizations": total,
            "success_rate": successful / total if total > 0 else 0,
            "average_speedup": sum(speedups) / len(speedups) if speedups else 0,
            "max_speedup": max(speedups) if speedups else 0,
        }
