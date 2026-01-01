
import json
from pathlib import Path
from django.core.management.base import BaseCommand, CommandError
from monitoring.models import ConformanceRun

class Command(BaseCommand):
    help = "Import a conformance run from summary JSON and (optionally) CSV paths."

    def add_arguments(self, parser):
        parser.add_argument("--summary", required=True, help="Path to <model>_<log>_summary.json")
        parser.add_argument("--compliant_csv", required=False, default="", help="Path to compliant CSV")
        parser.add_argument("--non_compliant_csv", required=False, default="", help="Path to non compliant CSV")
        parser.add_argument("--pnml", required=False, default="", help="Path to PNML (optional)")
        parser.add_argument("--log_csv", required=False, default="", help="Path to input log CSV (optional)")

    def handle(self, *args, **options):
        summary_path = Path(options["summary"])
        if not summary_path.exists():
            raise CommandError(f"Summary JSON not found: {summary_path}")

        with summary_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        stats = data.get("stats", {})
        adv = data.get("advanced_conformance", {})
        fitness = adv.get("fitness", {}) if isinstance(adv, dict) else {}

        log_fitness = fitness.get("log_fitness", None)
        avg_tf = fitness.get("average_trace_fitness", None)
        p_fit = fitness.get("percentage_of_fitting_traces", fitness.get("perc_fit_traces", None))
        precision = adv.get("precision", None) if isinstance(adv, dict) else None

        run = ConformanceRun.objects.create(
            model_file=data.get("model_file", ""),
            log_file=data.get("log_file", ""),
            timestamp_key_used=data.get("timestamp_key_used", ""),

            total_cases=int(stats.get("total_cases", 0)),
            compliant_cases=int(stats.get("compliant_cases", 0)),
            non_compliant_cases=int(stats.get("non_compliant_cases", 0)),
            compliant_percentage=float(stats.get("compliant_percentage", 0.0)),
            non_compliant_percentage=float(stats.get("non_compliant_percentage", 0.0)),

            log_fitness=(float(log_fitness) if log_fitness is not None else None),
            average_trace_fitness=(float(avg_tf) if avg_tf is not None else None),
            percentage_of_fitting_traces=(float(p_fit) if p_fit is not None else None),
            precision=(float(precision) if precision is not None else None),

            summary_json_path=str(summary_path),
            compliant_csv_path=str(options["compliant_csv"] or ""),
            non_compliant_csv_path=str(options["non_compliant_csv"] or ""),
            pnml_path=str(options["pnml"] or ""),
            log_csv_path=str(options["log_csv"] or ""),
        )

        self.stdout.write(self.style.SUCCESS(f"Imported run id={run.id}"))
