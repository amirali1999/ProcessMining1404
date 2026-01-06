
import json
import subprocess
from pathlib import Path
from django.conf import settings
from django.shortcuts import render, get_object_or_404, redirect
from .models import ConformanceRun
from .forms import ConformanceUploadForm


from pathlib import Path
from django.http import FileResponse, Http404

def index(request):
    runs = ConformanceRun.objects.order_by('-created_at')
    latest = runs.first()

    # سری زمانی برای نمودار (قدیمی به جدید)
    runs_rev = list(runs)[::-1]
    labels = [r.created_at.strftime('%Y-%m-%d %H:%M') for r in runs_rev]
    compliant_series = [r.compliant_percentage for r in runs_rev]
    fitness_series = [r.log_fitness for r in runs_rev]
    precision_series = [r.precision for r in runs_rev]

    return render(request, 'monitoring/index.html', {
        'runs': runs,
        'latest': latest,
        'labels': labels,
        'compliant_series': compliant_series,
        'fitness_series': fitness_series,
        'precision_series': precision_series,
    })


def run_detail(request, run_id: int):
    run = get_object_or_404(ConformanceRun, id=run_id)
    return render(request, 'monitoring/run_detail.html', {'run': run})


def upload_and_run(request):
    if request.method == "POST":
        form = ConformanceUploadForm(request.POST, request.FILES)
        if form.is_valid():
            media_root = Path(settings.MEDIA_ROOT)
            media_root.mkdir(parents=True, exist_ok=True)

            pnml = request.FILES["pnml_file"]
            log_csv = request.FILES["log_csv"]

            # هر اجرا در یک فولدر جدا ذخیره شود
            runs_dir = media_root / "runs"
            runs_dir.mkdir(exist_ok=True)
            run_dir = runs_dir / Path(f"run_{subprocess.check_output(['python3','-c','import time;print(int(time.time()))']).decode().strip()}")
            run_dir.mkdir(exist_ok=True)

            pnml_path = run_dir / pnml.name
            csv_path = run_dir / log_csv.name

            with open(pnml_path, "wb") as f:
                for chunk in pnml.chunks():
                    f.write(chunk)

            with open(csv_path, "wb") as f:
                for chunk in log_csv.chunks():
                    f.write(chunk)

            # اجرای اسکریپت تحلیل (در ریشه پروژه)
            cmd = [
                "python3",
                str(Path(settings.BASE_DIR) / "conformance_runner.py"),
                "--pnml", str(pnml_path),
                "--csv", str(csv_path),
                "--outdir", str(run_dir),
            ]
            subprocess.run(cmd, check=True)

            # پیدا کردن summary
            summary_path = run_dir / f"{pnml_path.stem}_{csv_path.stem}_summary.json"
            if not summary_path.exists():
                # fallback: اولین summary موجود
                summaries = list(run_dir.glob("*_summary.json"))
                if not summaries:
                    raise FileNotFoundError("Summary JSON پیدا نشد.")
                summary_path = summaries[0]

            with open(summary_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            stats = data.get("stats", {})
            adv = data.get("advanced_conformance", {}) if isinstance(data.get("advanced_conformance", {}), dict) else {}
            fitness = adv.get("fitness", {}) if isinstance(adv.get("fitness", {}), dict) else {}

            log_fitness = fitness.get("log_fitness")
            avg_tf = fitness.get("average_trace_fitness")
            p_fit = fitness.get("percentage_of_fitting_traces") or fitness.get("perc_fit_traces")
            precision = adv.get("precision")

            compliant_csv_path = run_dir / f"compliant_{pnml_path.stem}_{csv_path.stem}.csv"
            non_compliant_csv_path = run_dir / f"non_compliant_{pnml_path.stem}_{csv_path.stem}.csv"

            run = ConformanceRun.objects.create(
                model_file=data.get("model_file", pnml.name),
                log_file=data.get("log_file", log_csv.name),
                timestamp_key_used=data.get("timestamp_key_used", ""),

                total_cases=int(stats.get("total_cases", 0)),
                compliant_cases=int(stats.get("compliant_cases", 0)),
                non_compliant_cases=int(stats.get("non_compliant_cases", 0)),
                compliant_percentage=float(stats.get("compliant_percentage", 0.0)),
                non_compliant_percentage=float(stats.get("non_compliant_percentage", 0.0)),

                log_fitness=float(log_fitness) if log_fitness is not None else None,
                average_trace_fitness=float(avg_tf) if avg_tf is not None else None,
                percentage_of_fitting_traces=float(p_fit) if p_fit is not None else None,
                precision=float(precision) if precision is not None else None,

                summary_json_path=str(summary_path),
                compliant_csv_path=str(compliant_csv_path) if compliant_csv_path.exists() else "",
                non_compliant_csv_path=str(non_compliant_csv_path) if non_compliant_csv_path.exists() else "",
                pnml_path=str(pnml_path),
                log_csv_path=str(csv_path),
            )

            return redirect("monitoring:run_detail", run_id=run.id)
    else:
        form = ConformanceUploadForm()

    return render(request, "monitoring/upload.html", {"form": form})



from django.views.decorators.http import require_POST

@require_POST
def run_delete(request, run_id: int):
    run = get_object_or_404(ConformanceRun, id=run_id)
    run.delete()
    return redirect("monitoring:index")




def download_run_file(request, run_id: int, kind: str):
    run = get_object_or_404(ConformanceRun, id=run_id)

    mapping = {
        "summary": run.summary_json_path,
        "compliant": run.compliant_csv_path,
        "non_compliant": run.non_compliant_csv_path,
        "pnml": run.pnml_path,
        "input_csv": run.log_csv_path,
    }

    file_path = mapping.get(kind)
    if not file_path:
        raise Http404("نوع فایل نامعتبر است.")

    p = Path(file_path)
    if not p.exists():
        raise Http404("فایل پیدا نشد.")

    return FileResponse(open(p, "rb"), as_attachment=True, filename=p.name)
