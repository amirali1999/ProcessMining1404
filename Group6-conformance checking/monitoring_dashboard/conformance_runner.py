
import os
import json
import argparse
import pandas as pd

import pm4py
from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
from pm4py.objects.petri_net.importer import importer as petri_importer
from pm4py.visualization.petri_net import visualizer as petri_visualizer
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness
from pm4py.algo.evaluation.precision import algorithm as precision_evaluator


def load_pnml_model(pnml_file_path):
    net, initial_marking, final_marking = petri_importer.apply(pnml_file_path)
    return net, initial_marking, final_marking


def load_csv_log(csv_file_path, timestamp_key="time:timestamp"):
    df = pd.read_csv(csv_file_path)

    if timestamp_key not in df.columns:
        candidates = ["timestamp", "time", "datetime", "event_time", "time:timestamp"]
        found = next((c for c in candidates if c in df.columns), None)
        if found is None:
            raise KeyError(f"ستون زمان پیدا نشد. ستون های موجود: {list(df.columns)}")
        timestamp_key = found

    df[timestamp_key] = pd.to_datetime(df[timestamp_key], errors="coerce")
    if df[timestamp_key].isna().any():
        bad_rows = df[df[timestamp_key].isna()].head(5)
        raise ValueError(f"بعضی زمان ها قابل تبدیل به datetime نیستند. نمونه:\n{bad_rows}")

    if "case:concept:name" not in df.columns and "case_id" in df.columns:
        df["case:concept:name"] = df["case_id"]

    if "concept:name" not in df.columns and "activity" in df.columns:
        df["concept:name"] = df["activity"]

    if "case:concept:name" not in df.columns:
        raise KeyError("ستون case:concept:name یا case_id در CSV وجود ندارد.")
    if "concept:name" not in df.columns:
        raise KeyError("ستون concept:name یا activity در CSV وجود ندارد.")

    log = pm4py.format_dataframe(
        df,
        case_id="case:concept:name",
        activity_key="concept:name",
        timestamp_key=timestamp_key,
    )
    return log, df, timestamp_key


def apply_token_replay(log, net, initial_marking, final_marking):
    return token_replay.apply(log, net, initial_marking, final_marking)


def check_for_anomalies(replayed_traces, df):
    case_names = list(df["case:concept:name"].drop_duplicates())
    compliant_cases = []
    non_compliant_cases = []

    for i, trace_result in enumerate(replayed_traces):
        current_case_name = case_names[i]
        case_data = df[df["case:concept:name"] == current_case_name]

        if not trace_result.get("trace_is_fit", False):
            non_compliant_cases.append(case_data)
        else:
            compliant_cases.append(case_data)

    return compliant_cases, non_compliant_cases


def calculate_compliance_percentage(replayed_traces):
    total_cases = len(replayed_traces)
    compliant_cases = sum(1 for tr in replayed_traces if tr.get("trace_is_fit", False))
    non_compliant_cases = total_cases - compliant_cases

    compliant_percentage = (compliant_cases / total_cases)  if total_cases else 0.0
    non_compliant_percentage = (non_compliant_cases / total_cases) if total_cases else 0.0

    return {
        "total_cases": total_cases,
        "compliant_cases": compliant_cases,
        "non_compliant_cases": non_compliant_cases,
        "compliant_percentage": compliant_percentage,
        "non_compliant_percentage": non_compliant_percentage,
    }


def calculate_advanced_conformance(log, net, initial_marking, final_marking):
    fitness_res = replay_fitness.apply(
        log, net, initial_marking, final_marking,
        variant=replay_fitness.Variants.TOKEN_BASED
    )
    fitness_summary = {k: float(v) if isinstance(v, (int, float)) else v for k, v in fitness_res.items()}

    precision_value = precision_evaluator.apply(
        log, net, initial_marking, final_marking,
        variant=precision_evaluator.Variants.ALIGN_ETCONFORMANCE
    )

    return {"fitness": fitness_summary, "precision": float(precision_value)}


def save_results_to_csv(compliant_cases, non_compliant_cases, outdir, base_name):
    compliant_df = pd.concat(compliant_cases) if len(compliant_cases) else pd.DataFrame()
    non_compliant_df = pd.concat(non_compliant_cases) if len(non_compliant_cases) else pd.DataFrame()

    compliant_path = os.path.join(outdir, f"compliant_{base_name}.csv")
    non_compliant_path = os.path.join(outdir, f"non_compliant_{base_name}.csv")

    compliant_df.to_csv(compliant_path, index=False)
    non_compliant_df.to_csv(non_compliant_path, index=False)

    return compliant_path, non_compliant_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pnml", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    pnml_file_path = args.pnml
    csv_file_path = args.csv
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    pnml_file_name = os.path.splitext(os.path.basename(pnml_file_path))[0]
    csv_file_name = os.path.splitext(os.path.basename(csv_file_path))[0]
    base_name = f"{pnml_file_name}_{csv_file_name}"

    net, initial_marking, final_marking = load_pnml_model(pnml_file_path)
    log, df, timestamp_key = load_csv_log(csv_file_path, timestamp_key="time:timestamp")
    replayed_traces = apply_token_replay(log, net, initial_marking, final_marking)
    compliant_cases, non_compliant_cases = check_for_anomalies(replayed_traces, df)
    stats = calculate_compliance_percentage(replayed_traces)
    advanced = calculate_advanced_conformance(log, net, initial_marking, final_marking)

    compliant_path, non_compliant_path = save_results_to_csv(compliant_cases, non_compliant_cases, outdir, base_name)

    summary = {
        "model_file": os.path.basename(pnml_file_path),
        "log_file": os.path.basename(csv_file_path),
        "timestamp_key_used": timestamp_key,
        "stats": stats,
        "advanced_conformance": advanced
    }

    summary_path = os.path.join(outdir, f"{base_name}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    print(summary_path)
    print(compliant_path)
    print(non_compliant_path)

if __name__ == "__main__":
    main()
