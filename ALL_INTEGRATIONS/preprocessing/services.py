"""
Service layer for Event Log preprocessing (Group 3).

This module provides the OFFICIAL API for:
- Phase 1: Event Log loading & Smart Cleaning
- Phase 2: Trace Analysis & Frequent Pattern Mining

Other groups (4–7) MUST use only this module to access
preprocessing functionality.
"""

from typing import Literal, Dict, Any, Optional
from datetime import datetime
import io

import pandas as pd
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage

# ==============================
# Domain imports (Group 3)
# ==============================

from preprocessing.domain.log_preprocess import LogPreprocessor
from preprocessing.domain.trace_analyzer import TraceAnalyzer
from preprocessing.domain.pattern_miner import PatternMiner


# ======================================================================
# Phase 1 – Event Log Loading
# ======================================================================

def get_event_log_dataframe(
    event_log_id: int,
    version: Literal["raw", "cleaned"] = "raw"
) -> pd.DataFrame:
    """
    Load an event log as a pandas DataFrame.

    This is the CANONICAL way for all groups to access event log data.
    """

    from uploads.models import EventLog

    try:
        event_log = EventLog.objects.select_related("uploaded_file").get(pk=event_log_id)
    except EventLog.DoesNotExist:
        raise ValueError(f"EventLog with id={event_log_id} not found")

    if version == "raw":
        file_path = event_log.uploaded_file.file.path
        file_type = event_log.file_type
    else:
        if not event_log.cleaned_file_path:
            raise ValueError(
                f"Cleaned version not available for EventLog id={event_log_id}. "
                "Run Smart Clean first."
            )
        file_path = event_log.cleaned_file_path.path
        file_type = "parquet"

    return _load_dataframe_from_file(file_path, file_type)


def _load_dataframe_from_file(file_path: str, file_type: str) -> pd.DataFrame:
    """
    Internal helper to load a DataFrame from disk.
    Supports CSV, XES and Parquet.
    """

    if file_type == "parquet":
        return pd.read_parquet(file_path)

    if file_type == "csv":
        df = pd.read_csv(file_path)
        from pm4py.objects.log.util import dataframe_utils
        df = dataframe_utils.convert_timestamp_columns_in_df(df)
        return df

    if file_type == "xes":
        from pm4py.objects.log.importer.xes import importer as xes_importer
        from pm4py.objects.conversion.log import converter as log_converter
        log = xes_importer.apply(file_path)
        return log_converter.apply(
            log,
            variant=log_converter.Variants.TO_DATA_FRAME
        )

    raise ValueError(f"Unsupported file type: {file_type}")


def get_default_event_log_df(event_log_id: int) -> pd.DataFrame:
    """
    Load event log using the configured default source
    (raw or cleaned).
    """

    from uploads.models import EventLog

    try:
        event_log = EventLog.objects.get(pk=event_log_id)
    except EventLog.DoesNotExist:
        raise ValueError(f"EventLog with id={event_log_id} not found")

    return get_event_log_dataframe(
        event_log_id,
        version=event_log.default_source_for_downstream
    )


# ======================================================================
# Phase 1 – Smart Cleaning
# ======================================================================

def smart_clean_event_log(
    event_log_id: int,
    aggressive: bool = False,
    normalize_names: bool = True
) -> Dict[str, Any]:
    """
    Apply Group 3 Smart Clean pipeline to an event log.
    """

    from uploads.models import EventLog

    try:
        event_log = EventLog.objects.select_related("uploaded_file").get(pk=event_log_id)
    except EventLog.DoesNotExist:
        raise ValueError(f"EventLog with id={event_log_id} not found")

    df_raw = get_event_log_dataframe(event_log_id, version="raw")
    raw_stats = _compute_log_stats(df_raw)

    preprocessor = LogPreprocessor(df=df_raw.copy())
    df_cleaned = preprocessor.smart_clean(
        aggressive=aggressive,
        normalize_names=normalize_names,
        scope="all",
        inplace=True
    )

    cleaned_stats = _compute_log_stats(df_cleaned)
    cleaned_path = _save_cleaned_dataframe(
        df_cleaned,
        event_log_id,
        event_log.name
    )

    event_log.cleaned_file_path = cleaned_path
    event_log.meta_info = {
        **event_log.meta_info,
        "raw": raw_stats,
        "cleaned": cleaned_stats,
        "last_cleaned_at": pd.Timestamp.now().isoformat(),
    }
    event_log.save()

    return {
        "status": "ok",
        "raw": raw_stats,
        "cleaned": cleaned_stats,
        "cleaned_file_path": str(cleaned_path),
    }


def _compute_log_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute lightweight descriptive statistics for an event log.
    """

    stats: Dict[str, Any] = {
        "num_events": len(df),
        "num_columns": len(df.columns),
    }

    case_cols = ["case:concept:name", "case_id", "caseid", "case"]
    act_cols = ["concept:name", "activity", "activity_name"]
    time_cols = ["time:timestamp", "timestamp", "time"]

    case_col = next((c for c in case_cols if c in df.columns), None)
    act_col = next((c for c in act_cols if c in df.columns), None)
    time_col = next((c for c in time_cols if c in df.columns), None)

    if case_col:
        stats["num_cases"] = df[case_col].nunique()

    if act_col:
        stats["num_activities"] = df[act_col].nunique()
        stats["top_activities"] = (
            df[act_col].value_counts().head(10).to_dict()
        )

    if time_col and pd.api.types.is_datetime64_any_dtype(df[time_col]):
        stats["time_range"] = {
            "start": df[time_col].min().isoformat(),
            "end": df[time_col].max().isoformat(),
        }

    return stats


def _save_cleaned_dataframe(
    df: pd.DataFrame,
    event_log_id: int,
    log_name: str
) -> str:
    """
    Save cleaned DataFrame as Parquet via Django storage.
    """

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(c if c.isalnum() else "_" for c in log_name)

    filename = (
        f"cleaned_logs/{datetime.now().year}/"
        f"{datetime.now().month:02d}/"
        f"{datetime.now().day:02d}/"
        f"{safe_name}_{timestamp}.parquet"
    )

    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False, engine="pyarrow")
    buffer.seek(0)

    return default_storage.save(filename, ContentFile(buffer.read()))


# ======================================================================
# Phase 2 – Trace Analysis & Pattern Mining
# ======================================================================

def run_phase2_analysis(
    event_log_id: int,
    min_support: int = 2,
    max_pattern_len: Optional[int] = None
) -> Dict[str, Any]:
    """
    OFFICIAL Phase 2 API (Group 3).

    Steps:
    1. Load default event log
    2. Build traces (TraceAnalyzer)
    3. Mine frequent patterns (PatternMiner)
    """

    df = get_default_event_log_df(event_log_id)

    trace_analyzer = TraceAnalyzer(df)
    traces = trace_analyzer.build_traces()

    pattern_miner = PatternMiner(
        traces=traces,
        min_support=min_support
    )

    patterns_df = pattern_miner.mine_patterns(
        max_len=max_pattern_len
    )

    return {
        "status": "ok",
        "num_traces": int(len(traces)),
        "num_patterns": int(len(patterns_df)),
        "top_patterns": patterns_df.head(10).to_dict("records"),
    }
