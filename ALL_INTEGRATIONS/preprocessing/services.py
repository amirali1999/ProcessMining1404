"""
Service layer for Event Log preprocessing (Group 3 integration).

This module wraps Group 3's LogPreprocessor and provides clean interfaces
for other groups (4-7) to consume event logs consistently.
"""

import os
import sys
from pathlib import Path
from typing import Literal, Dict, Any, Optional
import pandas as pd
from django.conf import settings
from django.core.files.base import ContentFile

# Import Group 3's preprocessing code
sys.path.insert(0, str(Path(__file__).parent.parent / 'Group3'))
from log_preprocess import LogPreprocessor


def get_event_log_dataframe(
    event_log_id: int, 
    version: Literal["raw", "cleaned"] = "raw"
) -> pd.DataFrame:
    """
    Load an event log as a pandas DataFrame.
    
    This is the CANONICAL way for all groups to access event log data.
    
    Parameters
    ----------
    event_log_id : int
        The ID of the EventLog model instance
    version : {"raw", "cleaned"}, default="raw"
        Which version to load
        
    Returns
    -------
    pd.DataFrame
        The event log with standardized columns
        
    Raises
    ------
    ValueError
        If event log not found or cleaned version requested but doesn't exist
    """
    from uploads.models import EventLog
    
    try:
        event_log = EventLog.objects.select_related('uploaded_file').get(pk=event_log_id)
    except EventLog.DoesNotExist:
        raise ValueError(f"EventLog with id={event_log_id} not found")
    
    if version == "raw":
        file_path = event_log.uploaded_file.file.path
        file_type = event_log.file_type
    else:  # cleaned
        if not event_log.cleaned_file_path:
            raise ValueError(
                f"Cleaned version not available for EventLog id={event_log_id}. "
                "Run Smart Clean first."
            )
        file_path = event_log.cleaned_file_path.path
        file_type = 'parquet'  # We store cleaned logs as parquet for efficiency
    
    return _load_dataframe_from_file(file_path, file_type)


def _load_dataframe_from_file(file_path: str, file_type: str) -> pd.DataFrame:
    """
    Internal helper to load a DataFrame from disk.
    
    Handles CSV, XES, and Parquet formats using Group 3's LogPreprocessor.
    """
    if file_type == 'parquet':
        return pd.read_parquet(file_path)
    
    # Use Group 3's loader for CSV and XES
    preprocessor = LogPreprocessor()
    
    if file_type in ('csv', 'xes'):
        # Group3's load() expects files in ./data/ directory
        # We'll load directly instead
        if file_type == 'csv':
            df = pd.read_csv(file_path)
            # Apply PM4Py conversions if needed
            from pm4py.objects.log.util import dataframe_utils
            from pm4py.objects.conversion.log import converter as log_converter
            df = dataframe_utils.convert_timestamp_columns_in_df(df)
            # Store in preprocessor
            preprocessor.df = df
            preprocessor.df_original = df.copy()
            return df
        else:  # xes
            from pm4py.objects.log.importer.xes import importer as xes_importer
            from pm4py.objects.conversion.log import converter as log_converter
            log = xes_importer.apply(file_path)
            df = log_converter.apply(log, variant=log_converter.Variants.TO_DATA_FRAME)
            preprocessor.df = df
            preprocessor.df_original = df.copy()
            return df
    
    raise ValueError(f"Unsupported file type: {file_type}")


def smart_clean_event_log(
    event_log_id: int,
    aggressive: bool = False,
    normalize_names: bool = True
) -> Dict[str, Any]:
    """
    Apply Group 3's Smart Clean pipeline to an event log.
    
    This function:
    1. Loads the RAW DataFrame
    2. Applies Group 3's cleaning pipeline
    3. Saves the cleaned DataFrame to disk (Parquet format)
    4. Updates EventLog metadata
    5. Returns cleaning statistics
    
    Parameters
    ----------
    event_log_id : int
        The ID of the EventLog to clean
    aggressive : bool, default=False
        Whether to use aggressive cleaning
    normalize_names : bool, default=True
        Whether to normalize column names
        
    Returns
    -------
    dict
        Cleaning statistics including:
        - num_cases_raw, num_events_raw, num_activities_raw
        - num_cases_cleaned, num_events_cleaned, num_activities_cleaned
        - cleaned_file_path
    """
    from uploads.models import EventLog
    
    try:
        event_log = EventLog.objects.select_related('uploaded_file').get(pk=event_log_id)
    except EventLog.DoesNotExist:
        raise ValueError(f"EventLog with id={event_log_id} not found")
    
    # Load raw data
    df_raw = get_event_log_dataframe(event_log_id, version="raw")
    
    # Compute raw stats
    raw_stats = _compute_log_stats(df_raw)
    
    # Apply Group 3's smart_clean
    preprocessor = LogPreprocessor(df=df_raw.copy())
    df_cleaned = preprocessor.smart_clean(
        aggressive=aggressive,
        normalize_names=normalize_names,
        scope="all",
        inplace=True
    )
    
    # Compute cleaned stats
    cleaned_stats = _compute_log_stats(df_cleaned)
    
    # Save cleaned DataFrame to storage
    cleaned_file_path = _save_cleaned_dataframe(
        df_cleaned, 
        event_log_id, 
        event_log.name
    )
    
    # Update EventLog model
    event_log.cleaned_file_path = cleaned_file_path
    event_log.meta_info = {
        **event_log.meta_info,
        'raw': raw_stats,
        'cleaned': cleaned_stats,
        'last_cleaned_at': pd.Timestamp.now().isoformat(),
    }
    event_log.save()
    
    return {
        'status': 'ok',
        'raw': raw_stats,
        'cleaned': cleaned_stats,
        'cleaned_file_path': str(cleaned_file_path),
    }


def _compute_log_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute basic statistics for an event log DataFrame.
    
    Returns
    -------
    dict
        Statistics including num_cases, num_events, num_activities, etc.
    """
    stats = {
        'num_events': len(df),
        'num_columns': len(df.columns),
    }
    
    # Try to find case ID column (common names)
    case_col = None
    for col_name in ['case:concept:name', 'case_id', 'caseid', 'case']:
        if col_name in df.columns:
            case_col = col_name
            break
    
    if case_col:
        stats['num_cases'] = df[case_col].nunique()
    
    # Try to find activity column
    activity_col = None
    for col_name in ['concept:name', 'activity', 'activity_name']:
        if col_name in df.columns:
            activity_col = col_name
            break
    
    if activity_col:
        stats['num_activities'] = df[activity_col].nunique()
        stats['activities'] = df[activity_col].value_counts().head(10).to_dict()
    
    # Try to find timestamp column
    timestamp_col = None
    for col_name in ['time:timestamp', 'timestamp', 'time']:
        if col_name in df.columns:
            timestamp_col = col_name
            break
    
    if timestamp_col and pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        stats['time_range'] = {
            'start': df[timestamp_col].min().isoformat(),
            'end': df[timestamp_col].max().isoformat(),
        }
    
    return stats


def _save_cleaned_dataframe(
    df: pd.DataFrame, 
    event_log_id: int, 
    log_name: str
) -> str:
    """
    Save a cleaned DataFrame to storage in Parquet format.
    
    Returns the relative file path for Django FileField.
    """
    from django.core.files.storage import default_storage
    from datetime import datetime
    
    # Create a filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    safe_name = "".join(c if c.isalnum() else "_" for c in log_name)
    filename = f"cleaned_logs/{datetime.now().year}/{datetime.now().month:02d}/{datetime.now().day:02d}/{safe_name}_{timestamp}.parquet"
    
    # Save to temporary buffer
    import io
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=False, engine='pyarrow')
    buffer.seek(0)
    
    # Save via Django storage
    path = default_storage.save(filename, ContentFile(buffer.read()))
    
    return path


def get_default_event_log_df(event_log_id: int) -> pd.DataFrame:
    """
    Load event log using the configured default source.
    
    This is a convenience function for Groups 4-7 to use.
    It respects the `default_source_for_downstream` setting.
    
    Parameters
    ----------
    event_log_id : int
        The ID of the EventLog
        
    Returns
    -------
    pd.DataFrame
        The event log (either raw or cleaned based on settings)
    """
    from uploads.models import EventLog
    
    try:
        event_log = EventLog.objects.get(pk=event_log_id)
    except EventLog.DoesNotExist:
        raise ValueError(f"EventLog with id={event_log_id} not found")
    
    version = event_log.default_source_for_downstream
    return get_event_log_dataframe(event_log_id, version=version)


def get_event_log_table_data(
    event_log_id: int,
    version: Literal["raw", "cleaned"] = "raw",
    page: int = 1,
    page_size: int = 50
) -> Dict[str, Any]:
    """
    Get paginated table data for display in the UI.
    
    Parameters
    ----------
    event_log_id : int
        The ID of the EventLog
    version : {"raw", "cleaned"}
        Which version to display
    page : int, default=1
        Page number (1-indexed)
    page_size : int, default=50
        Number of rows per page
        
    Returns
    -------
    dict
        Contains 'columns', 'data', 'total_rows', 'page', 'total_pages'
    """
    df = get_event_log_dataframe(event_log_id, version=version)
    
    total_rows = len(df)
    total_pages = (total_rows + page_size - 1) // page_size
    
    # Validate page
    page = max(1, min(page, total_pages if total_pages > 0 else 1))
    
    # Get page slice
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    df_page = df.iloc[start_idx:end_idx]
    
    # Convert to JSON-serializable format
    # Handle timestamps and other non-serializable types
    df_page = df_page.copy()
    
    # Convert categorical columns to string first to avoid category errors
    for col in df_page.columns:
        if pd.api.types.is_categorical_dtype(df_page[col]):
            df_page[col] = df_page[col].astype(str)
        elif pd.api.types.is_datetime64_any_dtype(df_page[col]):
            df_page[col] = df_page[col].astype(str)
    
    # Replace NaN, inf, and -inf with None for JSON compatibility
    df_page = df_page.replace({pd.NA: None, pd.NaT: None})
    df_page = df_page.replace([float('inf'), float('-inf')], None)
    df_page = df_page.fillna(value='')  # Replace remaining NaNs with empty string
    
    return {
        'columns': df_page.columns.tolist(),
        'data': df_page.values.tolist(),
        'total_rows': total_rows,
        'page': page,
        'page_size': page_size,
        'total_pages': total_pages,
    }
