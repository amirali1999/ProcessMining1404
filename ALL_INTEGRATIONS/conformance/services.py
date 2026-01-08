"""
Conformance Checking Service Layer (Group 6)

Token Replay implementation for process compliance analysis.
NO file paths - uses existing EventLog and DiscoveredProcessModel data.
"""
from typing import Dict, List, Literal
import pandas as pd
import pm4py
from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
from pm4py.objects.log.obj import EventLog as PM4PyEventLog

from uploads.models import EventLog
from discovery.models import DiscoveredProcessModel
from discovery.services import pnml_string_to_petri_net
from preprocessing.services import get_event_log_dataframe, get_default_event_log_df
from .models import ConformanceResult


def run_token_replay_conformance(
    event_log_id: int,
    discovered_model_id: int,
    source: Literal["raw", "cleaned", "default"] = "default"
) -> Dict:
    """
    Run token replay conformance checking and persist results.
    
    This is the main entry point for Group 6 conformance checking.
    NO file paths - uses DataFrames and PNML from database.
    
    Parameters
    ----------
    event_log_id : int
        The ID of the EventLog to check
    discovered_model_id : int
        The ID of the DiscoveredProcessModel to check against
    source : {"raw", "cleaned", "default"}, default="default"
        Which version of the log to use
        
    Returns
    -------
    dict
        Conformance result with structure:
        {
            "conformance_result_id": int,
            "event_log_id": int,
            "discovered_model_id": int,
            "source": str,
            "stats": {
                "total_cases": int,
                "compliant_cases": int,
                "non_compliant_cases": int,
                "compliant_percentage": float,
                "non_compliant_percentage": float
            },
            "compliant_case_ids": List[str],
            "non_compliant_case_ids": List[str]
        }
        
    Raises
    ------
    EventLog.DoesNotExist
        If event log not found
    DiscoveredProcessModel.DoesNotExist
        If model not found
    ValueError
        If DataFrame doesn't have required columns
    """
    # 1. Load the event log DataFrame
    event_log = EventLog.objects.get(id=event_log_id)
    
    if source == "default":
        df = get_default_event_log_df(event_log_id)
    else:
        df = get_event_log_dataframe(event_log_id, version=source)
    
    # 2. Load the discovered model (Petri net from PNML)
    model = DiscoveredProcessModel.objects.get(id=discovered_model_id)
    pnml_string = model.pnml_content
    
    # Convert PNML to Petri net objects
    net, initial_marking, final_marking = pnml_string_to_petri_net(pnml_string)
    
    # 3. Convert DataFrame to pm4py EventLog
    # Ensure required columns exist
    required_columns = ['case:concept:name', 'concept:name', 'time:timestamp']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"DataFrame missing required columns: {missing_cols}")
    
    # Format dataframe for pm4py
    log = pm4py.format_dataframe(
        df,
        case_id='case:concept:name',
        activity_key='concept:name',
        timestamp_key='time:timestamp'
    )
    
    # Convert to pm4py EventLog object
    event_log_pm4py = pm4py.convert_to_event_log(log)
    
    # 4. Run Token Replay
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info(f"Running token replay on {len(event_log_pm4py)} traces")
    logger.info(f"Petri net has {len(net.places)} places, {len(net.transitions)} transitions")
    
    replayed_traces = token_replay.apply(
        event_log_pm4py,
        net,
        initial_marking,
        final_marking
    )
    
    logger.info(f"Token replay completed, got {len(replayed_traces)} results")
    
    # 5. Analyze compliance for each case
    # IMPORTANT: Match traces to case IDs using the trace attributes, not array index
    compliant_case_ids = []
    non_compliant_case_ids = []
    
    for trace, trace_result in zip(event_log_pm4py, replayed_traces):
        # Get case ID from the trace attributes
        case_id = trace.attributes.get('concept:name', None)
        if case_id is None:
            # Fallback to other possible attribute names
            case_id = trace.attributes.get('case:concept:name', str(len(compliant_case_ids) + len(non_compliant_case_ids)))
        
        case_id = str(case_id)
        
        # Check if trace is fit (compliant)
        is_fit = trace_result.get('trace_is_fit', False)
        
        if is_fit:
            compliant_case_ids.append(case_id)
        else:
            non_compliant_case_ids.append(case_id)
    
    # 6. Calculate statistics
    total_cases = len(replayed_traces)
    compliant_cases = len(compliant_case_ids)
    non_compliant_cases = len(non_compliant_case_ids)
    
    compliant_percentage = (compliant_cases / total_cases * 100) if total_cases > 0 else 0
    non_compliant_percentage = (non_compliant_cases / total_cases * 100) if total_cases > 0 else 0
    
    logger.info(f"Conformance results: {compliant_cases}/{total_cases} compliant ({compliant_percentage:.1f}%)")
    logger.info(f"Sample compliant cases: {compliant_case_ids[:5]}")
    logger.info(f"Sample non-compliant cases: {non_compliant_case_ids[:5]}")
    
    # 7. Persist result to database
    conformance_result = ConformanceResult.objects.create(
        event_log=event_log,
        discovered_model=model,
        source_version=source,
        total_cases=total_cases,
        compliant_cases=compliant_cases,
        non_compliant_cases=non_compliant_cases,
        compliant_percentage=compliant_percentage,
        non_compliant_percentage=non_compliant_percentage,
        compliant_case_ids=compliant_case_ids,
        non_compliant_case_ids=non_compliant_case_ids
    )
    
    # 8. Return result
    return {
        "conformance_result_id": conformance_result.id,
        "event_log_id": event_log_id,
        "discovered_model_id": discovered_model_id,
        "source": source,
        "stats": {
            "total_cases": total_cases,
            "compliant_cases": compliant_cases,
            "non_compliant_cases": non_compliant_cases,
            "compliant_percentage": round(compliant_percentage, 2),
            "non_compliant_percentage": round(non_compliant_percentage, 2)
        },
        "compliant_case_ids": compliant_case_ids,
        "non_compliant_case_ids": non_compliant_case_ids
    }


def get_conformance_cases(
    conformance_result_id: int,
    status: Literal["compliant", "non_compliant"],
    page: int = 1,
    page_size: int = 50
) -> Dict:
    """
    Get paginated list of cases (log rows) for a conformance result.
    
    Parameters
    ----------
    conformance_result_id : int
        The ID of the ConformanceResult
    status : {"compliant", "non_compliant"}
        Which cases to retrieve
    page : int, default=1
        Page number (1-indexed)
    page_size : int, default=50
        Number of cases per page
        
    Returns
    -------
    dict
        {
            "total_cases": int,
            "page": int,
            "page_size": int,
            "total_pages": int,
            "cases": List[dict]  # case rows with all event details
        }
    """
    # Get the conformance result
    result = ConformanceResult.objects.get(id=conformance_result_id)
    
    # Get the relevant case IDs
    if status == "compliant":
        case_ids = result.compliant_case_ids
    else:
        case_ids = result.non_compliant_case_ids
    
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Getting {status} cases for conformance result {conformance_result_id}")
    logger.info(f"Total {status} case IDs: {len(case_ids)}")
    logger.info(f"Sample case IDs (first 5): {case_ids[:5]}")
    
    # Load the DataFrame
    if result.source_version == "default":
        df = get_default_event_log_df(result.event_log.id)
    else:
        df = get_event_log_dataframe(result.event_log.id, version=result.source_version)
    
    logger.info(f"DataFrame shape: {df.shape}")
    logger.info(f"DataFrame case IDs (first 5): {df['case:concept:name'].unique()[:5].tolist()}")
    logger.info(f"DataFrame case ID type: {type(df['case:concept:name'].iloc[0])}")
    logger.info(f"Stored case ID type: {type(case_ids[0]) if case_ids else 'empty'}")
    
    # Ensure case IDs are the same type as DataFrame
    df_case_type = type(df['case:concept:name'].iloc[0])
    case_ids_converted = [df_case_type(cid) for cid in case_ids]
    
    # Filter to only the relevant cases
    filtered_df = df[df['case:concept:name'].isin(case_ids_converted)]
    logger.info(f"Filtered DataFrame shape: {filtered_df.shape}")
    
    # Pagination
    total_cases = len(case_ids)
    total_pages = (total_cases + page_size - 1) // page_size
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    
    # Get paginated case IDs
    paginated_case_ids = case_ids_converted[start_idx:end_idx]
    
    logger.info(f"Paginated case IDs (page {page}): {paginated_case_ids[:5]}")
    
    # Filter DataFrame to only paginated cases
    page_df = filtered_df[filtered_df['case:concept:name'].isin(paginated_case_ids)]
    
    logger.info(f"Page DataFrame shape: {page_df.shape}")
    
    # Convert to list of dicts, handling timestamps
    cases = []
    for _, row in page_df.iterrows():
        case_dict = {}
        for col in page_df.columns:
            val = row[col]
            # Convert timestamps to ISO format strings
            if pd.api.types.is_datetime64_any_dtype(type(val)) or hasattr(val, 'isoformat'):
                case_dict[col] = val.isoformat() if pd.notna(val) else None
            # Handle NaN values
            elif pd.isna(val):
                case_dict[col] = None
            else:
                case_dict[col] = val
        cases.append(case_dict)
    
    return {
        "total_cases": total_cases,
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages,
        "cases": cases
    }
