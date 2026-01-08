"""
Service layer for Process Discovery (Group 4 integration).

This module provides clean interfaces for running discovery algorithms
and manages the persistence of discovered models for Groups 5 & 6.

DO NOT access files directly - use preprocessing.services.get_event_log_dataframe()
"""

import io
import tempfile
import os
from typing import Literal, Dict, Any, Optional, Tuple
import pandas as pd
import pm4py
from pm4py.algo.discovery.alpha import algorithm as alpha_miner
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.objects.petri_net.obj import PetriNet, Marking

from preprocessing.services import get_event_log_dataframe
from .models import DiscoveredProcessModel


# ============================================
# PNML Serialization Helpers
# ============================================

def petri_net_to_pnml_string(
    net: PetriNet, 
    initial_marking: Marking, 
    final_marking: Marking
) -> str:
    """
    Convert a pm4py Petri net to PNML XML string (without writing to disk).
    
    Parameters
    ----------
    net : PetriNet
        The Petri net object from pm4py
    initial_marking : Marking
        The initial marking
    final_marking : Marking
        The final marking
        
    Returns
    -------
    str
        PNML XML content as string
    """
    # pm4py requires a file path, so we use a temporary file
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.pnml', delete=False, encoding='utf-8') as tmp_file:
        tmp_path = tmp_file.name
    
    try:
        # Write PNML to temporary file
        pm4py.write_pnml(net, initial_marking, final_marking, tmp_path)
        
        # Read it back as string
        with open(tmp_path, 'r', encoding='utf-8') as f:
            pnml_string = f.read()
        
        return pnml_string
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def pnml_string_to_petri_net(pnml_string: str) -> Tuple[PetriNet, Marking, Marking]:
    """
    Convert PNML XML string back to pm4py Petri net objects.
    
    Parameters
    ----------
    pnml_string : str
        PNML XML content
        
    Returns
    -------
    tuple
        (net, initial_marking, final_marking)
    """
    # pm4py requires a file path, so we use a temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pnml', delete=False, encoding='utf-8') as tmp_file:
        tmp_file.write(pnml_string)
        tmp_path = tmp_file.name
    
    try:
        # Read PNML from temporary file
        net, initial_marking, final_marking = pm4py.read_pnml(tmp_path)
        return net, initial_marking, final_marking
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def compute_petri_net_stats(net: PetriNet) -> Dict[str, int]:
    """
    Compute basic statistics about a Petri net.
    
    Parameters
    ----------
    net : PetriNet
        The Petri net object
        
    Returns
    -------
    dict
        Statistics: num_places, num_transitions, num_arcs
    """
    num_places = len(net.places)
    num_transitions = len(net.transitions)
    num_arcs = len(net.arcs)
    
    return {
        'num_places': num_places,
        'num_transitions': num_transitions,
        'num_arcs': num_arcs,
    }


# ============================================
# Discovery Algorithm Runners
# ============================================

def run_alpha_miner(
    event_log_id: int,
    source: Literal["raw", "cleaned"] = "raw",
    user_id: Optional[int] = None
) -> DiscoveredProcessModel:
    """
    Run Alpha Miner algorithm on an event log and persist the result.
    
    This is the main entry point for Group 1 to trigger Alpha Miner discovery.
    The resulting model can be used by Groups 5 (visualization) and 6 (conformance).
    
    Parameters
    ----------
    event_log_id : int
        The ID of the EventLog to discover from
    source : {"raw", "cleaned"}, default="raw"
        Which version of the log to use
    user_id : int, optional
        The ID of the user running discovery (for audit trail)
        
    Returns
    -------
    DiscoveredProcessModel
        The persisted model with PNML content
        
    Raises
    ------
    ValueError
        If event log not found or source version doesn't exist
    """
    from uploads.models import EventLog
    
    # Load the event log using preprocessing service
    df = get_event_log_dataframe(event_log_id, version=source)
    
    # Run Alpha Miner
    net, initial_marking, final_marking = alpha_miner.apply(df)
    
    # Convert to PNML string
    pnml_content = petri_net_to_pnml_string(net, initial_marking, final_marking)
    
    # Compute statistics
    stats = compute_petri_net_stats(net)
    
    # Get the EventLog instance
    event_log = EventLog.objects.get(pk=event_log_id)
    
    # Create and save the model
    model = DiscoveredProcessModel.objects.create(
        event_log=event_log,
        discovered_by_id=user_id,
        algorithm='alpha',
        source_version=source,
        pnml_content=pnml_content,
        num_places=stats['num_places'],
        num_transitions=stats['num_transitions'],
        num_arcs=stats['num_arcs'],
    )
    
    return model


def run_heuristics_miner(
    event_log_id: int,
    source: Literal["raw", "cleaned"] = "raw",
    user_id: Optional[int] = None,
    dependency_threshold: float = 0.5,
    and_threshold: float = 0.1,
    loop_two_threshold: float = 0.5
) -> DiscoveredProcessModel:
    """
    Run Heuristics Miner algorithm on an event log and persist the result.
    
    This is the main entry point for Group 1 to trigger Heuristics Miner discovery.
    The resulting model can be used by Groups 5 (visualization) and 6 (conformance).
    
    Parameters
    ----------
    event_log_id : int
        The ID of the EventLog to discover from
    source : {"raw", "cleaned"}, default="raw"
        Which version of the log to use
    user_id : int, optional
        The ID of the user running discovery
    dependency_threshold : float, default=0.5
        Threshold for dependency relation
    and_threshold : float, default=0.1
        Threshold for AND splits
    loop_two_threshold : float, default=0.5
        Threshold for length-two loops
        
    Returns
    -------
    DiscoveredProcessModel
        The persisted model with PNML content
    """
    from uploads.models import EventLog
    
    # Load the event log
    df = get_event_log_dataframe(event_log_id, version=source)
    
    # Run Heuristics Miner with parameters
    net, initial_marking, final_marking = heuristics_miner.apply(
        df,
        parameters={
            heuristics_miner.Variants.CLASSIC.value.Parameters.DEPENDENCY_THRESH: dependency_threshold,
            heuristics_miner.Variants.CLASSIC.value.Parameters.AND_MEASURE_THRESH: and_threshold,
            heuristics_miner.Variants.CLASSIC.value.Parameters.LOOP_LENGTH_TWO_THRESH: loop_two_threshold,
        }
    )
    
    # Convert to PNML
    pnml_content = petri_net_to_pnml_string(net, initial_marking, final_marking)
    
    # Compute stats
    stats = compute_petri_net_stats(net)
    
    # Get EventLog
    event_log = EventLog.objects.get(pk=event_log_id)
    
    # Create and save
    model = DiscoveredProcessModel.objects.create(
        event_log=event_log,
        discovered_by_id=user_id,
        algorithm='heuristics',
        source_version=source,
        pnml_content=pnml_content,
        num_places=stats['num_places'],
        num_transitions=stats['num_transitions'],
        num_arcs=stats['num_arcs'],
    )
    
    return model


# ============================================
# Query Helpers for Groups 5 & 6
# ============================================

def get_discovered_models(event_log_id: int) -> list[DiscoveredProcessModel]:
    """
    Get all discovered models for a given event log.
    
    Parameters
    ----------
    event_log_id : int
        The EventLog ID
        
    Returns
    -------
    list[DiscoveredProcessModel]
        All discovered models, ordered by most recent first
    """
    return list(
        DiscoveredProcessModel.objects.filter(event_log_id=event_log_id)
        .select_related('event_log', 'discovered_by')
        .order_by('-discovered_at')
    )


def get_pnml_content(model_id: int) -> str:
    """
    Get the PNML XML content for a discovered model.
    
    This is used by Groups 5 (visualization) and 6 (conformance checking)
    to reconstruct the Petri net without accessing files.
    
    Parameters
    ----------
    model_id : int
        The DiscoveredProcessModel ID
        
    Returns
    -------
    str
        PNML XML content
        
    Raises
    ------
    ValueError
        If model not found
    """
    try:
        model = DiscoveredProcessModel.objects.get(pk=model_id)
        return model.pnml_content
    except DiscoveredProcessModel.DoesNotExist:
        raise ValueError(f"Discovered model {model_id} not found")


def get_petri_net_from_model(model_id: int) -> Tuple[PetriNet, Marking, Marking]:
    """
    Reconstruct a pm4py Petri net from a stored model.
    
    This is a convenience function for Groups 5 & 6.
    
    Parameters
    ----------
    model_id : int
        The DiscoveredProcessModel ID
        
    Returns
    -------
    tuple
        (net, initial_marking, final_marking)
    """
    pnml_content = get_pnml_content(model_id)
    return pnml_string_to_petri_net(pnml_content)


# ============================================
# Visualization (Group 5 Integration)
# ============================================

def render_petrinet_image(model_id: int, format: str = 'png') -> bytes:
    """
    Render a Petri net as image (PNG or SVG) from a discovered model.
    
    Parameters
    ----------
    model_id : int
        The ID of the DiscoveredProcessModel
    format : str
        Output format: 'png' or 'svg' (default: 'png')
        
    Returns
    -------
    bytes
        Image data as bytes
        
    Raises
    ------
    DiscoveredProcessModel.DoesNotExist
        If model doesn't exist
    Exception
        If rendering fails
    """
    from pm4py.visualization.petri_net import visualizer as pn_visualizer
    import logging
    
    logger = logging.getLogger(__name__)
    
    # Get the model and its PNML content
    model = DiscoveredProcessModel.objects.get(id=model_id)
    pnml_string = model.pnml_content
    
    # Convert PNML to Petri net objects
    net, initial_marking, final_marking = pnml_string_to_petri_net(pnml_string)
    
    # Set visualization parameters
    parameters = {'format': format}
    
    # Render using pm4py
    gviz = pn_visualizer.apply(net, initial_marking, final_marking, parameters=parameters)
    
    # Create a unique temporary file without extension
    tmp_fd, tmp_path = tempfile.mkstemp(suffix='')
    os.close(tmp_fd)
    
    try:
        # Save visualization (pm4py will add the extension)
        logger.info(f"Saving {format} visualization to {tmp_path}")
        pn_visualizer.save(gviz, tmp_path)
        
        # Check all possible file paths
        possible_paths = [
            tmp_path,
            f"{tmp_path}.{format}",
            f"{tmp_path}.{format}.{format}",
        ]
        
        image_bytes = None
        found_path = None
        
        for path in possible_paths:
            if os.path.exists(path):
                found_path = path
                logger.info(f"Found rendered file at: {path}")
                
                if format == 'png':
                    with open(path, 'rb') as f:
                        image_bytes = f.read()
                else:  # svg
                    with open(path, 'r', encoding='utf-8') as f:
                        image_bytes = f.read().encode('utf-8')
                
                # Clean up this file
                try:
                    os.remove(path)
                except Exception as e:
                    logger.warning(f"Could not remove temp file {path}: {e}")
                
                break
        
        if image_bytes is None:
            # List all files in temp directory for debugging
            import glob
            temp_dir = os.path.dirname(tmp_path)
            temp_name = os.path.basename(tmp_path)
            similar_files = glob.glob(os.path.join(temp_dir, f"{temp_name}*"))
            error_msg = f"Failed to find {format.upper()} file. Checked: {possible_paths}. Similar files: {similar_files}"
            logger.error(error_msg)
            raise Exception(error_msg)
        
        return image_bytes
        
    except Exception as e:
        logger.error(f"Error rendering Petri net: {str(e)}")
        raise
    finally:
        # Clean up any remaining temporary files
        for path in possible_paths:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except:
                    pass


def render_petrinet_png_from_model(model_id: int) -> bytes:
    """
    Render a Petri net as PNG image from a discovered model.
    Backward compatibility wrapper.
    """
    return render_petrinet_image(model_id, format='png')


def render_petrinet_svg_from_model(model_id: int) -> bytes:
    """
    Render a Petri net as SVG image from a discovered model.
    """
    return render_petrinet_image(model_id, format='svg')
