from enum import Enum
import time
import pandas as pd
from itertools import product
from pm4py import util as pm_util
from pm4py.algo.discovery.alpha.data_structures import (
    alpha_classic_abstraction,
)
from pm4py.algo.discovery.alpha.utils import endpoints
from pm4py.objects.dfg.utils import dfg_utils
from pm4py.algo.discovery.dfg.variants import native as dfg_inst
from pm4py.objects.petri_net.utils.petri_utils import add_arc_from_to
from pm4py.util import exec_utils
from pm4py.util import constants
from enum import Enum
from typing import Optional, Dict, Any, Union, Tuple
from pm4py.objects.log.obj import EventLog
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.algo.discovery.alpha import algorithm as alpha_miner

# ----------------------- Parameters Enum -----------------------
class Parameters(Enum):
    ACTIVITY_KEY = constants.PARAMETER_CONSTANT_ACTIVITY_KEY
    START_TIMESTAMP_KEY = constants.PARAMETER_CONSTANT_START_TIMESTAMP_KEY
    TIMESTAMP_KEY = constants.PARAMETER_CONSTANT_TIMESTAMP_KEY
    CASE_ID_KEY = constants.PARAMETER_CONSTANT_CASEID_KEY

def read_file(path):#"ServiceTicket_Events.csv"
    log = pd.read_csv(path)
    return log

# ----------------------- Helper Functions -----------------------
def __add_source(net, start_activities, label_transition_dict):
    source = PetriNet.Place("start")
    net.places.add(source)
    for s in start_activities:
        add_arc_from_to(source, label_transition_dict[s], net)
    return source


def __add_sink(net, end_activities, label_transition_dict):
    end = PetriNet.Place("end")
    net.places.add(end)
    for e in end_activities:
        add_arc_from_to(label_transition_dict[e], end, net)
    return end


def __initial_filter(parallel_relation, pair):
    if (pair[0], pair[0]) in parallel_relation or (
        pair[1],
        pair[1],
    ) in parallel_relation:
        return False
    return True


def __pair_maximizer(alpha_pairs, pair):
    for alt in alpha_pairs:
        if (
            pair != alt
            and pair[0].issubset(alt[0])
            and pair[1].issubset(alt[1])
        ):
            return False
    return True


def __check_is_unrelated(
    parallel_relation, causal_relation, item_set_1, item_set_2
):
    S = set(product(item_set_1, item_set_2)).union(
        set(product(item_set_2, item_set_1))
    )
    for pair in S:
        if pair in parallel_relation or pair in causal_relation:
            return True
    return False


def __check_all_causal(causal_relation, item_set_1, item_set_2):
    S = set(product(item_set_1, item_set_2))
    for pair in S:
        if pair not in causal_relation:
            return False
    return True


#helper function when we have dfg as input
def apply_dfg(
    dfg: Dict[Tuple[str, str], int],
    parameters: Optional[Dict[Union[str, Parameters], Any]] = None,
) -> Tuple[PetriNet, Marking, Marking]:
    return apply_alpha_miner(dfg, None, None, parameters=parameters)

# ----------------------- Core Alpha Miner -----------------------

#get eventlog object and prepare dfg, start activities, end activities
def apply(log: EventLog, parameters: Optional[Dict[Union[str, Parameters], Any]] = None) -> Tuple[PetriNet, Marking, Marking]:
    """
    Apply Alpha Miner to a PM4Py EventLog and return PetriNet + Markings
    """
    if parameters is None:
        parameters = {}

    activity_key = exec_utils.get_param_value(
        Parameters.ACTIVITY_KEY,
        parameters,
        "concept:name"
    )

    # compute Direct Follow Graph (DFG)
    dfg = {k: v for k, v in dfg_inst.apply(log, parameters=parameters).items() if v > 0}
    start_activities = endpoints.derive_start_activities_from_log(log, activity_key)
    end_activities = endpoints.derive_end_activities_from_log(log, activity_key)

    return apply_alpha_miner(dfg, start_activities, end_activities, parameters)
#core function that apply alpha miner algorithm 

def apply_alpha_miner(
    dfg: Dict[str, int],
    start_activities: Union[None, Dict[str, int]],
    end_activities: Union[None, Dict[str, int]],
    parameters: Optional[Dict[Union[str, Parameters], Any]] = None,
) -> Tuple[PetriNet, Marking, Marking]:
    if parameters is None:
        parameters = {}

    
    #get name of activity column 
    # activity_key = exec_utils.get_param_value(
    #     Parameters.ACTIVITY_KEY,
    #     parameters,
    #     pm_util.xes_constants.DEFAULT_NAME_KEY,
    # )

    if start_activities is None:
        start_activities = dfg_utils.infer_start_activities(dfg)

    if end_activities is None:
        end_activities = dfg_utils.infer_end_activities(dfg)

    #create list of all activity labels (T(L))
    labels = set()
    for el in dfg:
        labels.add(el[0])
        labels.add(el[1])
    for a in start_activities:
        labels.add(a)
    for a in end_activities:
        labels.add(a)
    labels = list(labels)

    #calculate footprint
    # alpha_abstraction = alpha_classic_abstraction.ClassicAlphaAbstraction(
    #     start_activities, end_activities, dfg, activity_key=activity_key
    # )
    alpha_abstraction = alpha_classic_abstraction.ClassicAlphaAbstraction(
        start_activities, end_activities, dfg
    )
    #Generate alpha pairs (X(L)) 
    pairs = list(
        map(
            lambda p: ({p[0]}, {p[1]}),
            filter(
                lambda p: __initial_filter(
                    alpha_abstraction.parallel_relation, p
                ),
                alpha_abstraction.causal_relation,
            ),
        )
    )
    for i in range(0, len(pairs)):
        t1 = pairs[i]
        for j in range(i, len(pairs)):
            t2 = pairs[j]
            if t1 != t2:
                if t1[0].issubset(t2[0]) or t1[1].issubset(t2[1]):
                    if not (
                        __check_is_unrelated(
                            alpha_abstraction.parallel_relation,
                            alpha_abstraction.causal_relation,
                            t1[0],
                            t2[0],
                        )
                        or __check_is_unrelated(
                            alpha_abstraction.parallel_relation,
                            alpha_abstraction.causal_relation,
                            t1[1],
                            t2[1],
                        )
                    ):
                        new_alpha_pair = (t1[0] | t2[0], t1[1] | t2[1])
                        if new_alpha_pair not in pairs:
                            if __check_all_causal(
                                alpha_abstraction.causal_relation,
                                new_alpha_pair[0],
                                new_alpha_pair[1],
                            ):
                                pairs.append((t1[0] | t2[0], t1[1] | t2[1]))
    #apply maximization (Y(L))
    internal_places = filter(lambda p: __pair_maximizer(pairs, p), pairs)
    #create Petri Net Object (P(L),T(L),F(L))
    net = PetriNet("alpha_classic_net_" + str(time.time()))
    label_transition_dict = {}
        
    for i in range(0, len(labels)):
        label_transition_dict[labels[i]] = PetriNet.Transition(
            labels[i], labels[i]
        )
        net.transitions.add(label_transition_dict[labels[i]])

    src = __add_source(
        net, alpha_abstraction.start_activities, label_transition_dict
    )
    sink = __add_sink(
        net, alpha_abstraction.end_activities, label_transition_dict
    )

    for pair in internal_places:
        place = PetriNet.Place(str(pair))
        net.places.add(place)
        for in_arc in pair[0]:
            add_arc_from_to(label_transition_dict[in_arc], place, net)
        for out_arc in pair[1]:
            add_arc_from_to(place, label_transition_dict[out_arc], net)
    return net, Marking({src: 1}), Marking({sink: 1})

# ----------------------- Test module -----------------------
def test_module(net, initial_marking, final_marking):
    # event_log_path = "ServiceTicket_Events.csv"
    # net, initial_marking, final_marking = alpha_miner.apply(log)
    
    print("Transitions:")
    for t in net.transitions:
        print(t.name)
    
    print("\nPlaces:")
    for p in net.places:
        print(p.name)
    
    print("\nInitial Marking:")
    for place, tokens in initial_marking.items():
        print(place.name, ":", tokens)
    
    print("\nFinal Marking:")
    for place, tokens in final_marking.items():
        print(place.name, ":", tokens)
    # gviz = pn_visualizer.apply(net, initial_marking, final_marking)
    # pn_visualizer.view(gviz)  #این پنجره‌ای باز می‌کند که شبکه را نشان می‌د
