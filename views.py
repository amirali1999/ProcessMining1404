from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import EventLog
from .serializers import EventLogSerializer
from .alpha_miner_service import apply, test_module
import pandas as pd
from pm4py.objects.log.obj import EventLog as PM4PyEventLog
from pm4py.objects.log.obj import Trace, Event
from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import EventLog
import pandas as pd

from pm4py.visualization.petri_net import visualizer as pn_vis
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import EventLog
from .serializers import EventLogSerializer
import os
import graphviz
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.petri_net import utils as pn_utils
# from pm4py.algo.evaluation.replay_fitness import evaluator as fitness_evaluator
from pm4py.algo.evaluation.replay_fitness import algorithm as fitness_evaluator
# from pm4py.algo.evaluation.precision import evaluator as precision_evaluator
from pm4py.algo.evaluation.precision import algorithm as precision_evaluator
class UploadCSVView(APIView):

    def get(self, request):
        # Render HTML form
        return render(request, 'upload.html')

    def post(self, request):
        file = request.FILES.get('file')
        if not file:
            return render(request, 'upload.html', {'message': 'No file uploaded'})
        logName = file.name
        try:
            df = pd.read_csv(file)
            events = []
            for _, row in df.iterrows():
                events.append(EventLog(
                    log_name = logName,
                    case_id = row['case_id'],
                    activity = row['activity'],
                    timestamp = pd.to_datetime(row['timestamp'])
                ))
            EventLog.objects.bulk_create(events)
            return render(request, 'upload.html', {'message': 'CSV uploaded successfully'})
        except Exception as e:
            return render(request, 'upload.html', {'message': f'Error: {str(e)}'})

def queryset_to_pm4py_eventlog(queryset):
    """
    Convert Django queryset of EventLog to PM4Py EventLog
    """
    logs_dict = {}
    for event in queryset:
        if event.case_id not in logs_dict:
            logs_dict[event.case_id] = []
        logs_dict[event.case_id].append({
            'concept:name': event.activity,
            'time:timestamp': event.timestamp
        })

    pm4py_log = PM4PyEventLog()
    for case_id, events in logs_dict.items():
        trace = Trace()
        for e in sorted(events, key=lambda x: x['time:timestamp']):
            trace.append(Event(e))
        pm4py_log.append(trace)
    return pm4py_log

class AlphaMinerAPI(APIView):
    def get(self, request):
        """
        Render HTML page for selecting log_name and running Alpha Miner
        """
        log_names = EventLog.objects.values_list('log_name', flat=True).distinct()
        return render(request, 'alpha_miner.html', {'log_names': log_names})

    def post(self, request):
        """
        Run Alpha Miner on selected log_name and return PetriNet as JSON
        """
        log_name = request.data.get("log_name")
        if not log_name:
            return Response({"error": "log_name is required"}, status=400)
        
        logs = EventLog.objects.filter(log_name=log_name)
        if not logs.exists():
            return Response({"error": "No event logs found"}, status=status.HTTP_400_BAD_REQUEST)

        pm4py_log = queryset_to_pm4py_eventlog(logs)
        net, initial_marking, final_marking = apply(pm4py_log)

        ### Show net in output
        gviz_pn = pn_vis.apply(net, initial_marking, final_marking, variant=pn_vis.Variants.FREQUENCY)
        pn_vis.view(gviz_pn)
        ###
        # Convert PetriNet to JSON-friendly format
        result = {
            "transitions": [t.name for t in net.transitions],
            "places": [p.name for p in net.places]
        }

        # Print output in console for debug
        test_module(net, initial_marking, final_marking)

        return Response(result)




# Make sure Graphviz dot executable is set
os.environ["GRAPHVIZ_DOT"] = r"C:\Program Files\Graphviz\bin\dot.exe"
class HeuristicMinerAPI(APIView):
    def get(self, request):
        """
        Render HTML page for selecting log_name and running Alpha Miner
        """
        log_names = EventLog.objects.values_list('log_name', flat=True).distinct()
        return render(request, 'heuristic_miner.html', {'log_names': log_names})
    def post(self, request):
        """
        Run Alpha Miner on selected log_name and return PetriNet as JSON
        """
        log_name = request.data.get("log_name")
        if not log_name:
            return Response({"error": "log_name is required"}, status=400)
        
        logs = EventLog.objects.filter(log_name=log_name)
        if not logs.exists():
            return Response({"error": "No event logs found"}, status=status.HTTP_400_BAD_REQUEST)

        pm4py_log = queryset_to_pm4py_eventlog(logs)
        
        # Run Heuristic Miner
        heu_net = heuristics_miner.apply_heu(pm4py_log)

        # Convert Heuristic Net to Petri Net
        net, initial_marking, final_marking = heuristics_miner.apply(pm4py_log)

        # Visualize using Graphviz
        dot = graphviz.Digraph(comment=f'Heuristic Miner: {log_name}')
        for t in net.transitions:
            dot.node(t.name, t.name)
        for p in net.places:
            dot.node(str(p), '', shape='circle')
        for arc in net.arcs:
            dot.edge(str(arc.source), str(arc.target))
        # Render to static image
        graph_file = f'media/heuristic_{log_name}.png'
        dot.render(filename=graph_file, format='png', cleanup=True)
        
        ### Show net in output
        gviz_pn = pn_vis.apply(net, initial_marking, final_marking, variant=pn_vis.Variants.FREQUENCY)
        pn_vis.view(gviz_pn)
        ###

        # Calculate fitness
        fitness = fitness_evaluator.apply(pm4py_log, net, initial_marking, final_marking)

        # Calculate precision
        precision = precision_evaluator.apply(pm4py_log, net, initial_marking, final_marking)

        result = {
            "petri_net_image": graph_file + ".png",  # Full path to PNG
            "fitness": fitness,
            "precision": precision
        }

        return Response(result)

# class HeuristicMinerAPI(APIView):
#     def get(self, request):
#         """
#         Return distinct log_names for populating dropdown in HTML template
#         """
#         logs = EventLog.objects.values_list('log_name', flat=True).distinct()
#         log_names = [log for log in logs if log]  # remove empty strings
#         return Response({"log_names": log_names})
        
#     def post(self, request):
#         """
#         Run Heuristic Miner on all EventLogs with given log_name,
#         visualize Petri Net using Graphviz, and calculate fitness & precision.
#         """
#         log_name = request.data.get("log_name")
#         if not log_name:
#             return Response({"error": "log_name is required"}, status=400)

#         logs = EventLog.objects.filter(log_name=log_name)
#         if not logs.exists():
#             return Response({"error": "No event logs found"}, status=status.HTTP_400_BAD_REQUEST)

#         # Convert to PM4Py EventLog
#         pm4py_log = queryset_to_pm4py_eventlog(logs)

#         # Run Heuristic Miner
#         heu_net = heuristics_miner.apply_heu(pm4py_log)

#         # Convert Heuristic Net to Petri Net
#         net, initial_marking, final_marking = heuristics_miner.apply(pm4py_log)

#         # Visualize using Graphviz
#         dot = graphviz.Digraph(comment=f'Heuristic Miner: {log_name}')
#         for t in net.transitions:
#             dot.node(t.name, t.name)
#         for p in net.places:
#             dot.node(str(p), '', shape='circle')
#         for arc in net.arcs:
#             dot.edge(str(arc.source), str(arc.target))
#         # Render to static image
#         graph_file = f'media/heuristic_{log_name}.png'
#         dot.render(filename=graph_file, format='png', cleanup=True)

#         # Calculate fitness
#         fitness = fitness_evaluator.apply(pm4py_log, net, initial_marking, final_marking)

#         # Calculate precision
#         precision = precision_evaluator.apply(pm4py_log, net, initial_marking, final_marking)

#         result = {
#             "petri_net_image": graph_file + ".png",  # Full path to PNG
#             "fitness": fitness,
#             "precision": precision
#         }

#         return Response(result)
