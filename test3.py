import pm4py
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness
import pandas as pd
from pm4py.visualization.heuristics_net import visualizer as hn_vis
from pm4py.algo.evaluation.precision import algorithm as precision_algo
from pm4py.visualization.petri_net import visualizer as pn_vis
from pm4py.objects.log.util import dataframe_utils
from pm4py.objects.conversion.log import converter as log_converter
import graphviz
import os

# # Replace with your actual Graphviz bin folder
# graphviz.backend.EXECUTABLES['dot'] = r"C:\Program Files\Graphviz\bin\dot.exe"

os.environ["GRAPHVIZ_DOT"] = r"C:\Program Files\Graphviz\bin\dot.exe"
# بارگذاری داده
df = pd.read_csv(r"C:\Users\fs\Desktop\KAHANI\PM_Prj_Phase1\old dataset\ServiceTicket_Events.csv")
df["time:timestamp"] = pd.to_datetime(df["time:timestamp"], format="%d.%m.%Y %H:%M")
df = dataframe_utils.convert_timestamp_columns_in_df(df)
log = log_converter.apply(df)



heu_net = heuristics_miner.apply_heu(log, parameters={
        heuristics_miner.Variants.CLASSIC.value.Parameters.DEPENDENCY_THRESH: 0.50})

gviz_hn = hn_vis.apply(heu_net)
hn_vis.view(gviz_hn)




net, im, fm = heuristics_miner.apply(log, parameters={
        heuristics_miner.Variants.CLASSIC.value.Parameters.DEPENDENCY_THRESH: 0.50})

gviz_pn = pn_vis.apply(net, im, fm, variant=pn_vis.Variants.FREQUENCY)
pn_vis.view(gviz_pn)


fitness = replay_fitness.apply(log, net, im, fm, variant=replay_fitness.Variants.TOKEN_BASED)
print("\nFitness:", fitness)

print("\n")


precision = precision_algo.apply(log, net, im, fm, variant=precision_algo.Variants.ETCONFORMANCE_TOKEN)
print("\nPrecision:", precision)


fitness = fitness['log_fitness']
f_score = 2 * (fitness * precision) / (fitness + precision) if (fitness + precision) != 0 else 0
print("\nf_score:",f_score)