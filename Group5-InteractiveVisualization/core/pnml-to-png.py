# from pm4py.objects.petri_net.importer import pnml as pnml_importer
from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.visualization.petri_net import visualizer as pn_visualizer


net, im, fm = pnml_importer.apply("alpha_test1_slides.pnml")
gviz = pn_visualizer.apply(net, im, fm)
pn_visualizer.save(gviz, "static/core/output/alpha_test1_slides.png")


net, im, fm = pnml_importer.apply("heuristic_test1_slides.pnml")
gviz = pn_visualizer.apply(net, im, fm)
pn_visualizer.save(gviz, "static/core/output/heuristic_test1_slides.png")
