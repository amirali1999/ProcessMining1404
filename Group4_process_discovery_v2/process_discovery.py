import pandas as pd
import os
import pm4py
from pm4py.algo.discovery.alpha import algorithm as alpha_miner
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.visualization.petri_net import visualizer as pn_vis


class ProcessDiscovery:
    def __init__(self, df=None, path=None):
        self.inOut_path = path
        self.df = df
        self.event_log = None
        self.petrinet_alpha = None
        self.petrinet_heu = None

    # ------------------------------
    # Load CSV or Parquet
    # ------------------------------
    def read_file(self):
        if self.inOut_path is None:
            raise ValueError("No path provided")

        if self.inOut_path.endswith(".csv"):
            self.df = pd.read_csv(self.inOut_path)
        elif self.inOut_path.endswith(".parquet"):
            self.df = pd.read_parquet(self.inOut_path)
        else:
            raise ValueError("Unsupported file format")

        return self.df

    # ------------------------------
    # Alpha Miner
    # ------------------------------
    ## Provide a REAL folder file path here for store output
    def alpha_miner_service(self, output_dir=r".\data"):
        if self.df is None:
            self.df = self.read_file()
        
        net, im, fm = alpha_miner.apply(self.df)

        # Save PNML
        filename = os.path.splitext(os.path.basename(self.inOut_path))[0]
        pnml_path = os.path.join(output_dir, f"alpha_{filename}.pnml")

        pm4py.write_pnml(net, im, fm, pnml_path)

        self.petrinet_alpha = (net, im, fm)
        return self.petrinet_alpha

    # ------------------------------
    # Heuristics Miner
    # ------------------------------
    ## Provide a REAL folder file path here for store output        
    def heuristic_miner_service(self, output_dir=r".\data"):
        if self.df is None:
            self.df = self.read_file()
     
        net, im, fm = heuristics_miner.apply(self.df)

        filename = os.path.splitext(os.path.basename(self.inOut_path))[0]
        pnml_path = os.path.join(output_dir, f"heuristic_{filename}.pnml")

        pm4py.write_pnml(net, im, fm, pnml_path)

        self.petrinet_heu = (net, im, fm)
        return self.petrinet_heu

    # ------------------------------
    # Visualization for test
    # ------------------------------
    def visualize(self, net, im, fm):
        os.environ["GRAPHVIZ_DOT"] = r"C:\Program Files\Graphviz\bin\dot.exe"
        gviz = pn_vis.apply(net, im, fm)
        pn_vis.view(gviz)


# ============================================
# TEST EXAMPLE
# ============================================

def test_class():
    # Provide a REAL CSV file path here
    csv_path = r".\data\test1_slides.csv"

    p_discovery = ProcessDiscovery(path=csv_path)

    pn_alpha = p_discovery.alpha_miner_service()
    p_discovery.visualize(*pn_alpha)

    pn_heu = p_discovery.heuristic_miner_service()
    p_discovery.visualize(*pn_heu)


# Run the test
test_class()
