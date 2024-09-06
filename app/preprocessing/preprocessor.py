from .merging_data import DataMerger  # importing the DataMerger class
from .eda import EDAProcessor         # importing the EDAProcessor class

class MyPreProcessor:
    def __init__(self, data):
        self.data = data

    def preprocess(self):
        merger = DataMerger(self.data)
        merged_data = merger.merge()

        eda = EDAProcessor(merged_data)
        eda_results = eda.perform_eda()

        return eda_results
