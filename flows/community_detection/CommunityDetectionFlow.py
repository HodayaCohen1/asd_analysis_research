import os
import time
from abc import ABC

import pandas as pd

from flows.AbstractFlow import AbstractFlow
from ..community_detection.consts import *
from ..community_detection.utils import make_output_dir, handle_icd,handle_patients,  data_prep, get_top10_nn, create_nx_graph, get_community_louvain, \
    plot_community_lengths, add_demographic_data, get_community_label_propagation


class CommunityDetectionFlow(AbstractFlow, ABC):
    def __init__(self, type: str):
        super().__init__(type)

    def Init(self):
        super().Init()

    def run_flow(self, args=None):
        print(f'Starting {self.type} flow...')
        input_files = {'patients':args.patients_file, 'icd':args.icd_file}

        # ---------------------- Load CSV from local into Pandas DataFrame ----------------------
        try:
            df_dict = {}
            for name, file_path in input_files.items():
                df_dict[name] = pd.read_csv(file_path)
                print(f'{name} loaded successfully, shape: {df_dict[name].shape}')
        except FileNotFoundError as e:
            print('files do not exist.')
            raise e

        # ---------------------- Make Output Directory ----------------------
        output_dir = make_output_dir(args)

        # ---------------------- Handle icd dataframe ----------------------
        handle_icd(df_dict)

        # ---------------------- Handle Patients Dataframe ----------------------
        handle_patients(df_dict, args)

        # ---------------------- Data Preparation ----------------------
        data_prep(df_dict)

        # ---------------------- Get top 10 nearest neighbors ----------------------
        get_top10_nn(df_dict)

        # ---------------------- Create a graph ----------------------
        G = create_nx_graph(df_dict)

        # ---------------------- Perform Community Detection ----------------------
        if args.community_detection_type == 'louvain':
            get_community_louvain(df_dict, G, args, output_dir)
        elif args.community_detection_type == 'label_propagation':
            get_community_label_propagation(df_dict, G, args, output_dir)


        # ---------------------- Plot the Communities Length ----------------------
        plot_community_lengths(df_dict, args, output_dir)

        # ---------------------- Add Demographic Data ----------------------
        add_demographic_data(df_dict, args, output_dir)
