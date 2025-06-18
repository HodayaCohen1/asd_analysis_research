import os
import time
from abc import ABC
import pandas as pd
from sklearn.model_selection import KFold

from flows.AbstractFlow import AbstractFlow
from ..community_detection.consts import *
from ..community_detection.utils import make_output_dir, add_dempgraphic_data,  handle_icd,handle_patients,  data_prep, get_top10_nn, create_nx_graph, get_community_louvain, \
    plot_community_lengths, add_demographic_data, get_community_label_propagation, get_community_infomap, get_community_greedy_modularity, evaluate_clusters


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

        # ---------------------- Add Demographic Data ----------------------
        if args.population_type in ('Control', 'mixed'):
            add_dempgraphic_data(df_dict, args)

        # ---------------------- Handle icd dataframe ----------------------
        handle_icd(df_dict)

        # ---------------------- Handle Patients Dataframe ----------------------
        handle_patients(df_dict, args)

        if args.cv_ind:
            kf = KFold(n_splits=5, shuffle=True, random_state=42)

            # Dataframes to hold aggregated results
            aggregated_results = []

            for fold_index, (train_index, test_index) in enumerate(kf.split(df_dict['patients'])):
                print(f'Fold {fold_index+1} started...')
                print(f'Data shape: {len(train_index)}')

                output_dir_fold = os.path.join(output_dir, f'fold_{fold_index+1}')
                os.makedirs(output_dir_fold, exist_ok=True)

                data_prep(df_dict, train_index)
                get_top10_nn(df_dict)
                G = create_nx_graph(df_dict)
                if args.community_detection_type == 'louvain':
                    get_community_louvain(df_dict, G, args, output_dir_fold)
                else:
                    raise ValueError('Invalid community detection type for cross validation')
                plot_community_lengths(df_dict, args, output_dir_fold)
                add_demographic_data(df_dict, args, output_dir_fold)

                # Evaluate clusters using the evaluation function
                evaluation_results = evaluate_clusters(df_dict['df_communities_with_demographics'], fold_index+1)
                aggregated_results.append(evaluation_results)
            pd.DataFrame(aggregated_results).to_csv(os.path.join(output_dir, f'evaluation_results.csv'), index=False)

        else:

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
            elif args.community_detection_type == 'infomap':
                get_community_infomap(df_dict, G, args, output_dir)
            elif args.community_detection_type == 'greedy_modularity':
                get_community_greedy_modularity(df_dict, G, args, output_dir)
            else:
                raise ValueError('Invalid community detection type')

            # ---------------------- Plot the Communities Length ----------------------
            plot_community_lengths(df_dict, args, output_dir)

            # ---------------------- Add Demographic Data ----------------------
            add_demographic_data(df_dict, args, output_dir)
