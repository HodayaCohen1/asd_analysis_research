import json
import os
import time
import zipfile
from datetime import datetime, timedelta
from typing import Dict, List
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import infomap
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import xlsxwriter
import statsmodels.api as sm

from ..community_detection.consts import DESIRED_RACE_UNIQUE, DESIRED_MARITAL_STATUS_UNIQUE, DESIRED_GENDER_UNIQUE
from helpers.consts import *
from helpers.exceptions import FieldMissingException, ValueMissingException
from statsmodels.stats.outliers_influence import variance_inflation_factor


def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time of {func.__name__}: {execution_time:.6f} seconds")
        return result

    return wrapper


def make_output_dir(args):
    """
    Creates the output directory
    :param args: Arguments
    """
    output_dir = OUTPUT_DIR.format(args.type, args.population_type)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f'Output directory: {output_dir}')
    return output_dir

def add_dempgraphic_data(df_dict: dict, args):
    """
    Adds demographic data to the patients dataframe
    :param df_dict: Dictionary of datasets
    :param args: Arguments
    """
    print('-' * NUM_OF_HYPHENS)
    print(f'Adding demographic data to patients dataframe...')
    df_patients = df_dict['patients'].fillna(0)
    print('df_patients shape ', df_patients.shape)

    # load the full patients dataframe
    hf_d_patients = pd.read_csv('/Users/hodaya/Projects/Python/Thesis/Data Preparation/HF_D_PATIENT.csv')
    P_SK_gender_race_ms = df_patients[['PATIENT_SK']].merge(hf_d_patients[['PATIENT_SK', 'RACE', 'MARITAL_STATUS', 'GENDER']].drop_duplicates(), on='PATIENT_SK', how='left')
    print('P_SK_gender_race_ms shape ', P_SK_gender_race_ms.shape)

    # Initialize an empty DataFrame to store the common values
    P_SK_common_values = pd.DataFrame()

    # List of columns we're interested in
    cols = ['RACE', 'MARITAL_STATUS', 'GENDER']

    # Loop over the columns and calculate the mode
    for col in cols:
        P_SK_common_values[col] = P_SK_gender_race_ms.groupby('PATIENT_SK')[col].apply(lambda x: x.mode().values[0] if not x.mode().empty else np.nan)

    # Reset the index
    P_SK_common_values = P_SK_common_values.reset_index()
    P_SK_common_values['RACE'] = P_SK_common_values['RACE'].apply(lambda a: a if a in(DESIRED_RACE_UNIQUE) else 'Unknown')
    P_SK_common_values['MARITAL_STATUS'] = P_SK_common_values['MARITAL_STATUS'].apply(lambda a: a if a in(DESIRED_MARITAL_STATUS_UNIQUE) else 'Unknown')
    P_SK_common_values['GENDER'] = P_SK_common_values['GENDER'].apply(lambda a: a if a in(DESIRED_GENDER_UNIQUE) else 'Unknown')
    print('P_SK_common_values shape ', P_SK_common_values.shape)

    # Merge the common values with the original DataFrame
    df_patients = df_patients.merge(P_SK_common_values, on='PATIENT_SK', how='left')

    # move the three additional columns to the first position
    indx=1
    cols_name = ['RACE','GENDER', 'MARITAL_STATUS']
    for col in cols_name:
        first_col = df_patients.pop(col)
        df_patients.insert(indx, col, first_col)
        indx+=1

    df_dict['patients'] = df_patients
    print(f'Finished adding demographic data to patients dataframe!')


def create_icd_dict(df_dict: dict):
    """
    Updates the icd dataframe
    :param df_dict: Dictionary of datasets
    """
    print('-' * NUM_OF_HYPHENS)
    print(f'Handling icd dataframe...')
    df_icd = df_dict['icd']
    # df_icd['Phenotype'].fillna('0', inplace=True)
    # df_icd['Phenotype'] = df_icd['Phenotype'].str.lower()
    # df_dict['icd'] = df_icd
    phenotypes_names = df_icd[['PheCode','Phenotype']].astype(str).drop_duplicates()
    phenotypes_names = phenotypes_names.groupby('PheCode')['Phenotype'].min().reset_index()
    phe_codes_dict = dict(zip(phenotypes_names['PheCode'], phenotypes_names['Phenotype']))
    print(f'Finished creating icd dictionary!')
    return phe_codes_dict

def create_writer(writer_path, writer_name):
    # Create an Excel writer object
    writer = pd.ExcelWriter(f'{writer_path}/{writer_name}.xlsx', engine='xlsxwriter')
    return writer

def get_frequencies(df, cluster_data, cluster):
    """
    Calculates the frequency of each phenotype in the cluster
    :param df:
    :param cluster_data:
    :param cluster:
    :return:
    """
    # print('cluster', cluster)
    phenotypes_cols_indx = list(range(4, len(df.columns)-1))
    # print('columns to calc frequency', cluster_data.iloc[:,phenotypes_cols_indx].columns)

    # Get the binary phenotypes for the current cluster
    phenotypes = cluster_data.iloc[:,phenotypes_cols_indx]
    # print('phenotypes cols',phenotypes.columns )

    # Calculate the frequency of each phenotype in the cluster
    phenotype_frequencies_cluster = phenotypes.mean()

    # Calculate the frequency of each phenotype in the entire population
    phenotype_frequencies_population = df.iloc[:, phenotypes_cols_indx].mean()

    # Calculate the frequency of each phenotype in the entire population without the current cluster
    phenotype_frequencies_population_without_cluster = df[df['cluster'] != cluster].iloc[:, phenotypes_cols_indx].mean()

    # Calculate the Fold Change
    fold_change = phenotype_frequencies_cluster / phenotype_frequencies_population_without_cluster

    # Get the top 20 phenotypes based on frequency within the cluster
    top_20_frequency_phenotypes_cluster = phenotype_frequencies_cluster.nlargest(20)

    # Calculate TF-IDF for each phenotype
    tf_idf = {}
    total_clusters = df['cluster'].nunique()
    for phenotype in phenotypes.columns:
        # Calculate the number of clusters where the phenotype appears at least once
        clusters_with_phenotype = (df[df[phenotype].astype(int) > 0]['cluster'].nunique())
        # print('phenotype', phenotype)
        # print('clusters_with_phenotype', clusters_with_phenotype)
        # print('total_clusters', total_clusters)
        # If the phenotype doesn't appear in any cluster, we set the IDF to 0
        if clusters_with_phenotype == 0:
            idf = 0
        else:
            idf = np.log(total_clusters / clusters_with_phenotype)

        # The TF-IDF is the product of the phenotype's frequency in the cluster (TF) and IDF
        tf_idf[phenotype] = phenotype_frequencies_cluster[phenotype] * idf
        # print('phenotype_frequencies_cluster[phenotype]', phenotype_frequencies_cluster[phenotype])
    return [phenotypes, phenotype_frequencies_cluster, phenotype_frequencies_population, phenotype_frequencies_population_without_cluster, fold_change, top_20_frequency_phenotypes_cluster, tf_idf]

def freq_xl_append(results_df, phenotype, frequency_cluster, frequency_population, frequency_population_without_cluster, fold_change, phenotype_new, tf_idf):
    # Create a DataFrame for the new row
    new_row = pd.DataFrame({
        'Phecode': [phenotype],
        'Frequency in Cluster': [frequency_cluster],
        'Frequency in Entire Population': [frequency_population],
        'Frequency in Population without Cluster': [frequency_population_without_cluster],
        'Fold Change': [fold_change],
        'Phenotype (new)': [phenotype_new],
        'TF-IDF': [tf_idf]
    })

    # Concatenate the new row with the existing DataFrame
    results_df = pd.concat([results_df, new_row], ignore_index=True)
    return results_df

def extract_xl(df, writer, phe_codes_dict):
    print('-' * NUM_OF_HYPHENS)
    print(f'Calculating frequencies and saving to Excel...')
    label_phenotypes_dict = {}  # create an empty dictionary to store the phenotype indexes for each label
    for cluster in sorted(df['cluster'].unique()):
        print(f'Cluster {cluster} started...')
        cluster_data = df[df['cluster'] == cluster]  # Filter data for the current cluster

        freq_res = get_frequencies(df, cluster_data, cluster)
        phenotypes = freq_res[0]
        phenotype_frequencies_cluster = freq_res[1]
        phenotype_frequencies_population = freq_res[2]
        phenotype_frequencies_population_without_cluster = freq_res[3]
        phenotype_fold_change = freq_res[4]
        top_20_frequency_phenotypes_cluster = freq_res[5]
        tf_idf_cluster = freq_res[6]

        # Create an empty DataFrame to store the results
        freq_df = pd.DataFrame(columns=['Phecode', 'Frequency in Cluster', 'Frequency in Entire Population', 'Frequency in Population without Cluster', 'Fold Change', 'Phenotype (new)', 'TF-IDF'])

        # Filter the top phenotypes by fold change and iterate over each top phenotype
        for phenotype in top_20_frequency_phenotypes_cluster.index:

            # Extract frequencies in the cluster, entire population, and population without the cluster
            frequency_cluster = phenotype_frequencies_cluster[phenotype]
            frequency_population = phenotype_frequencies_population[phenotype]
            frequency_population_without_cluster = phenotype_frequencies_population_without_cluster[phenotype]
            fold_change = phenotype_fold_change[phenotype]
            # phe_def = phe_codes_dictionary[phenotype]
            phenotype_new = phe_codes_dict[phenotype]
            tf_idf = tf_idf_cluster[phenotype]
            # Append the results to the DataFrame
            freq_df = freq_xl_append(freq_df, phenotype, frequency_cluster, frequency_population, frequency_population_without_cluster, fold_change, phenotype_new, tf_idf)

        freq_df = freq_df[freq_df['Fold Change'] >= 1.2]
        # freq_df = freq_df[freq_df['Frequency in Cluster'] >= 0.01]

        # Save the results to a sheet in the Excel file for the current cluster
        freq_df.to_excel(writer, sheet_name=f'Cluster {cluster}', index=False)
        label_phenotypes_dict[cluster] = list(freq_df['Phecode'])

    sizes = df.groupby(['cluster']).size()
    sizes = sizes.sort_values(ascending=False)
    sizes_df = pd.DataFrame({'cluster': sizes.index, 'size': sizes.values})
    sizes_df.to_excel(writer, sheet_name='sizes', index=False)
    # Save the Excel file
    writer.close()
    return label_phenotypes_dict

def get_X(ASD_patient_PheCode_full_unstack):
    X = pd.DataFrame()
    X['RACE'] = ASD_patient_PheCode_full_unstack['RACE']
    # X['age_group'] = ASD_patient_PheCode_full_unstack['age_group']
    X['GENDER'] = ASD_patient_PheCode_full_unstack['GENDER']

    # convert all columns to categorical data type
    for col in X.columns:
        X[col] = X[col].astype('category')

    # set up the logistic regression model
    X = pd.get_dummies(X, drop_first=True) # convert categorical variables to indicator variables
    X = X.astype(int)
    return X

def perform_phewas(ASD_patient_PheCode_full_unstack, phewas_dir, phewas_name, label_phenotypes_dict):
    # Create a new Excel file
    X = get_X(ASD_patient_PheCode_full_unstack)
    # X = pd.DataFrame([])
    y = ASD_patient_PheCode_full_unstack['cluster']
    y = np.array(y)

    # Check for multicollinearity
    def calculate_vif(X):
        X_vif = sm.add_constant(X)
        vif_data = pd.DataFrame()
        vif_data["feature"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i + 1) for i in range(X.shape[1])]
        return vif_data

    # Assuming X is a DataFrame of binary predictors
    vif = calculate_vif(X)
    print(vif)

    with pd.ExcelWriter(f'{phewas_dir}/{phewas_name}.xlsx') as writer:
        for label, phenotypes in label_phenotypes_dict.items():
            if label == '-1':
                continue
            labeled_y = np.where(y == label, 1, 0)
            results = [] # store the results for this label
            for phenotype in phenotypes:
                X_reg = X.copy()
                X_reg['Phenotype'] = ASD_patient_PheCode_full_unstack[phenotype]
                # print('X_reg', X_reg)

                # # Adding a constant term for the intercept - for better odds ratio - it changes the results dramatically
                # X_reg = sm.add_constant(X_reg)

                # fit the logistic regression model
                logreg = sm.Logit(labeled_y, X_reg).fit(method='bfgs', maxiter=1000)

                # get the odds ratio, p-value, and confidence interval for the second column
                coef = logreg.params[-1] # coefficient
                odds_ratio = np.exp(coef) # odds ratio
                pvalue = logreg.pvalues[-1] # p-value
                conf_int_high = np.exp(logreg.conf_int()[1][-1]) # confidence interval
                conf_int_low = np.exp(logreg.conf_int()[0][-1]) # confidence interval
                # print('params[11]', logreg.params[11])
                # print('params', logreg.params)
                # print('pvalues', logreg.pvalues)

                # store the results in a DataFrame
                results.append({'Phecode': phenotype, 'odds_ratio': odds_ratio, 'pvalue': pvalue, 'conf_int_low': conf_int_low, 'conf_int_high': conf_int_high})

            # create a new sheet for this label and write the results
            if len(results) == 0:
                df = pd.DataFrame(columns=['Phecode', 'odds_ratio', 'pvalue', 'conf_int_low', 'conf_int_high'])
            else:
                df = pd.DataFrame(results)
            # df.set_index('Phecode', inplace=True)
            sheet_name = f'Cluster {label}'
            df.to_excel(writer, sheet_name=sheet_name, index=False)

            # print the results for this label
            print('='*30, f'Cluster {label}', '='*30)
            print(df)

def merge_files(phewas_dir, phewas_name, writer_path, writer_name, merge_name):
    excel_file1 = pd.ExcelFile(f'{phewas_dir}/{phewas_name}.xlsx')
    excel_file2 = pd.ExcelFile(f'{writer_path}/{writer_name}.xlsx')

    sh = excel_file2.sheet_names
    # sh = list(set(excel_file2.sheet_names) - set(['sizes']))

    with pd.ExcelWriter(f'{phewas_dir}/{merge_name}.xlsx') as writer:
        for i in range(len(excel_file1.sheet_names)):
            print(f'sheet1={excel_file1.sheet_names[i]}, sheet2={sh[i]}')
            df1 = excel_file1.parse(excel_file1.sheet_names[i])
            df2 = excel_file2.parse(sh[i])

            # Define the desired column order
            column_order = ['Phecode','Frequency in Cluster', 'odds_ratio', 'pvalue', 'conf_int_low', 'conf_int_high', 'Frequency in Entire Population', 'Frequency in Population without Cluster', 'Fold Change', 'Phenotype (new)', 'TF-IDF']

            # Merge the dataframes based on the key columns
            if df1['Phecode'].empty or df2['Phecode'].empty:
                merged_df = pd.DataFrame(columns=column_order)  # Creating an empty DataFrame when both 'Phecode' columns are empty
            else:
                merged_df = pd.merge(df1, df2, left_on='Phecode', right_on='Phecode')

            # Rearrange the columns in merged_df according to the desired order
            merged_df = merged_df[column_order]

            # merged_df.drop(columns=['phenotype'], inplace=True)
            # merged_df = merged_df[['Phecode'] + [col for col in merged_df.columns if col != 'Phecode']]

            # Save the merged dataframe to a new sheet
            merged_df.to_excel(writer, sheet_name=sh[i], index=False)
    return merged_df


def sizes_file_creation(df, phewas_dir, subdir_name):
    # Group by 'cluster' and compute the aggregations
    cluster_summary = df.groupby('cluster').agg(
        total_patients=('PATIENT_SK', 'size'),
        ASD_count=('ASD_ind', lambda x: (x == 1).sum()),
        control_count=('ASD_ind', lambda x: (x == 0).sum())
    ).reset_index()

    # Calculate percentage of ASD and control counts
    cluster_summary['ASD_percentage'] = (cluster_summary['ASD_count'] / cluster_summary['total_patients'])
    cluster_summary['control_percentage'] = (cluster_summary['control_count'] / cluster_summary['total_patients'])

    # Write to CSV
    sizes_file_name = 'clusters_sizes_by_type'
    if subdir_name:
        sizes_file_name = f'{sizes_file_name}_{subdir_name}'
    else:
        sizes_file_name = f'{sizes_file_name}'
    cluster_summary.to_csv(f'{phewas_dir}/{sizes_file_name}.csv', index=False)
    print('sizes file was created')
    return

def perform_phewas_analysis(df_dict, writer_path, writer_name, phewas_dir="", phewas_name="", merge_name="", phe_codes_dict={}, subdir_name="", phewas_flag=False):
    df = df_dict['patients'].copy()
    sizes_file_creation(df, phewas_dir, subdir_name)
    df.drop(['MARITAL_STATUS', 'ASD_ind'], inplace=True, axis=1)
    # column_indices = list(range(1790)) + [-1]
    # df = df.iloc[:, column_indices]
    print('column df', df.columns)
    rows_to_drop = len(df[df['cluster'].isna()])
    print(f'{rows_to_drop} rows were dropped')
    df = df[df['cluster'].isna()==False]
    writer = create_writer(writer_path, writer_name)
    label_phenotypes_dict = extract_xl(df, writer, phe_codes_dict)
    if phewas_flag:
        perform_phewas(df, phewas_dir, phewas_name, label_phenotypes_dict)
        phewas_df = merge_files(phewas_dir, phewas_name, writer_path, writer_name, merge_name)
    return phewas_df


def get_bonferroni_threshold(output_dir, phewas_name):
    xls = pd.ExcelFile(f'{output_dir}/{phewas_name}.xlsx')

    # get all sheet names
    sheet_names = xls.sheet_names
    print(f'num of sheets: {len(sheet_names)}')
    alpha = 0.05  # Set the desired alpha level here

    # Store significant phecodes for each cluster
    significant_phecodes = {}
    dfs={}
    num_tests = 0
    # Iterate over each sheet (i.e., cluster)
    for sheet in sheet_names:
        # Read the sheet into a DataFrame
        df = pd.read_excel(xls, sheet_name=sheet)
        dfs[sheet] = df
        # Perform Bonferroni correction
        num_tests = num_tests + len(df)
    print('num_tests', num_tests)
    bonferroni_alpha = alpha / num_tests
    print('bonferroni_alpha', bonferroni_alpha)
    return bonferroni_alpha, dfs

def filter_df_by_pvalue(output_dir, dfs, bonferroni_alpha, BONFERRONI_FILE_NAME):
    filtered_dfs = {}
    i=0
    asd_clusters = []
    with pd.ExcelWriter(f'{output_dir}/{BONFERRONI_FILE_NAME}.xlsx') as writer:
        for sheet, df in dfs.items():
            i+=1
            filtered_df = df.loc[df['pvalue'] < bonferroni_alpha]
            filtered_dfs[sheet] = filtered_df
            type_val = 'ASD' if filtered_df['Phecode'].isin([313.3]).any() else 'Control'

            # Write the filtered dataframe to a sheet in the new Excel file
            filtered_df.to_excel(writer, sheet_name=sheet, index=False)

            if type_val == 'ASD':
                asd_clusters.append(sheet)
            print(f"{sheet} type: {type_val}")
            print(f"{sheet}: num of clusters originally: {len(df)}, num of clusters after filter p-value: {len(filtered_dfs[sheet])}")
            # Print significant phecodes for each cluster
            print(f"{sheet} significant phecodes: {filtered_df['Phecode'].tolist()}")
            print('*'*50)
    print('asd_clusters', asd_clusters)


def handle_patients(df_dict: dict, args):
    """
    Updates the patients dataframe
    :param df_dict: Dictionary of datasets
    """
    print('-' * NUM_OF_HYPHENS)
    print(f'Handling patients dataframe...')
    df_patients = df_dict['patients']
    df_dict['patients_full'] = df_dict['patients'].copy()
    # Removing phecodes_F columns
    if not args.population_type == 'mixed':
        df_columns = list(df_patients.columns)[:1792]
        df_patients = df_patients[df_columns]

    df_patients = df_patients.drop(['RACE','GENDER','MARITAL_STATUS'], axis=1)
    df_patients['PATIENT_SK'] = df_patients['PATIENT_SK'].astype(str)
    df_patients.fillna(0, inplace=True)
    print(f'df_patients shape: {df_patients.shape}, cols: {df_patients.columns}')

    # drop duplicates samples (from the control sample)
    df_patients.drop_duplicates(inplace=True)

    if args.population_type == 'mixed':
        for code in ['264.9', '292.1', '300.1', '312.0', '313.1', '315.0', '315.1', '315.2', '348.8', '350.3', '752.0', '313.3']:
            # drop the ASD indicator column
            df_patients.drop(code, axis=1, inplace=True)
            print('code ', code, ' dropped')
        print('df after drop shape', df_patients.shape)

    # drop the ASD indicator column
    elif '313.3' in df_patients.columns:
        df_patients.drop('313.3', axis=1, inplace=True)

    # fill the NaN values with 0
    df_patients.fillna(0, inplace=True)
    df_patients = df_patients.reset_index(drop=True)

    df_dict['patients'] = df_patients
    print(f'Finished handling patients dataframe!')

@measure_time
def data_prep(df_dict: dict, train_index=None):
    print('-' * NUM_OF_HYPHENS)
    print(f'Data Preparation...')
    if train_index is not None:
        df1 = df_dict['patients'].iloc[train_index].reset_index(drop=True)
    else:
        df1 = df_dict['patients']

    # drop the patient_sk column
    df1_phecodes = df1.drop(['PATIENT_SK'], axis=1).values

    # drop rows with all zeros
    df1_phecodes = df1_phecodes[df1_phecodes.sum(axis=1) > 0]

    print('df shape (after removing duplicates and zero rows)', df1_phecodes.shape)

    #  convert the data to a torch object
    torch_control_patients_for_similarity = torch.tensor(df1_phecodes, dtype=torch.int32)
    print('torch_patients_for_similarity', torch_control_patients_for_similarity)
    print('torch_patients_for_similarity shape: ', np.array(torch_control_patients_for_similarity).shape)

    a = sum(np.array(torch_control_patients_for_similarity).T)
    print('num of phecodes per patient:', a[:10])
    print('mean num of phecodes: ', a.mean())

    #  convert the data to a sparse torch object
    sparse_torch = torch_control_patients_for_similarity.to_sparse_coo()
    print('sparse_torch', sparse_torch)

    # validate that each patient exists only once
    tst = df1[['PATIENT_SK', '8.0']].groupby('PATIENT_SK').agg('count').reset_index().sort_values(by='8.0', ascending=False)
    if len(tst[tst['8.0']>1]):
        print('error: there are patients with more than one row')
        return None
    else:
        print('no duplicates patients :)')

    # get the patient_sk from the indexes
    p_sk_list = list(df1['PATIENT_SK'][pd.DataFrame(sparse_torch.indices()[0].tolist())[0]])

    # get the phecodes from the indexes
    phecodes_list = list(df1.drop(['PATIENT_SK'], axis=1).columns[pd.DataFrame(sparse_torch.indices()[1].tolist())[0]])

    print('p_sk_list', p_sk_list[:10])
    print('len(p_sk_list)', len(p_sk_list))
    print('phecodes_list', phecodes_list[:10])

    # create a dataframe with the patient_sk and the phecodes
    p_sk_phecodes = pd.DataFrame({"item_id":p_sk_list, "user_id":phecodes_list})
    p_sk_phecodes['item_id'] = p_sk_phecodes['item_id'].astype(str)
    print('p_sk_phecodes', p_sk_phecodes)

    df_dict['p_sk_phecodes_torch'] = p_sk_phecodes
    print('data_prep successfully finished')

@measure_time
def get_top10_nn(df_dict: dict):
    p_sk_phecodes = df_dict['p_sk_phecodes_torch']
    m_1 = tc.recommender.item_similarity_recommender.create(tc.SFrame(p_sk_phecodes), verbose=False)
    print('recommender model: ', m_1)

    #get 10 nearest items graph
    nn_1 = m_1.get_similar_items(k=10)
    nn_1['edge'] = nn_1.apply(lambda a:(a['item_id'], a['similar']))
    nn_1['edge_weight'] = nn_1.apply(lambda a:(a['item_id'], a['similar'], round(a['score'],3)))
    nn_1.print_rows(num_rows=15)
    print('item_id len', len(nn_1['item_id'].unique()))
    print('similar len', len(nn_1['similar'].unique()))

    df_dict['df_10_nn'] = nn_1
    print('get_top10_nn successfully finished')

@measure_time
def create_nx_graph(df_dict: dict):
    nn_1 = df_dict['df_10_nn']
    G = nx.Graph()
    G.add_weighted_edges_from(nn_1['edge_weight'])
    print('len(G.edges)', len(G.edges))
    print('len(G.nodes)', len(G.nodes))

    print('create nx_graph successfully finished')
    return G

@measure_time
def get_community_louvain(df_dict: dict, G, args, output_dir):
    print('-' * NUM_OF_HYPHENS)
    print(f'Performing Louvain Community Detection...')
    start_time = time.time()

    partition = community_louvain.best_partition(G, weight='weight')

    # Create a dictionary for communities
    communities = {}
    for node, community in partition.items():
        if community not in communities:
            communities[community] = []
        communities[community].append(node)

    print(f'num of communities were found: {len(communities.items())}')
    # Print community sizes
    community_lengths = sorted([len(comm) for comm in communities.values()], reverse=True)
    print('communities_len: ', community_lengths)

    # Save communities to CSV
    df = pd.DataFrame.from_dict(communities, orient='index')
    df = df.transpose()
    # Compute lengths for each column
    column_lengths = df.count()

    # Sort column names by length
    sorted_columns = column_lengths.sort_values(ascending=False).index

    # Reorder columns
    df_sorted = df[sorted_columns]
    df_sorted.to_csv(os.path.join(output_dir, f'{args.population_type}_louvain_communities_10nn.csv'), index=False)
    df_dict['df_communities'] = df_sorted
    print("Runtime: %s seconds" % (time.time() - start_time))
    print('get_community_louvain successfully finished')
    return df_sorted

@measure_time
def get_community_label_propagation(df_dict: dict, G, args, output_dir):
    print('-' * NUM_OF_HYPHENS)
    print(f'Performing Label Propagation Community Detection...')

    communities_generator = nx.algorithms.community.label_propagation.asyn_lpa_communities(G, weight='weight')
    communities = {i: list(community) for i, community in enumerate(communities_generator)}

    print(f'num of communities were found: {len(communities.items())}')

    # Print community sizes
    community_lengths = [len(comm) for comm in communities.values()]
    print('communities_len: ', sorted(community_lengths, reverse=True))

    # Save communities to CSV
    df = pd.DataFrame.from_dict(communities, orient='index')
    df = df.transpose()
    df.to_csv(os.path.join(output_dir, f'{args.population_type}_label_propagation_communities_10nn'), index=False)
    df_dict['df_communities'] = df

    print('get_community_label_propagation successfully finished')
    return df

def get_community_infomap(df_dict: dict, G, args, output_dir):
    print('-' * NUM_OF_HYPHENS)
    print(f'Performing Infomap Community Detection...')

    im = infomap.Infomap()
    network = im.network

    # Create a mapping from node labels to integers
    node_mapping = {node: i for i, node in enumerate(G.nodes())}

    # Assuming G is your networkx graph
    for u, v, data in G.edges(data=True):
        network.addLink(node_mapping[u], node_mapping[v], float(data['weight']))

    im.run()
    # Create a dictionary for communities
    communities = {}
    for node in im.tree:
        if node.is_leaf:
            if node.module_id not in communities:
                communities[node.module_id] = []
            communities[node.module_id].append(node.node_id)

    print(f'num of communities were found: {len(communities.items())}')
    # Print community sizes
    community_lengths = [len(comm) for comm in communities.values()]
    print('communities_len: ', sorted(community_lengths, reverse=True))

    # Save communities to CSV
    df = pd.DataFrame.from_dict(communities, orient='index').transpose()
    df.to_csv(os.path.join(output_dir,f'{args.population_type}_infomap_communities_10nn.csv'), index=False)
    df_dict['df_communities'] = df

    print('get_community_infomap successfully finished')
    return df

def get_community_greedy_modularity(df_dict: dict, G, args, output_dir):
    print('-' * NUM_OF_HYPHENS)
    print(f'Performing Greedy Modularity Community Detection...')

    communities = nx.algorithms.community.greedy_modularity_communities(G, weight='weight', resolution=10)
    communities_len = [len(comm) for comm in communities]
    print('num of communities: ', len(communities))
    print('communities_len: ', communities_len)
    df = pd.DataFrame(communities).transpose()
    df.to_csv(os.path.join(output_dir,f'{args.population_type}_greedy_modularity_res10_communities_10nn.csv'), index=False)
    df_dict['df_communities'] = df
    print('get_community_greedy_modularity successfully finished')
    return df

@measure_time
def plot_community_lengths(df_dict: dict, args, output_dir):
    df = df_dict['df_communities']
    # Create a Series of community sizes
    community_sizes = df.count().sort_values(ascending=False)

    # Compute total number of data points across all communities
    total_data_points = community_sizes.sum()

    # Plot community sizes
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(community_sizes)), community_sizes)
    plt.xlabel('Communities')
    plt.ylabel('Size')
    title = f'Community Sizes for {args.population_type}:'
    plt.title(title)

    # Determine indices for the four bars to annotate
    indices_to_annotate = [int(len(bars) * frac) for frac in [0, 0.25, 0.5, 0.75]]

    # Adding percentage on the selected bars
    for i, bar in enumerate(bars):
        if i in indices_to_annotate:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 0.05, f'{(yval/total_data_points)*100:.1f}%', ha='center', va='bottom')

    # Adding total communities text inside the plot
    plt.text(0.1, 0.9, f'Total communities: {len(community_sizes)}', transform=plt.gca().transAxes)

    # Adjust subplot parameters to give specified padding
    plt.tight_layout()

    # Save the plot to a file
    plot_file_name = os.path.join(output_dir, f'plot_sample_{args.population_type}.png')
    plt.savefig(plot_file_name, format='png', dpi=300)

    plt.show()

    # Close the figure after saving to free up memory
    plt.close()
    print('plot_community_lengths successfully finished')

def df_to_list_of_lists(df):
    list = df.T.values.tolist()
    list_without_nan = [[x for x in sublist if pd.notnull(x)] for sublist in list]
    return list_without_nan

def add_demographic_data(df_dict: dict, args, output_dir):
    print('-' * NUM_OF_HYPHENS)
    print(f'Adding demographic data to communities...')

    df = df_dict['df_communities']
    df_patients_full = df_dict['patients_full']
    list_of_lists = df_to_list_of_lists(df)
    df_lengths = [len(x) for x in list_of_lists]
    print(f'{args.population_type} communities lengths {df_lengths}')
    all_pop_patients_clusters = pd.DataFrame(columns=['PATIENT_SK', 'cluster'])
    for c, list in enumerate(list_of_lists):
        all_pop_patients_clusters_tmp = pd.DataFrame(columns=['PATIENT_SK', 'cluster'])
        all_pop_patients_clusters_tmp['PATIENT_SK'] = list
        all_pop_patients_clusters_tmp['cluster'] = c
        # print(all_pop_patients_clusters_tmp.shape)
        all_pop_patients_clusters = pd.concat([all_pop_patients_clusters, all_pop_patients_clusters_tmp], axis=0, ignore_index=True)

    all_pop_patients_clusters['PATIENT_SK'] = all_pop_patients_clusters['PATIENT_SK'].astype(int)
    all_pop_patients_clusters = all_pop_patients_clusters.merge(df_patients_full, on = 'PATIENT_SK', how='left')
    all_pop_patients_clusters.drop_duplicates(inplace=True)
    print(all_pop_patients_clusters)
    df_dict['df_communities_with_demographics'] = all_pop_patients_clusters
    all_pop_patients_clusters.to_csv(os.path.join(output_dir, f'{args.population_type}_with_clusters_and_demographics.csv'), index=False)
    print('add_demographic_data successfully finished')


def evaluate_clusters(data_full, fold_index):
    # Calculate evaluation metrics
    data = data_full.drop(['PATIENT_SK', 'cluster', 'RACE', 'GENDER', 'MARITAL_STATUS'], axis=1)
    labels = data_full['cluster']

    silhouette = silhouette_score(data, labels)
    davies_bouldin = davies_bouldin_score(data, labels)
    calinski_harabasz = calinski_harabasz_score(data, labels)

    # You may also calculate other internal validation metrics if needed

    evaluation_results = {
        'fold_index': fold_index,
        'num_of_samples': len(data_full),
        'num_of_clusters': len(data_full['cluster'].unique()),
        'silhouette_score': silhouette,
        'davies_bouldin_index': davies_bouldin,
        'calinski_harabasz_index': calinski_harabasz
        # Add other metrics if needed
    }

    return evaluation_results

def delete_files(paths: list) -> None:
    """
    Delete the local files
    :param paths: List of paths to delete
    """
    print('Deleting local files...')
    for path in paths:
        try:
            if '/' in path:
                path = path.split('/')[1]
            os.remove(path)
        except:
            pass

    print('Finished deleting local files!')

