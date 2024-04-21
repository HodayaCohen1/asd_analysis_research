import argparse
import os
import time
import importlib
import inspect
from typing import List, Dict

import pandas as pd
import psutil

from helpers.consts import NUM_OF_HYPHENS
from helpers.exceptions import ArgumentMissingException


def parse_args():
    parser = argparse.ArgumentParser(description='Gets arguments')
    parser.add_argument('--type', '-t', type=str, required=True, help='Type name')
    parser.add_argument('--patients_file', '-f', type=str, required=True, help='input patients file')
    parser.add_argument('--icd_file', '-icd', type=str, required=True, help='input icd file')
    parser.add_argument('--population_type', '-pt', type=str, required=True, help='population type - can be ASD/ Control/mixed')
    parser.add_argument('--community_detection_type', '-ctt', type=str, required=False, help='community detection type - can be louvain/label_propagation/infomap/greedy_modularity')

    args = parser.parse_args()
    return args


def check_args(args):
    """
    Checks the existence of all args
    :param args: Arguments
    :return:
    """
    print('-' * NUM_OF_HYPHENS)
    print('Checking args...')
    if args.type is None:
        raise ArgumentMissingException('type argument is missing')
    if args.patients_file is None:
        raise ArgumentMissingException('patients file argument is missing')
    if args.icd_file is None:
        raise ArgumentMissingException('icd file argument is missing')
    if args.population_type is None:
        raise ArgumentMissingException('population type argument is missing')
    if args.population_type not in ['ASD', 'Control', 'mixed']:
        raise ArgumentMissingException('population type argument should be ASD/ Control/mixed')
    if args.type=='community_detection' and args.community_detection_type is None:
        raise ArgumentMissingException('community detection type argument is missing')
    print('Finished checking args!')

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
