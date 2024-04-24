import os
import warnings

import pandas as pd

from helpers.flow_factory import get_flow
from helpers.utils import parse_args, check_args

if __name__ == '__main__':
    print('Starting...')
    warnings.filterwarnings("ignore")
    # ---------------------- Init S3 client ----------------------
    args = parse_args()

    check_args(args)

    flow = get_flow(type=args.type)

    flow.Init()

    flow.run_flow(args=args)

    print(f'Finished {flow}')

    os._exit(0)
