'''
Author: Hao Luo
Sep. 2021
'''

import os 
import argparse
from process import test_process
from config import configurations

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Beam prediction."
    )
    parser.add_argument(
        "--test_data_path", required=True, type=str,
        help="path of testing data"
    )
    parser.add_argument(
        "--load_model_path", required=True, type=str,
        help="path of pretrained model"
    )
    args = parser.parse_args()

    config = configurations()
    config.test_data_path = args.test_data_path
    config.load_model_path = args.load_model_path
    print ('config:\n', vars(config))
    test_process(config)
