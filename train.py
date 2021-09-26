'''
Author: Hao Luo
Sep. 2021
'''

import argparse
from process import train_process
from config  import configurations

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train the model."
    )
    parser.add_argument(
        "--trn_data_path", required=True, type=str,
        help="path of training data"
    )
    parser.add_argument(
        "--val_data_path", required=True, type=str,
        help="path of validation data"
    )
    parser.add_argument(
        "--store_model_path", required=True, type=str,
        help="path of checkpoint"
    )
    parser.add_argument(
        "--load_model_path", default=None, type=str,
        help="path of pretrained model"
    )
    args = parser.parse_args()

    config = configurations()
    config.trn_data_path = args.trn_data_path
    config.val_data_path = args.val_data_path
    config.store_model_path = args.store_model_path
    config.load_model_path = args.load_model_path
    print ('config:\n', vars(config))
    
    train_losses, val_losses, top_1_acc = train_process(config)
