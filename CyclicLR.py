import sys
sys.path.append("..")
import argparse
# from process.data import *
# from process.augmentation import *
# from metric import *
# from loss.cyclic_lr import CosineAnnealingLR_with_Restart
# import time
# from utils import get_model, get_augment, get_n_params
from test import run_test
from train import run_train
from realtime import run_realtime


# run train or dest function, dependant on input arguments
def main(config):

    # if config.device == "auto":
    #     device =
    # elif config.device == "cpu":
    #     net = net.cpu()
    # elif config.device == "gpu":
    #     net = net.cuda()
    # else:q
    #     raise ValueError

    if config.mode.startswith("train"):
        run_train(config)

    elif config.mode.startswith("test"):
        # take the best model on validation set
        config.pretrained_model = r'global_min_acer_model.pth'
        run_test(config, dir='global_test_36_TTA')

    elif config.mode == 'realtime':
        config.pretrained_model = r'global_min_acer_model.pth'
        run_realtime(config, dir='global_test_36_TTA')

    return


# parse input arguments and pass them to main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_fold_index', type=int, default = -1)

    parser.add_argument('--model', type=str, default='model_A')
    parser.add_argument('--image_mode', type=str, default='color', choices=['color', 'depth', 'ir'])
    parser.add_argument('--image_size', type=int, default=48) # empirically adjusted

    parser.add_argument('--train_batch_size', type=int, default=128)
    parser.add_argument('--valid_batch_size', type=int, default=128)
    parser.add_argument('--test_batch_size', type=int, default=20)
    # parser.add_argument('--cycle_num', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--epochs_valid_start', type=int, default=50)  # number of random restarts

    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'infer_test', 'test', 'test-single', 'realtime'])

    parser.add_argument('--stream', default=None)

    parser.add_argument('--pretrained_model', type=str, default=None)
    # parser.add_argument('--dataset_path', type=str, default="/home/loki/Datasets/spoofing/NUAA/Detectedface52")

    parser.add_argument('--dataset_workers', type=int, default=4)
    parser.add_argument('--device', type=str, choices=["auto", "gpu", "cpu"], default="auto")

    # TODO: implement
    # parser.add_argument('--test-sample', type=str, default=None)
    parser.add_argument('--train_list', type=str, default=None)
    parser.add_argument('--validation_list', type=str, default=None)
    parser.add_argument('--test_list', type=str, default=None)

    config = parser.parse_args()

    # print(config)
    # print(config.dataset_path)

    main(config)

    print("end")
