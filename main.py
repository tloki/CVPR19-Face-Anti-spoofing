import sys
sys.path.append("..")
import argparse
from test import run_test
from train import run_train
from realtime import run_realtime


# run train or dest function, dependant on input arguments
def main(cfg):

    # TODO: implement device selection
    # if config.device == "auto":
    #     device =
    # elif config.device == "cpu":
    #     net = net.cpu()
    # elif config.device == "gpu":
    #     net = net.cuda()
    # else:q
    #     raise ValueError

    if cfg.mode.startswith("train"):
        run_train(cfg)

    elif cfg.mode.startswith("test"):
        # take the best model on validation set
        cfg.pretrained_model = r'global_min_acer_model.pth'
        run_test(cfg, dir='global_test_36_TTA')

    elif cfg.mode == 'realtime':
        cfg.pretrained_model = r'global_min_acer_model.pth'
        run_realtime(cfg)

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

    parser.add_argument('--num_restarts', type=int, default=50) # number of random restarts
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--epochs_valid_start', type=int, default=50)

    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'infer_test', 'test', 'test-single', 'realtime'])

    parser.add_argument('--stream', default=None)

    parser.add_argument('--pretrained_model', type=str, default=None)

    parser.add_argument('--dataset_workers', type=int, default=4)
    parser.add_argument('--device', type=str, choices=["auto", "gpu", "cpu"], default="auto")

    parser.add_argument('--train_list', type=str, default=None)
    parser.add_argument('--validation_list', type=str, default=None)
    parser.add_argument('--test_list', type=str, default=None)

    config = parser.parse_args()

    main(config)

    print("end")
