import sys
sys.path.append("..")
import argparse
from preprocessing.data import *
from preprocessing.augmentation import *
from metric import *
from loss.cyclic_lr import CosineAnnealingLR_with_Restart
import time
from utils import get_model, get_augment, get_n_params


def run_realtime(config, dir):
    out_dir = './models'
    config.model_name = config.model + '_' + config.image_mode + '_' + str(config.image_size)
    out_dir = os.path.join(out_dir,config.model_name)
    initial_checkpoint = config.pretrained_model
    augment = get_augment(config.image_mode)

    # net ---------------------------------------
    net = get_model(model_name=config.model, num_class=2, is_first_bn=True)
    net = torch.nn.DataParallel(net)

    if initial_checkpoint is not None:
        save_dir = os.path.join(out_dir + '/checkpoint', dir, initial_checkpoint)
        initial_checkpoint = os.path.join(out_dir +'/checkpoint',initial_checkpoint)
        print('\tinitial_checkpoint = %s\n' % initial_checkpoint)
        net.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))
        if not os.path.exists(os.path.join(out_dir + '/checkpoint', dir)):
            os.makedirs(os.path.join(out_dir + '/checkpoint', dir))

    from mtcnn.mtcnn import MTCNN
    detector = MTCNN()

    test_dataset = FDDataset(mode='realtime', modality=config.image_mode,image_size=config.image_size,
                             fold_index=config.train_fold_index, augment=augment, dataset_path=None,
                             stream_device=config.stream, detect_function=detector.detect_faces)

    # test_loader = DataLoader(test_dataset,
    #                          shuffle=False,
    #                          batch_size=36,
    #                          drop_last=False,
    #                          num_workers=1)

    ins = []

    net.eval()

    out = infer_test_infinite(net, test_dataset)

    np.set_printoptions(precision=2, suppress=True)
    preds = np.array(out).tolist()

    summary = []

    for (filename, label), out in zip(ins, preds):
        # print(filename, label, out, sep="\t")
        summary.append([filename, label, out])

    summary = sorted(summary, key=lambda f: f[2], reverse=True)
    for f, l, o in summary:
        print(f, l, o)

    # print(np.array(out) * 100)
    print('done')