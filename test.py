import sys
sys.path.append("..")
from preprocessing.data import *
from preprocessing.augmentation import *
from metric import *
import time
from utils import get_model, get_augment, get_n_params
from torch.utils.data import DataLoader

def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth
    # Print header
    print("    " + empty_cell, end=" ")
    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=" ")
    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end=" ")
        for j in range(len(labels)):
            cell = "%{0}.1f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=" ")
        print()


# test (inference)
def run_test(config, dir):

    device = "cuda"

    out_dir = './models'
    config.model_name = config.model + '_' + config.image_mode + '_' + str(config.image_size)
    out_dir = os.path.join(out_dir,config.model_name)
    initial_checkpoint = config.pretrained_model
    augment = get_augment(config.image_mode)

    # net ---------------------------------------
    net = get_model(model_name=config.model, num_class=2, is_first_bn=True)

    # TODO: only if multiple GPUs
    net = torch.nn.DataParallel(net)

    net.to(device)

    if initial_checkpoint is not None:
        save_dir = os.path.join(out_dir + '/checkpoint', dir, initial_checkpoint)
        initial_checkpoint = os.path.join(out_dir + '/checkpoint',initial_checkpoint)
        print('\t' + 'initial_checkpoint = {}\n'.format(initial_checkpoint))
        net.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))
        if not os.path.exists(os.path.join(out_dir + '/checkpoint', dir)):
            os.makedirs(os.path.join(out_dir + '/checkpoint', dir))

    test_dataset = FDDataset(mode='test', modality=config.image_mode,image_size=config.image_size,
                             fold_index=config.train_fold_index, augment=augment, dataset_path=config.test_list)

    print("test batch size: {}".format(config.test_batch_size if config.test_batch_size > 1
                                        else str(config.test_batch_size) + " (realtime)"))

    test_loader = DataLoader(test_dataset,
                             shuffle=False,
                             batch_size=config.test_batch_size,
                             drop_last=False,
                             num_workers=4)

    ins = []
    labels = []

    with open(config.test_list) as test_list:
        # import os.path
        for line in test_list.readlines():
            filename, _, _, label  = line.split(" ")
            filename = filename.strip()
            label = int(label)
            labels.append(label)
            filename = os.path.basename(filename)
            # print(filename, label, sep="\t")
            ins.append([filename, label])
            # print(line)

    net.to(device)
    net.eval()

    # start_time = time.time()
    out = infer_test(net, test_loader, device)
    # time_delta = time.time() - start_time
    # print("testing took {:.2f} seconds".format(time_delta))
    # print("percentage predicted true:", 100*np.mean(out > 0.5))

    np.set_printoptions(precision=2, suppress=True)
    preds = np.array(out).tolist()


    from sklearn.metrics import confusion_matrix
    y_pred = [int(i + 0.5) for i in out]

    print_cm(confusion_matrix(y_true=labels, y_pred=y_pred), labels=["True", "False"])

    summary = []

    for (filename, label), out in zip(ins, preds):
        # print(filename, label, out, sep="\t")
        summary.append([filename, label, out])

    # summary = sorted(summary, key=lambda f: f[2], reverse=True)
    # for f, l, o in summary:
    #     print(f, l, o)

    # print(np.array(out) * 100)
    print('done')