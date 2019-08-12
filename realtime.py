from metric import *
from mtcnn.mtcnn import MTCNN
from preprocessing.data import *
from preprocessing.augmentation import *
from utils import get_model, get_augment
sys.path.append("..")


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

    detector = MTCNN()

    test_dataset = FDDataset(mode='realtime', modality=config.image_mode,image_size=config.image_size,
                             fold_index=config.train_fold_index, augment=augment, dataset_path=None,
                             stream_device=config.stream, detect_function=detector.detect_faces)

    net.eval()

    infer_test_infinite(net, test_dataset)
    print('done')