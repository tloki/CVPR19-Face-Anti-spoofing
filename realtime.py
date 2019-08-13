from metric import *
from mtcnn.mtcnn import MTCNN
from preprocessing.data import *
from preprocessing.augmentation import *
from utils import get_model, get_augment
sys.path.append("..")


def infer_test_realtime(net, test_loader):

    import cv2
    cv2.namedWindow("detected")

    while True:
        inpt, raw_image = test_loader[0][0:2]
        inp = inpt.size()
        b, n, c, w, h = inp
        print(b, n, c, w, h)
        inpt = inpt.view(b*n, c, w, h)
        inpt = inpt.cuda() if torch.cuda.is_available() else inpt.cpu()

        with torch.no_grad():
            logit, _, _ = net(inpt)
            logit = logit.view(b, n, 2)
            logit = torch.mean(logit, dim=1, keepdim=False)
            prob = F.softmax(logit, 1)

        is_real = list(prob.data.cpu().numpy()[:, 1])[0]

        if is_real > 0.5:
            print("OK {}".format(is_real), end=" ")
        else:
            print("FAKE {}".format(is_real), end=" ")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def run_realtime(config):
    out_dir = './models'
    config.model_name = config.model + '_' + config.image_mode + '_' + str(config.image_size)
    out_dir = os.path.join(out_dir, config.model_name)
    initial_checkpoint = config.pretrained_model

    net = get_model(model_name=config.model, num_class=2, is_first_bn=True)
    net = torch.nn.DataParallel(net)

    initial_checkpoint = os.path.join(out_dir + '/checkpoint', initial_checkpoint)
    print('initial_checkpoint: {}'.format(initial_checkpoint))
    net.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))

    detector = MTCNN()

    augment = get_augment(config.image_mode)
    test_dataset = FDDataset(mode='realtime', modality=config.image_mode, image_size=config.image_size,
                             fold_index=config.train_fold_index, augment=augment, dataset_path=None,
                             stream_device=config.stream, detect_function=detector.detect_faces)

    net.eval()

    infer_test_realtime(net, test_dataset)
    print('done')
