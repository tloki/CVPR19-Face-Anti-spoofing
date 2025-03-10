from preprocessing.data_helper import *
from torch.utils.data import Dataset


class FDDataset(Dataset):
    def __init__(self, mode, modality='color', fold_index=-1, image_size=48, augment=None, augmentor=None,
                 balance=True, dataset_path=None, stream_device=None, width=1280, height=720, display_window=None,
                 detect_function=None):

        super(FDDataset, self).__init__()
        print('fold: '+str(fold_index))
        print(modality)

        if dataset_path is None:
            dataset_path = DATA_ROOT

        self.dataset_path = dataset_path

        self.mode = mode
        self.modality = modality

        self.augment = augment # this is set
        self.augmentor = augmentor # this is None
        self.balance = balance

        self.channels = 3

        self.stream_service = stream_device
        self.width = width
        self.height = height

        self.detect_function = detect_function

        self.display_window = display_window

        # TODO: is this unused?
        self.train_image_path = TRN_IMGS_DIR
        self.test_image_path = TST_IMGS_DIR

        self.image_size = image_size
        self.fold_index = fold_index  # ovo doslovno nicemu ne sluzi

        self.capture = None
        self.set_mode(self.mode,self.fold_index)

    def set_mode(self, mode, fold_index):
        self.mode = mode
        self.fold_index = fold_index
        print('fold index set: ', fold_index)

        # differentiate between train and test
        # the only difference being - path of txt file with a list of data
        # train set in shuffled + balanced
        if self.mode == 'test':
            self.test_list = load_test_list(path=self.dataset_path)
            self.num_data = len(self.test_list)
            print('set dataset mode: test')
            n_out_samples = 10
            n_out_samples = min(n_out_samples, len(self.test_list))
            print("first {} samples:".format(n_out_samples))
            for i in range(n_out_samples):
                print(self.test_list[i])
            print("dataset length (number of samples): ", end="")

        # validation
        elif self.mode == 'val':
            self.val_list = load_val_list(path=self.dataset_path)
            self.num_data = len(self.val_list)
            print('set dataset mode: test')

        # train
        elif self.mode == 'train':
            self.train_list = load_train_list(path=self.dataset_path)

            # random.shuffle(self.train_list)
            np.random.shuffle(self.train_list)
            self.num_data = len(self.train_list)

            # just split list in positives and negatives...
            if self.balance:
                self.train_list = transform_balance(self.train_list)
            print('set dataset mode: train')

        # video file or webcam
        elif self.mode == 'realtime':
            print('set realtime camera/video capture')

            if self.stream_service.isdigit():
                self.stream_service = int(self.stream_service)
            self.capture = cv2.VideoCapture(self.stream_service)
            self.capture.set(3, self.width)
            self.capture.set(4, self.height)
            self.num_data = None

        else:
            ValueError("unimplemented mode '{}'".format(self.mode))

        print(self.num_data)
        print()

    def livestream(self):
        # print("started live detection loop:")
        detected = False
        crop_img = None
        # cv2.namedWindow("detected")
        while not detected:
            # video stream util face is detected
            ret, frame = self.capture.read()

            # cv2.imshow("detected", frame)
            # return frame
            if not ret:
                raise RuntimeError("invalid frame return value")

            detection = self.detect_function(frame)
            # cv2.imshow("detected", frame)
            # print("n")

            if not detection:
                detected = False
                # print(".", end="")
                sys.stdout.flush()
            else:
                # print()
                detected = True
                # print("detected")

                for i, bbox in enumerate(detection):
                    bbox = bbox['box']
                    pt1 = bbox[0], bbox[1]
                    pt2 = bbox[0] + bbox[2], bbox[1] + int(bbox[3])

                    # frame = np.zeros_like(frame)
                    # cv2.rectangle(frame, pt1, pt2, color=(0, 255, 0), thickness=2)
                    # cv2.putText(frame, "id: "+ str(i), (bbox[0], bbox[1] - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    #             fontScale=1, color=(0, 0, 255), thickness=2)

                    # TODO: do not hardcode face index
                    if i == 0:
                        crop_img = frame[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
                        # print(crop_img)

                        try:
                            cv2.imshow("detected", crop_img)
                        except:
                            detected = False
                            continue
        return crop_img

    # get i-th item, except if balance param is set - in that case return random sample, ignoring the given index
    def __getitem__(self, index): # FDDataset_instance[index]

        image = None

        if self.fold_index is None:
            # print('WRONG!!!!!!! fold index is NONE!!!!!!!!!!!!!!!!!')
            raise ValueError("fold index is NONE")

        # get some color, depth, ir, label (color, depth and ir are PATHS, not imgs (yet))
        if self.mode == 'train':
            # choose true/false sample with equal probs,
            # does not take 'index' as parameter
            if self.balance:
                # 'uniform' - not really but ok for what it is...
                # if random.randint(0, 1)==0:
                if np.random.randint(0, 2) == 0:
                    tmp_list = self.train_list[0]
                else:
                    tmp_list = self.train_list[1]

                # random sample, given label (label of random choice in list)
                pos = np.random.randint(0, len(tmp_list) - 1 + 1)
                color, depth, ir, label = tmp_list[pos]
            else:
                color, depth, ir, label = self.train_list[index]

        elif self.mode == 'val':
            color, depth, ir, label = self.val_list[index]

        elif self.mode == 'test':
            color, depth, ir = self.test_list[index][0:3]
            test_id = color + ' ' + depth + ' ' + ir

        elif self.mode == 'realtime':
            self.modality = None
            image = self.livestream()

        else:
            raise ValueError("mode expected to be in (train, test, val, realtime), but got '{}' instead".format(
                self.mode))

        img_path = None
        # get full path, choose only a needed modality
        if self.modality == 'color':
            img_path = os.path.join(self.dataset_path, color)
        elif self.modality == 'depth':
            img_path = os.path.join(self.dataset_path, depth)
        elif self.modality == 'ir':
            img_path = os.path.join(self.dataset_path, ir)

        if self.mode != 'realtime':
            # load BGR (color) image, resize photo to 112 x 112
            image = cv2.imread(img_path, 1)
            if image is None:
                raise RuntimeError("unable to open image {}".format(img_path))

        if self.mode == 'realtime':

            from copy import deepcopy
            raw_img = deepcopy(image)

            print("size: {}\t".format(image.shape).expandtabs(10), end="")
            image = self.augment(image, target_shape=(self.image_size, self.image_size, 3), is_infer=True)
            n = len(image)
            image = np.concatenate(image, axis=0)
            image = np.transpose(image, (0, 3, 1, 2))
            image = image.astype(np.float32)
            image = image.reshape([n, self.channels, self.image_size, self.image_size])
            image = np.array([image])
            image = image / 255.0

            inpt = torch.FloatTensor(image)

            return inpt, deepcopy(raw_img)

        elif self.mode == 'train':
            # self.augment (for color) == color_augumentor, ir = ir_augumentor, dpth = depth_augumentor
            # randomly flip horizontally, flip vertically, blur, rotate, crop part of photo, blackout part of photo
            image = self.augment(image, target_shape=(self.image_size, self.image_size, 3))

            # resize to dest
            image = cv2.resize(image, (self.image_size, self.image_size))

            # channel, width, height
            image = np.transpose(image, (2, 0, 1))
            image = image.astype(np.float32)
            image = image.reshape([self.channels, self.image_size, self.image_size])
            image = image / 255.0  # convert to float 0-1
            label = int(label)

            # return tensor of 3 channel image, and 1x1 tensor with label (integer)
            inpt = torch.FloatTensor(image)
            labl = torch.LongTensor(np.asarray(label).reshape([-1]))
            return inpt, labl

        elif self.mode == 'val':
            # make 36 patches from image
            image = self.augment(image, target_shape=(self.image_size, self.image_size, 3), is_infer=True)

            n = len(image)

            image = np.concatenate(image,axis=0)

            image = np.transpose(image, (0, 3, 1, 2))
            image = image.astype(np.float32)
            image = image.reshape([n, self.channels, self.image_size, self.image_size])
            image = image / 255.0
            label = int(label)

            inpt = torch.FloatTensor(image)
            labl = torch.LongTensor(np.asarray(label).reshape([-1]))
            return inpt, labl

        elif self.mode == 'test':
            image = self.augment(image, target_shape=(self.image_size, self.image_size, 3), is_infer = True)
            n = len(image)
            image = np.concatenate(image,axis=0)
            image = np.transpose(image, (0, 3, 1, 2))
            image = image.astype(np.float32)
            image = image.reshape([n, self.channels, self.image_size, self.image_size])
            image = image / 255.0

            inpt = torch.FloatTensor(image)

            return inpt, test_id

        else:
            raise ValueError("Expected mode to be in (train, val, test), got '{}'".format(self.mode))


    def __len__(self):
        return self.num_data


# check #################################################################
def run_check_train_data():
    from preprocessing.augmentation import color_augumentor
    augment = color_augumentor
    dataset = FDDataset(mode='train', fold_index=-1, image_size=32,  augment=augment)
    print(dataset)

    num = len(dataset)
    for m in range(num):
        i = np.random.choice(num)
        image, label = dataset[m]
        print(image.shape)
        print(label.shape)


# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))
    run_check_train_data()


