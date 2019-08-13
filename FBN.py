import torch
from model.FaceBagNet_model_A import Net
from preprocessing.augmentation import color_augumentor
from copy import deepcopy
import numpy as np
import torch.nn.functional as F
import cv2
from mtcnn.mtcnn import MTCNN


class FaceBagNet:
    def __init__(self, model_path, patch_size=48, torch_device="cpu"):

        # TODO: bn, id_class?
        self.model_path = model_path
        self.patch_size = patch_size

        self.neural_net = Net(num_class=2, id_class=300, is_first_bn=True)
        self.neural_net.load_pretrain(self.model_path)

        self.neural_net = torch.nn.DataParallel(self.neural_net)

        self.neural_net.to(torch_device)

        self.torch_device = torch_device

        # TODO: this line
        self.neural_net.eval()

        self.augmentor = color_augumentor

    # returns probability that the image is genuine (not presented)
    def predict(self, full_size_image):

        image = deepcopy(full_size_image)  # TODO: remove copying

        image = self.augmentor(image, target_shape=(self.patch_size, self.patch_size, 3), is_infer=True)

        n = len(image)
        image = np.concatenate(image, axis=0)
        image = np.transpose(image, (0, 3, 1, 2))
        image = image.astype(np.float32)
        image = image.reshape([n, 3, self.patch_size, self.patch_size])
        image = np.array([image])
        image = image / 255.0

        input_tensor = torch.FloatTensor(image)

        shape = input_tensor.shape
        b, n, c, w, h = shape
        # print(b, n, c, w, h)

        input_tensor = input_tensor.view(b * n, c, w, h)

        # inpt = inpt.cuda() if torch.cuda.is_available() else inpt.cpu()
        input_tensor = input_tensor.to(self.torch_device)

        # print(input_tensor)

        with torch.no_grad():
            logit, _, _ = self.neural_net(input_tensor)
            logit = logit.view(b, n, 2)
            logit = torch.mean(logit, dim=1, keepdim=False)
            prob = F.softmax(logit, 1)

        is_real = list(prob.data.cpu().numpy()[:, 1])[0]

        return is_real


if __name__ == "__main__":

    model48 = FaceBagNet("model48.pth", 48, "cuda")
    # model32 = FaceBagNet("model32.pth", 32, "cuda")
    model16 = FaceBagNet("model16.pth", 16, "cuda")

    detector = MTCNN()

    # capture = cv2.VideoCapture("images/video.mp4")
    capture = cv2.VideoCapture(0)

    capture.set(3, 1280)
    capture.set(4, 720)

    while True:
        ret, frame = capture.read()
        detection = detector.detect_faces(frame)

        if detection:

            for i, bbox in enumerate(detection):
                bbox = bbox['box']
                pt1 = bbox[0], bbox[1]
                pt2 = bbox[0] + bbox[2], bbox[1] + int(bbox[3])

                # TODO: do not hardcode face index
                if i == 0:
                    crop_img = frame[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
                    cv2.imshow("detected face", crop_img)
                    print("model48: {} model16: {}".format(model48.predict(crop_img),
                                                           model16.predict(crop_img)))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
