from time import time
import cv2

class FPSmeter(object):
    def __init__(self, font=cv2.FONT_HERSHEY_SIMPLEX, posx=None, posy=None, fontScale=0.5, fontColor=(255,0,0)):
        self.font = font
        self.posx = posx
        self.posy = posy
        self.time = time()
        self.scale = fontScale
        self.color = fontColor
        self.fps = None
        self.font = font
        return

    def tick(self, img):
        width, height, _ = img.shape
        now = time()
        delta = now - self.time
        self.time = now

        self.fps = 1 // delta
        if self.posx is None or self.posy is None:
            cv2.putText(img, "FPS: " + str(int(self.fps)), (height - 76, 14), self.font,
                        self.scale, self.color, 1)
        else:
            raise NotImplementedError