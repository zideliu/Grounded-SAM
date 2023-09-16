import cv2
import numpy as np

from PIL import Image


def crop_image(img,w_st):
    w_ed = w_st+320
    img = img[:,w_st:w_ed,:]
    img = Image.fromarray(img)
    img = img.resize((512,512))
    return img

for i in range(41,80):
    im_path = "output/img/frame"+str(i)+".png"
    img = np.array(Image.open(im_path).convert('RGB'))
    img = crop_image(img,200)
    img.save("frame/frame"+str(i)+".png")
