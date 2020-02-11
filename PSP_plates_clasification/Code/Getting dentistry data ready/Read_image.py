import matplotlib.image as mpimg
import numpy as np
import os
import skimage
import cv2

def read_image(dir_name):
    list = []
    i = 0
    for filename in os.listdir(dir_name):
        realname = dir_name + filename
        img = cv2.imread(realname)
        list.append(img)
        #print("read image: "+ str(img.size))
        i += 1
        print("Read %s images" %i)
    a = np.array(list)
    a = skimage.img_as_ubyte(a)
    print("termina de hacer read image")
    return(a)
