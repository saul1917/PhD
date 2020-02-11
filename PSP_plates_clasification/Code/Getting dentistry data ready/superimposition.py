from Read_image import read_image
import numpy as np
import os
import matplotlib.image as mpimg
import skimage
import matplotlib.pyplot as plt
import cv2
import math
import time
#from cv2_superposition import cut
import glob
def show(a):
    plt.imshow(a,cmap = 'gray')



# def clipped_add(a,b):
#     a = a.astype('uint16')
#     b = b.astype('uint16')
#     c = a+b
#     np.clip(c, 0, 255)
#     c = c.astype('uint8')
#     return c

def cut(img,a,b):
    copy = np.copy(img)
    copy[copy > b] = 0
    copy[copy <= a] = 0
    return copy


def exp_w(a, b, x):
    e = math.e
    y = a*(b**x)
    return y

def get_para(x1,y1,x2,y2):
    b = (y2/y1) ** (1/(x2-x1))
    a = y1/(b ** x1)
    return a, b



dir1 = 'selected_blank/' #original images of the 25 selected artifacts
dir2 = 'Test/' #contrast limited images of the 25 selected artifacts
dir3 = 'original_plates/'
label = os.listdir(dir1)

blank = read_image(dir1)
cl = read_image(dir2)
real = read_image(dir3)

cmos = blank[0].astype('float64')
#cv2.resize(cv2.cvtColor(skimage.img_as_ubyte(mpimg.imread('CMOS Control Image.PNG')), cv2.COLOR_BGR2GRAY), blank.shape[1:][::-1])
print("cmos" + str(cmos))
a = 0.15905414575341018
b = 1.0057432466391787
print("a y b")
#a, b = get_para(30, 0.1, 255, 0.2)
print("para")
for i in range(1):
    print("entraal loop 25")
    print("cl: " + str(cl.size))
    print("cl[i]: " + str(cl[i].size))
    w = exp_w(a, b, cl[i])
    print("w: " + str(w.size))
    m = cl[i] * w
    print("m: " + str(m.size))
    cmos = cmos.astype('float64')
    print("cmos 2: " + str(cmos.size))

    blend = cv2.addWeighted(m, 1, cmos, 0.85, 0)


    #exit()
    print("blend")
    high = cut(blank[i],110,255).astype('float64')
    high[high > 0] += 80
    high[high > 255] = 255
    blend = np.maximum(blend, high)
    blend[blank[1] == 0] = 0
    cv2.imwrite('./new/' + label[i], blend)
    print('Saved %d images-----equalized' % (i + 1))

print("a: ", a)
print("b: ", b)
