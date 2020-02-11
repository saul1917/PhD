import numpy as np
import os
import glob
from PIL import Image
import sys
import matplotlib.pyplot as plt
from matplotlib import cm
sys.path.append('../')


def getLR(hr_data):
    imgfft = np.fft.fftn(hr_data)
    x_center = imgfft.shape[0] // 2
    y_center = imgfft.shape[1] // 2
    z_center = imgfft.shape[2] // 2
    imgfft[x_center-20 : x_center+20, y_center-20 : y_center+20, z_center-20 : z_center+20] = 0
    imgifft = np.fft.ifftn(imgfft)
    img_out = abs(imgifft)
    print(type(img_out))
    return img_out


im = Image.open("Bridge.jpg").convert('LA')
im.show()

npIm = np.array(im)

output = getLR(npIm)
print(output.shape)
print(np.max(output))
print(np.min(output))


plt.matshow(output[:,:,0])
plt.colorbar()
plt.show()



