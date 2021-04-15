from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from PIL import Image, ImageFilter
from scipy import signal
from scipy import ndimage
 


def transform_hard_colors(img, show=False):
    green = lambda p: ( p[1] > max(p[0],p[2]) + 10 )
    yellow = lambda p: ( min(p[0],p[1]) // 2 > p[2] )
    blue = lambda p: ( p[2] > 230 and p[2] > max(p[0],p[1]) + 15 ) 
    white = lambda p: ( p[0] > 252 and p[1] > 252 and p[2] > 252 )
    img = np.asarray(img).astype(np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if green(img[i,j]) or yellow(img[i,j]) or blue(img[i,j]) or white(img[i,j]):
                img[i,j] = np.array([60, 60, 190])
    img = Image.fromarray(img)
    if show:
        fig = plt.figure(figsize=(8,5))
        plt.title("Norm", fontsize=16)
        plt.axis('off')
        plt.imshow(img)
    return img


def norm_img(img, show=False):
    img = np.asarray(img).astype(float) / 255
    data = img.flatten().reshape(-1, 3)
    data = (data.T / (np.linalg.norm(data, axis=1) + 1e-3)).T
    data[:,0] += (data[:,0].mean() - 0.45) / 0.8
    data[:,1] += (data[:,1].mean() - 0.45) / 0.8
    data[:,2] += (data[:,2].mean() - 0.45) / 0.8

    m1 = data[:,0] + data[:,2] > 3 * data[:,1]
    m2 = abs(data[:,0] - data[:,2]) < 0.35
    data[m1 * m2] = np.array([195, 155, 204]) / 255

    data = np.clip(data, 0, 1)
    img = data.reshape(img.shape)
    if show:
        fig = plt.figure(figsize=(8,5))
        plt.title("Norm", fontsize=16)
        plt.axis('off')
        plt.imshow(img)
    img = (img * 255).astype(np.uint8)
    img = Image.fromarray(img)

    return img


def color_transform(img, show=False):
    img = np.asarray(img).astype(int)
    m = img[:,:,0].mean()
    delta = (m - 196) * 30
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i,j,0] - delta + 20> (img[i,j,1] + img[i,j,2]):
                img[i,j] = np.array([0, 0, 0])
    img = Image.fromarray(img.astype(np.uint8))
    if show:
        fig = plt.figure(figsize=(8,5))
        plt.title("Norm", fontsize=16)
        plt.axis('off')
        plt.imshow(img)
    return img
    

def normalized_img(img):
    img = transform_hard_colors(img)
    img = norm_img(img)
    img = color_transform(img)
    return img

 