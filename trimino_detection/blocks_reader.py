from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from PIL import Image, ImageFilter
from scipy import signal

from scipy import ndimage



def mask_contrast(mask_c, window=1):
    mask = mask_c.copy()
    for i in range(window, mask.shape[0] - window):
        for j in range(window, mask.shape[1] - window):
            mask[i,j] = mask_c[i-window:i+window-1, j-window:j+window-1].min()
    return mask


def conv_mask(mask, avg_window, bound):
    conv = np.ones(avg_window**2).reshape(avg_window, -1) / avg_window**2
    mask = (signal.convolve2d(mask, conv, boundary='symm', mode='same') > bound).astype(int)
    return mask


def read_points(sbb_c, mode):
    sbb = sbb_c.copy()

    def check(p):
        cp = p.copy()
        cp[cp.shape[0] // 2, cp.shape[1] // 2] = 1
        if cp.min() == 1:
            return False
        return True

    w = 2
    if mode == "up":
        it = range(sbb.shape[0]-w-1, w-1,-1)
    elif mode == "down":
        it = range(w, sbb.shape[0]-w)
    sbb[:w], sbb[-w:], sbb[:,:w], sbb[:,-w:] = 1, 1, 1, 1

    for i in it:
        for j in range(w, sbb.shape[1]-w):
            if check(sbb[i-w:i+w,j-w:j+w]):
                sbb[i,j] = 1
    w = 4
    if mode == "up":
        it = range(sbb.shape[0]-w-1, w-1,-1)
    elif mode == "down":
        it = range(w, sbb.shape[0]-w)
    for i in it:
        for j in range(w, sbb.shape[1]-w):
            if check(sbb[i-w:i+w,j-w:j+w]):
                sbb[i,j] = 1

    return (sbb != 1).sum() - 1


class BlocksReader:

    def __init__(self, img_reader):
        self.img_reader = img_reader
        self.m = img_reader.metric
        self.fblocks = []
        avg_window = 5
        bound = 0.5
        conv = np.ones(avg_window**2).reshape(avg_window, -1) / avg_window**2
        for i in range(len(self.img_reader.triangles.blocks)):
            angle = self.img_reader.triangles.rotate_angles[i]
            fblock = self.img_reader.triangles.blocks[i]
            pattern = self.img_reader.metric.patterns[self.img_reader.triangles.rotate_patterns[i]]

            # matching and rotation
            fblock = fblock * pattern
            fblock = ndimage.rotate(fblock, self.img_reader.metric.true_angle - angle, reshape=False).copy()
            fblock = mask_contrast(fblock, window=1)
            fblock = conv_mask(fblock, 4, 0.6)
            fblock = conv_mask(fblock, 3, 0.5)
            
            self.fblocks.append(fblock.copy())

        self.sbb = []
        self.points_number = []
        for i in range(len(self.fblocks)):
            self.sbb.append([
                self.fblocks[i][0:80,40:120].copy(),
                self.fblocks[i][75:160,0:80].copy(),
                self.fblocks[i][75:160,80:160].copy()
            ])
            self.points_number.append([
                read_points(self.sbb[-1][0], mode="up"),
                read_points(self.sbb[-1][1], mode="down"),
                read_points(self.sbb[-1][2], mode="down"),
            ])
      
    def get_block(self, idx):
        if len(self.fblocks) <= idx or idx < 0:
            print("Invalid block number")
            return None
            
        print("Block №" + "%3d" % idx+  "/"+ "%3d" % len(self.fblocks), " Points: ", self.points_number[idx])
        fig = plt.figure(figsize=(15,3))
        ax = fig.add_subplot(151)
        plt.axis('off')
        plt.imshow(self.img_reader.triangles.img_blocks[idx])
        ax = fig.add_subplot(152)
        plt.axis('off')
        plt.imshow(self.fblocks[idx])
        ax = fig.add_subplot(153)
        plt.axis('off')
        plt.imshow(self.sbb[idx][0])
        ax = fig.add_subplot(154)
        plt.axis('off')
        plt.imshow(self.sbb[idx][1])
        ax = fig.add_subplot(155)
        plt.axis('off')
        plt.imshow(self.sbb[idx][2])  

        
    
def apply_mask(base_img, mask, show=False):
    masked_img = base_img.copy()
    masked_img[mask == 0] = np.array([0, 0, 0])
    if show:
        fig = plt.figure(figsize=(8,5))
        plt.title("Masked image", fontsize=16)
        plt.axis('off')
        plt.imshow(masked_img)
    return masked_img


def format_output(img_reader, blocks_reader):
    with open("output/prediction.txt", "w") as fout:
        fout.write(str(img_reader.num_triangles) + "\n")
        for i in range(img_reader.num_triangles):
            point = img_reader.triangles.triangle_centers[i]
            fout.write("%4d" % point[0] + " " + "%4d" % point[1])
            sbb = blocks_reader.points_number[i]
            for j in range(3):
                fout.write(" " + str(sbb[j]))
            fout.write("\n")

 