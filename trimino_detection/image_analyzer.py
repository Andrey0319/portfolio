from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from PIL import Image, ImageFilter
from scipy import signal
from scipy import ndimage

from data_preprocess import normalized_img



def to_array(img):
    return np.asarray(img) / 255


def blur(img):
    new_img = Image.fromarray((img * 255).astype(np.uint8))
    new_img = new_img.filter(ImageFilter.BLUR)
    return to_array(new_img)


class Data:

    def __init__(self, img, blur=False):
        self.base_img = np.asarray(img)
        if blur:
            active_img = img.filter(ImageFilter.BLUR)
            for i in range(4):
                active_img = active_img.filter(ImageFilter.BLUR)
        else:
            active_img = img

        self.mx = np.asarray(active_img) / 255
        self.data = self.mx.flatten().reshape(-1, 3)

    def add_magical_features(self):
        self.data = np.hstack((self.data, (self.data[:,0] * self.data[:,1]).reshape(-1,1)))
        self.data = np.hstack((self.data, (self.data[:,3] / (self.data[:,2] + 1e-3)).reshape(-1,1)))
        self.data = np.hstack((self.data, (self.data[:,3] * self.data[:,2]).reshape(-1,1)))

    def take_subsample(self, every=3):
        return self.data[::every]

    def norm(self, show=False):
        self.data = (self.data.T / (np.linalg.norm(self.data, axis=1) + 1e-3)).T
        self.data = np.clip(self.data, 0, 1)
        if show:
            fig = plt.figure(figsize=(8,5))
            plt.title("Norm", fontsize=16)
            plt.axis('off')
            plt.imshow(self.data.reshape(self.mx.shape))
        return self.data.reshape(self.mx.shape)

    def cluster(self, method="gaussian", show=False):
        subsample = self.take_subsample(every=20)

        if method == "gaussian":
            self.gm = GaussianMixture(n_components=2, random_state=0).fit(subsample)
            labels = self.gm.predict(self.data)
        elif method == "kmeans":
            self.gm = KMeans(n_clusters=2, random_state=0).fit(subsample)
            labels = self.gm.predict(self.data)

        self._gm_mask = labels.reshape(self.mx.shape[0], self.mx.shape[1])
        if show:
            fig = plt.figure(figsize=(8,5))
            plt.title(method, fontsize=16)
            plt.axis('off')
            plt.imshow(self._gm_mask)
        return self._gm_mask

    def average_alg(self, avg_window=3, show=False):
        self.add_magical_features()
        lbl1 = self.cluster(method="kmeans", show=False)
        lbl2 = self.cluster(method="gaussian", show=False)
        conv = np.ones(avg_window**2).reshape(avg_window, -1) / avg_window**2
        lbl1 = signal.convolve2d(lbl1, conv, boundary='symm', mode='same')
        lbl2 = signal.convolve2d(lbl2, conv, boundary='symm', mode='same')
        lbl = ((0.5 * lbl1 + 0.5 * lbl2) > 0.5).astype(int)
        if show:
            fig = plt.figure(figsize=(8,5))
            plt.title("Average method", fontsize=16)
            plt.axis('off')
            plt.imshow(lbl)
        self._avg_mask = lbl
        return lbl

    def apply_mask(self, mask, show=False):
        self.masked_img = self.base_img.copy()
        self.masked_img[mask == 0] = np.array([0, 0, 0])
        if show:
            fig = plt.figure(figsize=(8,5))
            plt.title("Masked image", fontsize=16)
            plt.axis('off')
            plt.imshow(self.masked_img)
        return self.masked_img



    def trimino_material(self, masked_img, method="gaussian", clusters=10, show=False):
        new_img = masked_img.copy()
        self.material = new_img.flatten().reshape(-1, 3)
        idx = np.where(np.linalg.norm(self.material, axis=1) > 0.05)
        label_mask = np.zeros(self.material.shape[0]) - 10
        self.material = self.material[idx]
        self.material = (self.material.T / (np.linalg.norm(self.material, axis=1) + 1e-3)).T

        if method == "gaussian":
            self.gm = GaussianMixture(n_components=clusters, random_state=0).fit(self.material)
            labels = self.gm.predict(self.material)
        elif method == "kmeans":
            self.gm = KMeans(n_clusters=clusters, random_state=0).fit(self.material)
            labels = self.gm.predict(self.material)

        label_mask[idx] = labels * 10
        label_mask = label_mask.reshape(masked_img.shape[0], masked_img.shape[1]).astype(int)

        unique_labels = np.unique(label_mask.flatten())
        count = np.zeros(unique_labels.shape[0]).astype(int)
        for i in range(count.shape[0]):
            count[i] = (label_mask == unique_labels[i]).sum()
        trimino_material = unique_labels[count.argsort()[-2]]
        mask = (label_mask == trimino_material).astype(int)
        new_img[np.where(label_mask != trimino_material)] = np.array([0,0,0])

        if show:
            fig = plt.figure(figsize=(8,5))
            plt.title("Labels", fontsize=16)
            plt.axis('off')
            plt.imshow(new_img)
        
        return new_img, mask



def blur_mask(segment, avg_window=8, bound=0.5):
    conv = np.ones(avg_window**2).reshape(avg_window, -1) / avg_window**2
    blur_w = (signal.convolve2d(segment, conv, boundary='symm', mode='same') > bound).astype(int)
    new_segment = (signal.convolve2d(blur_w, conv, boundary='symm', mode='same') > bound).astype(int)
    return new_segment


class TriangleMetric(object):

    def __init__(self, pattern, true_angle):
        self.main_pattern = pattern
        self.true_angle = true_angle
        self.patterns = []
        self.sums = []
        self.angle_step = 2
        for i in range(0, 180, self.angle_step):
            self.patterns.append(ndimage.rotate(pattern, i, reshape=False).copy())
            self.sums.append(self.patterns[-1].sum())

    def __call__(self, block):
        m = 0
        mi = 0
        for i in range(len(self.patterns)):
            p = (block * self.patterns[i]).sum() / self.sums[i]
            if p > m:
                m = p
                mi = i
        return m, mi
    
    def show_pattern(self, save_file=None):
        p = self.main_pattern.copy()
        p = ndimage.rotate(p, self.true_angle, reshape=False)
        fig = plt.figure(figsize=(8,4))
        plt.axis('off')
        plt.title("Main Pattern", fontsize=16)
        plt.imshow(p)
        plt.show()
        if save_file is not None:
            fig.savefig(save_file)
        return




def find_center(mask):
    x, y = 0, 0
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i,j]:
                y += i
                x += j
    s = mask.sum()
    return (int(y / s), int(x / s))


# ------------- triangle class

class Triangles:

    def __init__(self, image, mask_img, metric, num, sz=160, show=False):
        self.metric = metric
        self.img = image
        mask = mask_img.copy()
        mask = blur_mask(mask)
        match_mx = np.zeros(mask.shape)
        mi_mx = np.zeros(mask.shape).astype(int)
        b = sz // 2
        for i in range(b, mask.shape[0] - b, 3):
            for j in range(b, mask.shape[1] - b, 3):
                match_mx[i,j], mi_mx[i,j] = metric(mask[i-b:i+b,j-b:j+b])

        angles = mi_mx.flatten()  
        patterns = []
        blocks = []
        coords = []
        tr_centers = []
        r_angles = []
        img_blocks = []

        for i in range(num):
            match_mx = match_mx.flatten()
            point = match_mx.argmax()
            y = point // mask.shape[1]
            x = point % mask.shape[1]
            match_mx = match_mx.reshape(mask.shape)
            match_mx[y-b:y+b,x-b:x+b] = 0
            cy, cx = find_center(metric.patterns[angles[point]])

            coords.append([x, y])
            blocks.append(mask_img[y-b:y+b,x-b:x+b])
            patterns.append(angles[point])
            tr_centers.append([x - b + int(cx), y - b + int(cy)])
            r_angles.append(angles[point] * metric.angle_step)
            img_blocks.append(image[y-b:y+b,x-b:x+b])

        self.triangle_centers = tr_centers
        self.block_centers = coords
        self.blocks = blocks
        self.rotate_patterns = patterns
        self.rotate_angles = r_angles
        self.img_blocks = img_blocks

# -------------- triangle class
            

def num_triangles(base_mask, window=13, show=False):
    base_mask[:window] = 0
    base_mask[-window:] = 0
    base_mask[:,:window] = 0
    base_mask[:, -window:] = 0

    cmask = base_mask.copy()
    mask = base_mask.copy()
    mask = blur_mask(mask, avg_window=10)
    for i in range(window, mask.shape[0]-window):
        for j in range(window, mask.shape[1]-window):
            cmask[i,j] = mask[i-window:i+window, j-window:j+window].min()
    
    cmask = blur_mask(cmask, avg_window=8, bound=0.4)
    cmask = blur_mask(cmask, avg_window=8, bound=0.4)
    cmask = blur_mask(cmask, avg_window=7, bound=0.5)
    cmask = blur_mask(cmask, avg_window=7, bound=0.8)

    if show:
        fig = plt.figure(figsize=(8,5))
        plt.title("Centroids", fontsize=16)
        plt.axis('off')
        plt.imshow(cmask)

    def check(p):
        cp = p.copy()
        cp[cp.shape[0] // 2, cp.shape[0] // 2] = 0
        if cp.max() == 1:
            return True
        return False

    w = 20
    for i in range(w, mask.shape[0]-w-1):
        for j in range(w, mask.shape[1]-w-1):
            if cmask[i,j] == 1 and check(cmask[i-w:i+w+1, j-w:j+w+1]):
                cmask[i,j] = 0
    return cmask.sum(), cmask


class ImageReader:

    def __init__(self, nimg):
        data = Data(nimg)
        mask = data.average_alg(avg_window=10, show=False)
        masked_img = data.apply_mask(mask, show=False)
        compress_img, mask = data.trimino_material(masked_img, method="gaussian", clusters=2, show=False)
        self.masked_img = compress_img
        self.mask = mask

    def find_triangles(self, metric, image):
        cm, cmask = num_triangles(self.mask, window=15, show=False)
        self.num_triangles = cm
        self.metric = metric
        self.triangles = Triangles(np.asarray(image), self.mask, metric, num=cm)

    
    def visualize_centroids(self, save_file=None):
        mask = np.zeros(self.mask.shape)
        w = 5
        for i in range(len(self.triangles.triangle_centers)):
            x = self.triangles.block_centers[i][0]
            y = self.triangles.block_centers[i][1]
            mask[y-w:y+w, x-w:x+w] = 1

        fig = plt.figure(figsize=(12, 8))
        plt.axis('off')
        plt.title("Centroids", fontsize=16)
        plt.imshow(mask)
        if save_file is not None:
            fig.savefig(save_file)

    def visualize_mask(self, save_file=None):
        cmask = self.mask.copy()
        cmask = blur_mask(cmask, avg_window=7, bound=0.3)
        fig = plt.figure(figsize=(12, 8))
        plt.axis('off')
        plt.title("Image mask", fontsize=16)
        plt.imshow(cmask)
        if save_file is not None:
            fig.savefig(save_file)




def create_pattern():
	img = Image.open("данные/Pict_2_3.bmp")
	img = normalized_img(img)
	data = Data(img)
	mask = data.average_alg(avg_window=10, show=False)
	masked_img = data.apply_mask(mask, show=False)
	compress_img, mask = data.trimino_material(masked_img, method="gaussian", clusters=2)
	def_window = mask[60:220,257:417]
	noisy_tr = blur_mask(def_window)
	true_tr = noisy_tr.copy()
	true_tr[140:,:] = 0
	true_tr[:,117:] = 0
	return true_tr

 