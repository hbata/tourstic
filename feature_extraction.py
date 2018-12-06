import numpy as np
import cv2
import os as sos
import pandas as pd
import csv
import skimage.io as sio
from skimage.feature import greycomatrix, greycoprops
from multiprocessing.pool import Pool
import matplotlib.pyplot as plt
from tqdm import tqdm
import mahotas as mt


# cv2.getGaborKernel(ksize, sigma, theta, lambda, gamma, psi, ktype)
# ksize - size of gabor filter (n, n)
# sigma - standard deviation of the gaussian function
# theta - orientation of the normal to the parallel stripes
# lambda - wavelength of the sunusoidal factor
# gamma - spatial aspect ratio
# psi - phase offset
# ktype - type and range of values that each pixel in the gabor kernel can hold
def label_images(datapath):
    folders = [f for f in sos.listdir(datapath) if sos.path.isdir(datapath) and f.endswith('_resized')]
    image_paths = []
    for folder in folders:
        if folder.startswith('petra'):
            img_class = 'petra'
        else:
            img_class = 'theater'
        image_files = [datapath + '/' + folder + '/' + ff for ff in sos.listdir(datapath + '/' + folder) if
                       sos.path.isfile(sos.path.join(datapath + '/' + folder, ff))]
        image_paths.append(image_files)
    resultFile = open('labeled_images.csv', 'w')
    wr = csv.DictWriter(resultFile, lineterminator='\n', fieldnames=['id', 'path', 'class'])
    wr.writeheader()
    idx = 0
    for paths in image_paths:
        for img in paths:
            ps = img.split('/')
            im_id = ps[-1].split('.')[0]
            if ps[1].split('_')[0] == 'petra':
                img_class = 'petra'
            else:
                img_class = 'theater'
            wr.writerow({'id': idx, 'path': img, 'class': img_class})
            idx += 1


def read_image(id):
    im_df = pd.read_csv('labeled_images.csv')
    path = im_df['path'][id]
    img = sio.imread(path)
    return img


def read_mask(id):
    im_df = pd.read_csv('labeled_images.csv')
    path = im_df['path'][id]
    img = sio.imread(path)
    return img.astype(np.uint8)


def build_gabor_filter_bank():
    theta = np.arange(0, np.pi, 3 * np.pi / 180)  # range of theta
    omega = np.arange(0.2, 0.8, 0.01)  # range of omega
    params = [(t, o) for o in omega for t in theta]
    ksize = 10
    filters = []
    for (th, og) in params:
        g_params = {'ksize': (ksize, ksize), 'sigma': 7.0, 'theta': th, 'lambd': 10.5,
                    'gamma': og, 'psi': 0, 'ktype': cv2.CV_32F}
        g_kernel = cv2.getGaborKernel(**g_params)
        g_kernel /= 1.5 * g_kernel.sum()
        filters.append((g_kernel, g_params))
    return filters


def gabor_process(img, filters):
    """ returns the img filtered by the filter list
    """
    accum = np.zeros_like(img)
    for kern, params in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        cv2.imshow('filtered image', fimg)
        cv2.imshow('original_img', img)
        np.maximum(accum, fimg, accum)
    return accum


def glcm_props(patch):
    lf = []
    props = ['dissimilarity', 'contrast', 'homogeneity', 'energy', 'correlation']

    # left pixels
    glcm = greycomatrix(image=patch, distances=[2], angles=[0], levels=64, symmetric=False, normed=False)
    for f in props:
        lf.append(greycoprops(glcm, f)[0, 0])

    # upper pixels
    glcm = greycomatrix(patch, [2], [np.pi / 4], 64, symmetric=False, normed=False)
    for f in props:
        lf.append(greycoprops(glcm, f)[0, 0])

    return lf


def patch_gen(img, PAD=0):
    # img1 = (img * 255).astype(np.uint8)

    W = 200
    imgx = np.zeros((200 + PAD * 2, 200 + PAD * 2), dtype=img1.dtype)
    imgx[PAD:W + PAD, PAD:W + PAD] = img1
    imgx[:PAD, PAD:W + PAD] = img1[PAD:0:-1, :]
    imgx[-PAD:, PAD:W + PAD] = img1[W - 1:-PAD - 1:-1, :]
    imgx[:, :PAD] = imgx[:, PAD * 2:PAD:-1]
    imgx[:, -PAD:] = imgx[:, W + PAD - 1:-PAD * 2 - 1:-1]

    xx, yy = np.meshgrid(np.arange(0, W), np.arange(0, W))
    xx, yy = xx.flatten() + PAD, yy.flatten() + PAD

    for x, y in zip(xx, yy):
        patch = imgx[y - PAD:y + PAD + 1, x - PAD:x + PAD + 1]
        yield patch


def glcm_feature(img, verbose=False):
    W, NF, PAD = 200, 10, 0

    if img.sum() == 0:
        return np.zeros((W, W, NF), dtype=np.float32)

    l = []
    with Pool(8) as pool:
        for p in tqdm(pool.imap(glcm_props, patch_gen(img, PAD)), total=W * W, disable=not verbose):
            l.append(p)

    fimg = np.array(l, dtype=np.float32).reshape(200, 200, -1)
    return fimg


def visualize_glcm(imgid):
    img = read_image(imgid)
    mask = read_mask(imgid)

    fimg = glcm_feature(img, verbose=1)
    print(fimg.shape)

    _, (ax0, ax1) = plt.subplots(1, 2, figsize=(6, 3))
    ax0.imshow(img)
    ax1.imshow(mask)
    plt.show()

    amin = np.amin(fimg, axis=(0, 1))
    amax = np.amax(fimg, axis=(0, 1))
    fimg = (fimg - amin) / (amax - amin)

    # fimg[...] = np.power(fimg[...], 3)
    # fimg[..., 9] = np.power(fimg[...], 3)

    _, axs = plt.subplots(2, 5, figsize=(15, 6))
    axs = axs.flatten()

    for k in range(fimg.shape[-1]):
        axs[k].imshow(fimg[..., k])
    plt.show()


def extract_harlik_features(image, image_files):
    image_names = pd.read_csv(image_files, sep=',')
    # image_names = image_names.drop(columns=['id'])
    train_features=[]
    for img_id in image_names['id']:
        img = read_image(img_id)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        textures= mt.features.haralick(gray_img)
        train_features.append(textures)
    return np.asarray(train_features)


# g_kernel = cv2.getGaborKernel((30, 30), 7.0, np.pi / 2, 15.0, np.pi / 20, 0, ktype=cv2.CV_32F)
# img = cv2.imread('dataset/petra_resized/1.jpg')
trainids = pd.read_csv('labeled_images.csv')['id'].tolist()
imid = np.random.choice(trainids)
img = read_image(imid)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
filters = build_gabor_filter_bank()
pp = gabor_process(img, filters)
# label_images('dataset')
textures = extract_harlik_features(img, 'labeled_images.csv')
print(textures[0,:,12])
cv2.imshow('gabor img', pp)
# filtered_img = cv2.filter2D(img, cv2.CV_8UC3, g_kernel)
#
# cv2.imshow('image', img)
# cv2.imshow('filtered image', filtered_img)
#
# h, w = g_kernel.shape[:2]
# g_kernel = cv2.resize(g_kernel, (3 * w, 3 * h), interpolation=cv2.INTER_CUBIC)
# cv2.imshow('gabor kernel (resized)', g_kernel)
cv2.waitKey(0)
cv2.destroyAllWindows()
visualize_glcm(imid)
