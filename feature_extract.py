from PIL import Image
from skimage.feature import local_binary_pattern, greycomatrix, greycoprops
from skimage.filters import gabor
from skimage import io as sio
from skimage import img_as_ubyte
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import pandas as pd
import cv2


def import_images(input_dir):
    labeled_images = pd.read_csv(input_dir, sep=',')
    image_paths = labeled_images['path']
    images = []
    labels = []
    for i, path in enumerate(image_paths):
        img = sio.imread(path, as_gray=True)
        label = labeled_images['class'][i]
        if label.upper().lower() == 'petra':
            n_label = 1
        else:
            n_label = 0
        images.append(img)
        labels.append(n_label)
    return images, labels


def calculate_local_binary_patterns(image_arr, P, R, method):
    feat_lbp = local_binary_pattern(image_arr, P, R, method=method)
    feat_lbp = np.uint8((feat_lbp / feat_lbp.max()) * 255)
    lbp_image = Image.fromarray(feat_lbp)
    return lbp_image, feat_lbp


def lbp_enery_entropy(feat_lbp, n_bins):
    lbp_hist, _ = np.histogram(feat_lbp, n_bins)
    lbp_hist = np.array(lbp_hist, dtype=np.float)
    lbp_prob = np.divide(lbp_hist, np.sum(lbp_hist))
    lbp_energy = np.sum(lbp_prob ** 2)
    lbp_entropy = -np.sum(np.multiply(lbp_prob, np.log2(lbp_prob)))
    return lbp_energy, lbp_entropy


def compute_glcm_features(img_arr, distances, angles, levels, symmetric=True, normed=True,
                          props=['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']):
    # img_arr=np.uint8(img_arr)
    img_arr = img_as_ubyte(img_arr)
    glcm_mat = greycomatrix(img_arr, distances, angles, levels=levels, symmetric=symmetric, normed=normed)
    contrast = greycoprops(glcm_mat, prop=props[0])
    dissimilarity = greycoprops(glcm_mat, prop=props[1])
    homogeneity = greycoprops(glcm_mat, prop=props[2])
    energy = greycoprops(glcm_mat, prop=props[3])
    correlation = greycoprops(glcm_mat, prop=props[4])
    return contrast, dissimilarity, homogeneity, energy, correlation


def gabor_features(img_arr, frequency, theta, bandwidth, sigma=None):
    gabor_parts = []
    gabor_ampl = []
    for (th, f) in zip(theta, bandwidth):
        gabor_real, gabor_imag = gabor(img_arr, frequency, th, f)
        gabor_parts.append((gabor_real, gabor_imag))
        gabor_filt = np.sqrt((gabor_real ** 2 + gabor_imag ** 2))
        gabor_ampl.append(gabor_filt)
    return np.array(gabor_ampl), np.array(gabor_parts)


def gabor_energy_entropy(gabor_filt, n_bins):
    gabor_hist, _ = np.histogram(gabor_filt, n_bins)
    gabor_hist = np.array(gabor_hist, dtype=np.float)
    gabor_prob = np.divide(gabor_hist, np.sum(gabor_hist))
    gabor_energy = np.sum(gabor_prob ** 2)
    gabor_entropy = -np.sum(np.multiply(gabor_prob, np.log2(gabor_prob)))
    return gabor_energy, gabor_entropy


def build_gabor_filter_bank():
    theta = np.arange(0, np.pi / 2, 3 * np.pi / 180)  # range of theta
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


def main():
    theta = np.arange(0, np.pi / 2, 3 * np.pi / 180)  # range of theta
    omega = np.arange(0.2, 0.8, 0.1)  # range of omega
    trainids = pd.read_csv('labeled_images.csv')['id'].tolist()
    images, img_labels = import_images('labeled_images.csv')
    idd = np.random.choice(trainids)
    lbp_img, feat_lbp = calculate_local_binary_patterns(np.array(images[idd]), 8, 1, 'uniform')
    feature_labels = []
    data_mat = np.zeros(shape=(len(images), 9))
    for i, img in enumerate(images):
        print(np.shape(img))
        print(img_labels[i])
        feature_labels.insert(i, img_labels[i])
        lbp_image, feat_lbp = calculate_local_binary_patterns(img, 8, 1, 'uniform')
        feat_lbp = feat_lbp.reshape(img.shape[0] * img.shape[1])
        lbp_energy, lbp_entropy = lbp_enery_entropy(feat_lbp, 8)
        contrast, dissimilarity, homogeneity, energy, correlation = compute_glcm_features(np.array(img), [2], [0], 256)
        feat_glcm = np.array([contrast[0][0], dissimilarity[0][0], homogeneity[0][0], energy[0][0], correlation[0][0]])
        print('feat_glcm:', feat_glcm.shape)
        gabor_ampl, gabor_parts = gabor_features(img, 0.6, theta, omega)
        gabor_energies = []
        gabor_entropies = []

        for i in range(gabor_ampl.shape[0]):
            gabor_energy, gabor_entropy = gabor_energy_entropy(gabor_ampl[i], 8)
            gabor_energies.append(gabor_energy)
            gabor_entropies.append(gabor_entropy)
        gabor_energies = np.array(gabor_energies)
        gabor_entropies = np.array(gabor_entropies)
        all_features = np.concatenate(
            ([lbp_energy, lbp_entropy], feat_glcm, [np.mean(gabor_energies), np.mean(gabor_entropies)]), axis=0)
        data_mat[i, :] = all_features
    with open('dataset/features_mat.pkl', 'wb') as fm:
        pkl.dump(data_mat, fm)
    feature_labels = np.array(feature_labels)
    with open('dataset/labels_mat.pkl', 'wb') as fl:
        pkl.dump(feature_labels, fl)

    # with open('dataset/features_mat.pkl','rb') as f:
    #     features=pkl.load(f)
    #     print(np.shape(features[2]))

    lbp_energy, lbp_entropy = lbp_enery_entropy(feat_lbp, 8)
    print(img_labels[idd])
    print('LBP energy = ' + str(lbp_energy))
    print('LBP entropy = ' + str(lbp_entropy))
    contrast, dissimilarity, homogeneity, energy, correlation = compute_glcm_features(np.array(images[idd]), [2], [0],
                                                                                      256)
    print('Contrast = ' + str(contrast[0][0]))
    print('Dissimilarity = ' + str(dissimilarity[0][0]))
    print('Homogeneity = ' + str(homogeneity[0][0]))
    print('Energy = ' + str(energy[0][0]))
    print('Correlation = ' + str(correlation[0][0]))
    # theta = np.arange(0, np.pi / 2, 3 * np.pi / 180)  # range of theta
    # omega = np.arange(0.2, 0.8, 0.1)  # range of omega
    gabor_ampl, gabor_parts = gabor_features(images[idd], 0.6, theta, omega)
    print(np.shape(gabor_ampl), np.shape(gabor_parts))
    gabor_energy, gabor_entropy = gabor_energy_entropy(gabor_ampl[3], 8)
    print(gabor_entropy.shape)
    plt.figure('gabor')
    plt.imshow(gabor_ampl[3])
    print('Gabor energy = ' + str(gabor_energy))
    print('Gabor entropy = ' + str(gabor_entropy))
    plt.figure('lbp')
    plt.imshow(lbp_img, cmap='gray')

    # plt.imshow(images[idd])
    # print(labels[idd])
    plt.show()


if __name__ == '__main__':
    main()
