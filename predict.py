import numpy as np
from preprocess_imgs import parallel_resize
import os
import csv
from sklearn.metrics import classification_report, confusion_matrix
from feature_extract import import_images, calculate_local_binary_patterns, lbp_enery_entropy, compute_glcm_features, \
    gabor_features, gabor_energy_entropy

import pickle as pkl
import pandas as pd
import matplotlib.pyplot as plt
import itertools


def label_images(datapath):
    folders = [f for f in os.listdir(datapath) if os.path.isdir(datapath) and f.endswith('_resized')]
    image_paths = []
    for folder in folders:
        if folder.startswith('petra'):
            img_class = 'petra'
        else:
            img_class = 'theater'
        image_files = [datapath + '/' + folder + '/' + ff for ff in os.listdir(datapath + '/' + folder) if
                       os.path.isfile(os.path.join(datapath + '/' + folder, ff))]
        # print(image_files)
        image_paths.append(image_files)
    resultFile = open(datapath + '/labeled_test_images.csv', 'w')
    wr = csv.DictWriter(resultFile, lineterminator='\n', fieldnames=['id', 'path', 'class'])
    wr.writeheader()
    idx = 0
    for paths in image_paths:
        for img in paths:
            ps = img.split('/')
            print(ps)
            im_id = ps[-1].split('.')[0]
            if ps[2].split('_')[0] == 'petra':
                img_class = 'petra'
            else:
                img_class = 'theater'
            wr.writerow({'id': idx, 'path': img, 'class': img_class})
            idx += 1


def main(resize=False, label=False):
    test_dir = 'dataset/test_set'
    # out_dir1 = 'dataset/test_set/petra_resized'
    # out_dir2 = 'dataset/test_set/theater_resized'
    basewidth = 200
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    test_folders = [f for f in os.listdir(test_dir) if os.path.isdir(test_dir) and f.endswith('a') or f.endswith('r')]

    if resize:
        for folder in test_folders:
            input_dir = test_dir + '/' + folder
            out_dir = test_dir + '/' + folder + '_resized'
            parallel_resize(input_dir, out_dir, basewidth)

    if label:
        label_images(test_dir)

    theta = np.arange(0, np.pi / 2, 3 * np.pi / 180)  # range of theta
    omega = np.arange(0.2, 0.8, 0.1)  # range of omega
    trainids = pd.read_csv(test_dir + '/labeled_test_images.csv')['id'].tolist()
    images, img_labels = import_images(test_dir + '/labeled_test_images.csv')
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
    with open(test_dir + '/features_test_mat.pkl', 'wb') as fm:
        pkl.dump(data_mat, fm)
    feature_labels = np.array(feature_labels)
    with open(test_dir + '/labels_test_mat.pkl', 'wb') as fl:
        pkl.dump(feature_labels, fl)


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def classification_report_csv(report):
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-3]:
        row = {}
        row_data = line.split('      ')
        row['class'] = row_data[0]
        row['precision'] = float(row_data[1])
        row['recall'] = float(row_data[2])
        row['f1_score'] = float(row_data[3])
        row['support'] = float(row_data[4])
        report_data.append(row)
    dataframe = pd.DataFrame.from_dict(report_data)
    dataframe.to_csv('model/classification_report.csv', index=False)


if __name__ == '__main__':
    # main()

    with open('model/svm_slf.pkl', 'rb') as f:
        svm_clf = pkl.load(f)

    with open('dataset/test_set/labels_test_mat.pkl', 'rb') as f:
        y_true = pkl.load(f)

    with open('dataset/test_set/features_test_mat.pkl', 'rb') as f:
        test_features = pkl.load(f)

    print(y_true.shape)
    print(test_features.shape)

    y_pred = svm_clf.predict(test_features)
    class_labels = ['theater', 'petra']

    classification_dict = classification_report(y_true, y_pred)
    print(classification_dict)
    classification_report_csv(classification_dict)

    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure('un-normalized_conf_matrix')
    plot_confusion_matrix(conf_matrix, classes=class_labels,
                          title='Confusion matrix, without normalization')
    plt.savefig('model/un-normalized_conf_matrix.png')
    plt.figure('normalized_conf_matrix')
    plot_confusion_matrix(conf_matrix, classes=class_labels, normalize=True,
                          title='Normalized confusion matrix')
    plt.savefig('model/normalized_conf_matrix.png')
    plt.show()
