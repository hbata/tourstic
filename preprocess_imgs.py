import numpy as np
import cv2
import PIL
from PIL import Image
import os
from multiprocessing.pool import Pool


def rename_imgs():
    data_dir = 'dataset'
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    folders = [f for f in os.listdir(data_dir) if os.path.isdir(data_dir) and f.endswith('h') or f.endswith('r')]
    print(folders)
    images = {}
    for folder in folders:
        image_files = [ff for ff in os.listdir(data_dir + '/' + folder) if
                       os.path.isfile(os.path.join(data_dir + '/' + folder, ff))]
        for idx, filename in enumerate(image_files):
            f_0 = os.path.splitext(filename)[0]
            f_1 = os.path.splitext(filename)[1]
            new_f = str(idx) + f_1
            os.rename(os.path.join(data_dir + '/' + folder, filename), os.path.join(data_dir + '/' + folder, new_f))


def image_resize(basewidth, input_dir, outdir):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    images = [ff for ff in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, ff))]
    for img_file in images:
        img = Image.open(input_dir + '/' + img_file)
        w_perc = (basewidth / float(img.size[0]))
        hsize = int((float(img.size[1]) * float(w_perc)))
        img = img.resize((basewidth, hsize), PIL.Image.ANTIALIAS)
        img.save(outdir + '/' + img_file)


def image_resize2(img_file, out_dir, basewidth):
    for im in img_file:
        image = Image.open(im)
        w_perc = (basewidth / float(image.size[0]))
        hsize = int((float(image.size[1]) * float(w_perc)))
        # img = image.resize((basewidth, hsize), PIL.Image.ANTIALIAS)
        img = image.resize((basewidth, basewidth))
        name = im.split('/')[-1]
        print(name)
        img.save(out_dir + '/' + name)


def yielder(input_dir, chunk_size=50):
    images = [input_dir + '/' + ff for ff in os.listdir(input_dir) if
              os.path.isfile(os.path.join(input_dir, ff)) and not ff.endswith('.lnk')]
    chunk_end = 0
    f_end = len(images)
    while True:
        chunk_start = chunk_end
        imgs = images[chunk_start:chunk_start + chunk_size]
        chunk_end += chunk_size
        yield imgs
        if chunk_end > f_end:
            break


def parallel_resize(input_dir, out_dir, basewidth):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with Pool(processes=8) as p:
        jobs = []
        for img in yielder(input_dir):
            jobs.append(p.apply_async(image_resize2, (img, out_dir, basewidth)))
        for job in jobs:
            job.get()


if __name__ == '__main__':
    # rename_imgs()
    out_dir1 = 'dataset/petra_resized'
    input_dir1 = 'dataset/khazneh'
    out_dir2 = 'dataset/theater_resized'
    input_dir2 = 'dataset/theater'
    basewidth = 200
    parallel_resize(input_dir1, out_dir1, basewidth)
    parallel_resize(input_dir2, out_dir2, basewidth)
