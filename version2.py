import _init_paths

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage import io
from timer import Timer
import cv2
from datetime import datetime

import caffe

test_file = 'test.txt'
file_path_img = 'JPEGImages'
file_path_label = 'SegmentationClass'
save_path = 'output/results'

test_prototxt = 'Models/test.prototxt'
weight = 'Training/Seg_iter_10000.caffemodel'

layer = 'conv_seg'
save_dir = False  # True

if save_dir:
    save_dir = save_path
else:
    save_dir = False

# load net
net = caffe.Net(test_prototxt, weight, caffe.TEST)

# load test.txt
test_img = np.loadtxt(test_file, dtype=str)


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


# seg test
print
'>>>', datetime.now(), 'Begin seg tests'

n_cl = net.blobs[layer].channels
hist = np.zeros((n_cl, n_cl))

# timers
_t = {'im_seg': Timer()}

# load image and label
i = 0
for img_name in test_img:
    _t['im_seg'].tic()
    img = Image.open(os.path.join(file_path_img, img_name + '.jpg'))
    img = img.resize((512, 384), Image.ANTIALIAS)

    in_ = np.array(img, dtype=np.float32)
    in_ = in_[:, :, ::-1]  # rgb to bgr
    in_ -= np.array([[[68.2117, 78.2288, 75.4916]]])  # 数据集平均值，根据需要修改
    in_ = in_.transpose((2, 0, 1))

    label = Image.open(os.path.join(file_path_label, img_name + '.png'))
    label = label.resize((512, 384), Image.ANTIALIAS)  # 图像大小（宽，高），根据需要修改
    label = np.array(label, dtype=np.uint8)

    # shape for input (data blob is N x C x H x W), set data
    net.blobs['data'].reshape(1, *in_.shape)
    net.blobs['data'].data[...] = in_

    net.forward()
    _t['im_seg'].toc()

    print
    'im_seg: {:d}/{:d} {:.3f}s' \
        .format(i + 1, len(test_img), _t['im_seg'].average_time)
    i += 1

    hist += fast_hist(label.flatten(), net.blobs[layer].data[0].argmax(0).flatten(), n_cl)

    if save_dir:
        seg = net.blobs[layer].data[0].argmax(axis=0)
        result = np.array(img, dtype=np.uint8)
        index = np.where(seg == 1)
        for i in xrange(len(index[0])):
            result[index[0][i], index[1][i], 0] = 255
            result[index[0][i], index[1][i], 1] = 0
            result[index[0][i], index[1][i], 2] = 0
        result = Image.fromarray(result.astype(np.uint8))
        result.save(os.path.join(save_dir, img_name + '.jpg'))

iter = len(test_img)
# overall accuracy
acc = np.diag(hist).sum() / hist.sum()
print
'>>>', datetime.now(), 'Iteration', iter, 'overall accuracy', acc
# per-class accuracy
acc = np.diag(hist) / hist.sum(1)
print
'>>>', datetime.now(), 'Iteration', iter, 'mean accuracy', np.nanmean(acc)
# per-class IU
iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
print
'>>>', datetime.now(), 'Iteration', iter, 'mean IU', np.nanmean(iu)
freq = hist.sum(1) / hist.sum()
print
'>>>', datetime.now(), 'Iteration', iter, 'fwavacc', \
(freq[freq > 0] * iu[freq > 0]).sum()
