# 输入：测试集标注文件夹和对应的预测文件夹，图片命名为 标注+x；predict+x
# 输出：四个指标的平均值

from eval_segm import *
import cv2 as cv


def threshold_demo(a):
    gray = cv.cvtColor(a, cv.COLOR_RGB2GRAY)  # 把输入图像灰度化
    # 直接阈值化是对输入的单通道矩阵逐像素进行阈值分割。
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_TRIANGLE)
    print("threshold value %s"%ret)
    return binary


biaozhu_path = ''
predict_path = ''

pa = []
mpa = []
miou = []
fwiou = []

number_photo = 1
for i in range(number_photo):
    i = i+1
    a = cv.imread(biaozhu_path + 'biaozhu' + str(i) + '.png')
    b = cv.imread(predict_path + 'predict' + str(i) + '.png')
    '''
    cv.namedWindow("a", cv.WINDOW_NORMAL)
    cv.imshow("a", a)
    cv.waitKey()
    cv.namedWindow("b", cv.WINDOW_NORMAL)
    cv.imshow("b", b)
    cv.waitKey()
    '''
    binary=threshold_demo(a)
    binary1=threshold_demo(b)
    '''
    cv.namedWindow("binary", cv.WINDOW_NORMAL)
    cv.imshow("binary", binary)
    cv.waitKey()
    cv.namedWindow("binary1", cv.WINDOW_NORMAL)
    cv.imshow("binary1", binary1)
    cv.waitKey()
    '''
    # binary1=cv.resize(binary1,(224,224))
    print(binary.shape)
    print(binary1.shape)

    # 计算分割指标
    pa_temporary=pixel_accuracy(binary,binary1)
    mpa_temporary=mean_accuracy(binary,binary1)
    miou_temporary=mean_IU(binary,binary1)
    fwiou_temporary=frequency_weighted_IU(binary,binary1)

    pa.append(pa_temporary)
    mpa.append(mpa_temporary)
    miou.append(miou_temporary)
    fwiou.append(fwiou_temporary)

print(sum(pa)/number_photo)
print(sum(mpa)/number_photo)
print(sum(miou)/number_photo)
print(sum(fwiou)/number_photo)
