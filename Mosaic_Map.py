import numpy as np
import math
from osgeo import gdal, osr
import cv2


def load_image(filename):
    image= gdal.Open(filename)
    img_width = image.RasterXSize
    img_height = image.RasterYSize
    img_geotrans = image.GetGeoTransform()
    img_proj= image.GetProjection()
    img_data = image.ReadAsArray(0, 0, img_width, img_height)
    del image
    return img_proj, img_geotrans, img_data


# Tif文件写入
def WriteTifImg(filename, im_proj, im_geotrans, im_data, datatype=None):
    '''功能：用于写TIF格式的遥感图像，同时兼容一个通道 和 三个通道
       返回值：im_proj : 地图投影信息，保持与输入图像相同
             im_geotrans : 仿射矩阵,计算当前图像块的仿射信息
             im_data：通道顺序位 [channel,height,width]， 当前图像块的像素矩阵，
             datatype：指定当前图像数据的数据类型，默认和输入的im_data类型相同'''

    # gdal数据类型包括
    # gdal.GDT_Byte,
    # gdal .GDT_UInt16, gdal.GDT_Int16, gdal.GDT_UInt32, gdal.GDT_Int32,
    # gdal.GDT_Float32, gdal.GDT_Float64

    # 判断栅格数据的数据类型
    if datatype is None:  # im_data.dtype.name数据格式
        if 'int8' in im_data.dtype.name:
            datatype = gdal.GDT_Byte
        elif 'int16' in im_data.dtype.name:
            datatype = gdal.GDT_UInt16
        else:
            datatype = gdal.GDT_Float32

    # 判读数组维数
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape

        # 创建文件
    driver = gdal.GetDriverByName("GTiff")  # 数据类型必须有，因为要计算需要多大内存空间
    dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)

    dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
    dataset.SetProjection(im_proj)  # 写入投影

    if im_bands == 1:
        dataset.GetRasterBand(1).WriteArray(im_data)  # 写入数组数据
    else:
        for i in range(im_bands):  # 按波段写入
            dataset.GetRasterBand(i + 1).WriteArray(im_data[i])

    del dataset

def Mosaic_Map(input_image1, input_image2, d):
    '''
    :param image1: fixed image
    :param image2: moving image
    :param d: the length of mosaic_map's each cell
    :return:the mosaic_map
    '''
    # input_image = [H,W.C]---numpy array
    # copy input image
    image1 = np.copy(input_image1)
    image2 = np.copy(input_image2)
    flag1 = flag2 = False
    if image1.ndim == 2:
        image1 = np.expand_dims(image1, 2)
        flag1 = True
    [m1, n1, p1] = image1.shape
    m11 = math.ceil(m1 / d)
    n11 = math.ceil(n1 / d)
    for i in range(0, m11, 2):
        for j in range(1, n11, 2):
            image1[(i * d + 1):((i + 1) * d + 1), (j * d + 1):((j + 1) * d+1), :] = 0
    for i in range(1, m11, 2):
        for j in range(0, n11, 2):
            image1[(i * d + 1):((i + 1) * d + 1), (j * d + 1):((j + 1) * d+1), :] = 0
    image1 = image1[:m1, :n1, :]


    if image2.ndim == 2:
        image2 = np.expand_dims(image2, 2)
        flag2 = True
    [m2, n2, p2] = image2.shape
    m22 = math.ceil(m2 / d)
    n22 = math.ceil(n2 / d)
    for i in range(0, m22, 2):
        for j in range(0, n22, 2):
            image2[(i * d + 1): ((i + 1) * d+1), (j * d + 1): ((j + 1) * d+1), :] = 0
    for i in range(1, m22, 2):
        for j in range(1, n22, 2):
            image2[(i * d + 1): ((i + 1) * d+1), (j * d + 1): ((j + 1) * d+1), :] = 0
    image2 = image2[:m2, :n2, :]
    image3 = image1 + image2
    if flag1 == flag2 == True:
        image3 = np.squeeze(image3)
    return image3