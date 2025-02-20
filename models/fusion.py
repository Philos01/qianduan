import base64
from io import BytesIO

from PIL import Image
from osgeo import gdal
import numpy as np
import cv2
import scipy.ndimage
import math
import datetime
import time

from sync import sync_finish


def write_image(filename, img_proj, img_geotrans, img_data):
    # 判断栅格数据类型
    if 'int8' in img_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in img_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    # 判断数组维度
    if len(img_data.shape) == 3:
        img_bands, img_height, img_width = img_data.shape
    elif len(img_data.shape) == 2:
        (img_height, img_width) = img_data.shape
        img_bands = 1

        # 创建文件
    driver = gdal.GetDriverByName('GTiff')
    image = driver.Create(filename, img_width, img_height, img_bands, datatype)

    image.SetGeoTransform(img_geotrans)
    image.SetProjection(img_proj)

    if img_bands == 1:
        image.GetRasterBand(1).WriteArray(img_data)
    else:
        for i in range(img_bands):
            image.GetRasterBand(i + 1).WriteArray(img_data[i])

    del image  # 删除变量,保留数据


def load_image(filename):
    image = gdal.Open(filename)
    img_width = image.RasterXSize
    img_height = image.RasterYSize
    img_geotrans = image.GetGeoTransform()
    img_proj = image.GetProjection()
    img_data = image.ReadAsArray(0, 0, img_width, img_height)
    del image
    return img_proj, img_geotrans, img_data


def load_panimage(filename):
    image = gdal.Open(filename)
    img_width = image.RasterXSize
    img_height = image.RasterYSize
    img_geotrans = image.GetGeoTransform()
    img_proj = image.GetProjection()
    img_data = image.ReadAsArray(0, 0, img_width, img_height)
    del image
    return img_data

    # 图像向下插值的下采样函数
    # 宽度在前，高度在后


def BCIxcy(img, m, n):
    return cv2.resize(img, (m, n), interpolation=cv2.INTER_AREA)

    # 图像向上插值的上采样函数
    # 宽度在前，高度在后


def BCI(img, m, n):
    return cv2.resize(img, (m, n), interpolation=cv2.INTER_CUBIC)


def inputdataconvert(data, height, width, bands):
    b = np.zeros((height, width, bands))
    for w in range(bands):
        b[:, :, w] = data[w, :, :]
    return b


def outputdataconvert(data, height, width, bands):
    b = np.zeros((bands, height, width))
    for w in range(bands):
        b[w, :, :] = data[:, :, w]
    return b


def hismatch(A, B):
    A_mean = np.mean(np.mean(A, 0))
    B_mean = np.mean(np.mean(B, 0))
    A_std = np.std(A)
    B_std = np.std(B)
    K1 = B_std / A_std
    K2 = B_mean - (K1 * A_mean)
    result = A * K1 + K2
    return result


def regress(y, x):
    a = np.dot(x.T, x)
    b = np.dot(x.T, y)
    return np.dot(np.linalg.pinv(a), b)


def GradientCalculate(image, height, width):
    Hx = np.array([-1, 1]).reshape((1, -1))
    Hy = np.array([-1, 1]).reshape((-1, 1))

    gradientValue = np.zeros((height, width))
    gradientX = scipy.ndimage.convolve(image, Hx, mode='nearest')
    # gradientX = cv2.filter2D(b, -1, Hx)
    gradientY = scipy.ndimage.convolve(image, Hy, mode='nearest')
    gradientX = np.multiply(gradientX, gradientX)
    gradientY = np.multiply(gradientY, gradientY)
    gradientValue = np.sqrt(gradientX + gradientY)

    return gradientValue

    # 公式法实现多元线性回归


def MTFFastSpeSpa(I_MS, sensor, ratio):
    # I_MS_LP = np.zeros((I_MS.shape[0], I_MS.shape[1], I_MS.shape[2]))
    I_MS_LP = np.zeros((I_MS.shape[2], I_MS.shape[0], I_MS.shape[1]))
    I_MS1 = outputdataconvert(I_MS, I_MS.shape[0], I_MS.shape[1], I_MS.shape[2])

    nBands = I_MS.shape[2]
    if sensor == 'QB':
        GNyq = [0.34, 0.32, 0.30, 0.22]
    elif sensor == 'IKONOS':
        GNyq = [0.26, 0.28, 0.29, 0.28]
    elif sensor == 'GeoEye1':
        GNyq = [0.23, 0.23, 0.23, 0.23]
    elif sensor == 'none':
        GNyq = [0.29] * nBands
    N = 31
    fcut = 1 / ratio

    for i in range(nBands):
        alpha = math.sqrt((N * (fcut / 2)) ** 2 / (-2 * math.log(GNyq[i])))
        H = np.multiply(cv2.getGaussianKernel(N, alpha), (cv2.getGaussianKernel(N, alpha)).T)
        np.set_printoptions(precision=4)
        Hd = H / H.max()
        kernel = np.kaiser(N, 0.5)
        h = fwind1(Hd, kernel, N)
        # I_MS_LP[:, :, i] = cv2.filter2D(I_MS[:, :, i].astype('float32'), -1, np.real(fwindResult),
        #                                 borderType=cv2.BORDER_REPLICATE)
        I_MS_LP[i, :, :] = cv2.filter2D(I_MS1[i, :, :].astype('float32'), -1, np.real(h),
                                        borderType=cv2.BORDER_REPLICATE)
    I_MS_LP1 = inputdataconvert(I_MS_LP, I_MS_LP.shape[1], I_MS_LP.shape[2], I_MS_LP.shape[0])
    I_Filtered = I_MS_LP1
    return I_Filtered


def fwind1(Hd, kernel, N):
    n = N
    t = [0 for i in range(n)]
    t1 = [[0 for i in range(n)] for i in range(n)]
    t2 = [[0 for i in range(n)] for i in range(n)]
    k = [0 for i in range(n * n)]
    t12 = [0 for i in range(n * n)]
    out = [[0 for i in range(n)] for i in range(n)]
    outMat_fftshift = np.zeros((N, N))
    outMat_fftshiftRot = np.zeros((N, N))
    outMat_fftshiftRot_ifft = np.zeros((N, N))
    outMat_final = np.zeros((N, N))
    outMat_finalRot = np.zeros((N, N))
    fwindResult = np.zeros((N, N))
    for i in range(n):
        t[i] = float(-(n - 1) / 2 + i) * 2 / float(n - 1)

    for i in range(n):
        for j in range(n):
            t1[i][j] = t[j]
            t2[i][j] = t[i]
    for i in range(n):
        for j in range(n):
            t12[i * N + j] = math.sqrt(t1[i][j] * t1[i][j] + t2[i][j] * t2[i][j])
            if t12[i * N + j] < t[0] or t12[i * N + j] > t[n - 1]:
                t12[i * N + j] = 0
    hstep = t[2] - t[1]
    for i in range(n):
        for j in range(n):
            k[i * N + j] = 1 + math.floor((t12[i * N + j] - t[0]) / hstep)
            if k[i * N + j] < 0:
                k[i * N + j] = 0
            elif k[i * N + j] > N - 2:
                k[i * N + j] = N - 2
    for i in range(n):
        for j in range(n):
            kflag = int(k[i * N + j])
            s = float((t12[i * N + j] - t[kflag]) / hstep)
            out[i][j] = kernel[kflag] + s * (kernel[kflag + 1] - kernel[kflag])
            if (i != ((N - 1) / 2) and j != ((N - 1) / 2) and out[i][j] >= 1):
                out[i][j] = 0
    outMat = np.rot90(Hd, 2)
    outMat_fftshift = np.fft.fftshift(outMat)
    outMat_fftshiftRot = np.rot90(outMat_fftshift, 2)
    outMat_fftshiftRot_ifft = np.fft.ifft2(outMat_fftshiftRot)
    outMat_final = np.fft.fftshift(outMat_fftshiftRot_ifft)
    if all(np.nanmax(abs(np.imag(outMat_final)), axis=1) < math.sqrt(np.spacing(1))):
        outMat_final = np.real(outMat_final)
    outMat_finalRot = np.rot90(outMat_final, 2)

    fwindResult = np.multiply(outMat_finalRot, out)
    return fwindResult


def findindices_pan_ms(imagepath1, imagepath2):
    # proj0, geotrans0, hs_original = UpdateThread.load_image(imagepath0)  # hs
    proj, geotrans, ms = load_image(imagepath1)  # ms
    ms = ms.astype(np.float32)
    proj1, geotrans1, pan = load_image(imagepath2)  # pan

    # bands0, height0, width0 = hs_original.shape  # hs_original
    bands1, height1, width1 = ms.shape  # ms
    width0 = round(width1 / 1)
    height0 = round(height1 / 1)
    height2, width2 = pan.shape  # pa

    # 下采样
    hs1 = np.zeros((bands1, height0, width0))
    for w in range(bands1):
        hs1[w, :, :] = BCIxcy(ms[w, :, :], width0, height0)
    LPANdown = BCIxcy(pan, width0, height0)
    d = hs1.mean(0)
    LPANdown = hismatch(LPANdown, d)  # 直方图归匹配后的pan

    for i in range(bands1):
        tmp = np.array(hs1[i, :, :]).reshape((height0 * width0, 1), order='F')
        if i == 0:
            res = tmp
        else:
            res = np.column_stack((res, tmp))
    x1 = res
    y1 = np.array(LPANdown).reshape((-1, 1), order='F')
    cc = regress(y1, x1)
    del hs1, LPANdown, tmp, res, d
    return cc, proj1, geotrans1


def MTFFastSpeSpa_band(I_MS, sensor, ratio):
    GNyq = [0.29]
    N = 31
    fcut = 1 / ratio

    alpha = math.sqrt((N * (fcut / 2)) ** 2 / (-2 * math.log(GNyq[0])))
    H = np.multiply(cv2.getGaussianKernel(N, alpha), (cv2.getGaussianKernel(N, alpha)).T)
    np.set_printoptions(precision=4)
    Hd = H / H.max()
    kernel = np.kaiser(N, 0.5)
    h = fwind1(Hd, kernel, N)
    I_MS_LP = cv2.filter2D(I_MS.astype('float32'), -1, np.real(h), borderType=cv2.BORDER_REPLICATE)
    return I_MS_LP


def MS_data_process(ms_path, pan_path, cc):
    proj, geotrans, ms = load_image(ms_path)
    ms = ms.astype(np.float32)
    proj2, geotrans2, pan = load_image(pan_path)
    bands, height_ms, width_ms = ms.shape
    height_pan, width_pan = pan.shape
    del pan
    gradientMulti = np.ones((bands, 1), dtype=np.int16)
    sum1 = np.ones((height_ms, width_ms), dtype=np.int16)
    I = np.ones((height_ms, width_ms), dtype=np.int16)
    for i in range(bands):
        tmp = ms[i, :, :]
        sum1 = sum1 + tmp
        I = I + cc[i] * tmp
        gradientMultiMatrix = GradientCalculate(ms[i, :, :], height_ms, width_ms)
        gradientMulti[i] = np.sum(np.sum(gradientMultiMatrix)) / (height_ms * width_ms)
    d = sum1 / bands
    d = BCI(d, width_pan, height_pan)
    I = BCI(I, width_pan, height_pan)
    gradientI = GradientCalculate(ms.mean(0), height_ms, width_ms)
    gradientI = np.sum(np.sum(gradientI)) / (height_ms * width_ms)
    WeightDist = gradientMulti / gradientI
    del ms, sum1, tmp, gradientMultiMatrix, gradientI
    return d, I, WeightDist, bands, height_pan, width_pan


def PAN_data_process(pan_path, d, I):
    proj, geotrans, pan = load_image(pan_path)
    pan1 = hismatch(pan, d)
    pan_matched = hismatch(pan1, I)
    del pan1
    panFlag_matched = hismatch(pan, I)
    EdgePAN = pan_matched - I
    del pan_matched, pan
    kernel = np.array((
        [-0.1667, -0.6667, -0.1667],
        [-0.6667, 4.3333, -0.6667],
        [-0.1667, -0.6667, -0.1667]), dtype="float32")  # 定义滤波算子
    EdgePAN_Sharp = cv2.filter2D(panFlag_matched, -1, kernel, borderType=cv2.BORDER_REPLICATE)
    del panFlag_matched
    return EdgePAN.astype(np.int16), EdgePAN_Sharp.astype(np.int16)


def SpectralAdjust_band(i, gd_raster, EdgePAN, EdgePAN_Sharp, multilow, final_file_path, WeightDist, lamda):
    sensor = 'none'
    multilow_height = multilow.shape[0]
    multilow_width = multilow.shape[1]
    pan_height = EdgePAN.shape[0]
    pan_width = EdgePAN.shape[1]
    ratio = ((pan_height / multilow_height + pan_width / multilow_width) / 2)

    # 上采样
    multi1 = BCI(multilow, pan_width, pan_height)
    fused = multi1 + WeightDist[i] * (EdgePAN + lamda * (EdgePAN_Sharp - EdgePAN))
    del multilow, EdgePAN, EdgePAN_Sharp, multi1
    print('第' + str(i) + '个：' + datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S'))
    gd_raster.GetRasterBand(i + 1).WriteArray(fused)
    del fused

    # del multi1, MTFresult, DownResult, DifferImage, DifferImageBlur, DifferImageBlurUP


def SARF_MsPan_Fuse(job_id, pan_dir, ms_dir, save_dir, file_name, lamda0):
    # proj0, geotrans0, pan = load_image(pan_image_path)
    # proj, geotrans, ms = load_image(ms_image_path)
    # ms = ms.astype(np.float32)
    # imageHR = pan.astype(np.float32)
    #
    # lamda2 = 0.3
    # Outdata1 = inputdataconvert(ms, ms.shape[1], ms.shape[2], ms.shape[0])
    # Out2 = Test_FastSpaSpePansharp_realTest20161031_Test_V2_4_HP(Outdata1, imageHR, lamda2)
    # Outdata2 = outputdataconvert(Out2, Out2.shape[0], Out2.shape[1], Out2.shape[2])
    # write_image(patch_save + file_name, proj0, geotrans0, Outdata2)

    # HS:img_name2
    # MS:img_name1
    # PAN:img_name
    # 保存地址+文件：filename
    # image_path3：hs_patch
    # image_path2:ms_patch
    # image_path1:pan_patch
    # image_path4://gf3//
    # image_path5://tmp_file//
    # lamda
    lamda = lamda0
    # img_name = pan_dir
    # img_name1 = ms_dir
    filename = save_dir + file_name
    time_start = time.time()  # 融合程序计时
    print('开始时间：' + datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S'))
    # image_path6 = img_name1
    cc, proj0, geotrans0 = findindices_pan_ms(ms_dir, pan_dir)
    print('找到系数：' + datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S'))
    d, I, WeightDist, bands, height_pan, width_pan = MS_data_process(ms_dir, pan_dir, cc)
    print('MS：' + datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S'))
    EdgePAN, EdgePAN_Sharp = PAN_data_process(pan_dir, d, I)
    print('PAN：' + datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S'))
    proj, geotrans, ms = load_image(ms_dir)  # ms
    ms = ms.astype(np.int16)
    gd_driver = gdal.GetDriverByName('MEM')
    gd_raster = gd_driver.Create('pippo', width_pan, height_pan, bands, gdal.GDT_UInt16)
    gd_raster.SetGeoTransform(geotrans0)
    gd_raster.SetProjection(proj0)
    for i in range(bands):
        # self.pool.submit(UpdateThread.SpectralAdjust_band, i, gd_raster, EdgePAN, EdgePAN_Sharp, HS_MS_File[i, :, :], self.filename, WeightDist, self.lamda)
        SpectralAdjust_band(i, gd_raster, EdgePAN, EdgePAN_Sharp, ms[i, :, :], filename, WeightDist, lamda)
    # self.pool.shutdown(True)
    print('融合结束：' + datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S'))
    dset_tiff_out = gdal.GetDriverByName('GTiff')
    dset_tiff_out.CreateCopy(filename, gd_raster)

    print('输出文件：' + datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S'))
    time_over = time.time()
    time_during = time_over - time_start
    print('共计用时：' + str(time_during))
    _, _, final_result = load_image(filename)
    _, _, ms_show = load_image(ms_dir)
    _, _, pan_show = load_image(pan_dir)

    def image_to_base64(img):
        buffered = BytesIO()
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # 将 OpenCV 图像转为 PIL 格式
        img_pil.save(buffered, format="JPEG")  # 或者 'PNG'，根据需要
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def image_to_base64_ms(img):
        buffered = BytesIO()
        img_pil = Image.fromarray(img)  # 从 RGB NumPy 数组创建 PIL 图像
        img_pil.save(buffered, format="JPEG")  # 保存为 JPEG 格式
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    msre = image_to_base64_ms((ms_show[:3, :, :].transpose(1, 2, 0)).astype(np.uint8))
    panre = image_to_base64(pan_show.astype(np.uint8))
    fusion_result = image_to_base64_ms((final_result[:3, :, :].transpose(1, 2, 0)).astype(np.uint8))
    annotations = [{
        "imagere1": msre,
        "imagere2": panre,
        "resultimage2": fusion_result}
    ]
    # print(job_id, "cal finished!", str(annotations))
    sync_finish(job_id, annotations)
    return annotations, 200


if __name__ == "__main__":
    # 分波段融合
    # 大
    # pan_image_path = 'F:\\2-DataSet\\202404尖兵项目课题二稀疏目标变化事件的遥感时空稳健检测技术第一批标注样本（脱密变形后）\\水源水体污染\\01蓝藻水华样本\\样本对\\影像-20221110-ZY1F-020\\原始数据\\TEST\\PAN_r1c1.tif'
    #
    # ms_image_path = 'F:\\2-DataSet\\202404尖兵项目课题二稀疏目标变化事件的遥感时空稳健检测技术第一批标注样本（脱密变形后）\\水源水体污染\\01蓝藻水华样本\\样本对\\影像-20221110-ZY1F-020\\原始数据\\TEST\\MUX_r1c1.tif'
    #
    # patch_save = 'F:\\2-DataSet\\202404尖兵项目课题二稀疏目标变化事件的遥感时空稳健检测技术第一批标注样本（脱密变形后）\\水源水体污染\\01蓝藻水华样本\\样本对\\影像-20221110-ZY1F-020\\原始数据\\'

    # 中
    # pan_image_path = 'D:\\project_file\\data2\\data2_zero\\onepatch\\pan_r3c4.tif'
    # ms_image_path = 'D:\\project_file\\data2\\data2_zero\\onepatch\\ms_r3c4.tif'
    # patch_save = 'F:\\2-DataSet\\test\\'
    # 小
    pan_image_path = r"E:\file_zxt\geo_client\geo_client\mspanfusion-pairs\PAN-GF1.tif"
    ms_image_path = r"E:\file_zxt\geo_client\geo_client\mspanfusion-pairs\GF1.tif"
    patch_save = r'E:\file_zxt\geo_client\geo_client\mspanfusion-pairs'
    # #输出图象
    # pan_image_path = 'F:\\2-DataSet\\尖兵数据\\样例数据\\土地淹没\\GF1D_PMS_E121.3_N30.2_20220728_L1A1257099461\\GF1D_PMS_E121.3_N30.2_20220728_L1A1257099461-PAN.tiff'
    # ms_image_path =  'F:\\2-DataSet\\尖兵数据\\样例数据\\土地淹没\\GF1D_PMS_E121.3_N30.2_20220728_L1A1257099461\\GF1D_PMS_E121.3_N30.2_20220728_L1A1257099461-MUX.tiff'
    # patch_save = 'F:\\2-DataSet\\尖兵数据\\样例数据\\土地淹没\\'
    # 4波段 3万
    # 4波段 3万
    # pan_image_path = 'F:\\2-DataSet\\尖兵数据\\样例数据\\岱山县建筑\\北京二号TRIPLESAT_3_PMS_L1_20211123013545_0039E9VI_003_0120210726004001_069\\PAN\\TRIPLESAT_3_PAN_L1_20211123013545_0039E9VI_003_0120210726004001_069.tif'
    # ms_image_path = 'F:\\2-DataSet\\尖兵数据\\样例数据\\岱山县建筑\\北京二号TRIPLESAT_3_PMS_L1_20211123013545_0039E9VI_003_0120210726004001_069\\MS\\TRIPLESAT_3_MS_L1_20211123013545_0039E9VI_003_0120210726004001_069.tif'
    # patch_save = 'F:\\2-DataSet\\尖兵数据\\样例数据\\岱山县建筑\\'
    # gf
    # pan_image_path = 'F:\\2-DataSet\\20230707第一批样本\\耕地破坏\\0.原始\\中分\\GF1B_PMS_E120.8_N29.7_20230126_L1A1228247571\\GF1B_PMS_E120.8_N29.7_20230126_L1A1228247571-PAN.tiff'
    # ms_image_path = 'F:\\2-DataSet\\20230707第一批样本\\耕地破坏\\0.原始\\中分\\GF1B_PMS_E120.8_N29.7_20230126_L1A1228247571\\GF1B_PMS_E120.8_N29.7_20230126_L1A1228247571-MUX.tiff'
    # patch_save = 'F:\\2-DataSet\\20230707第一批样本\\耕地破坏\\中分\\'
    # test
    # pan_image_path = 'F:\\2-DataSet\\尖兵数据\\样例数据\\岱山县建筑\\模拟实验\\pan_down.tif'
    # ms_image_path = 'F:\\2-DataSet\\尖兵数据\\样例数据\\岱山县建筑\\模拟实验\\ms_down.tif'
    # patch_save = 'F:\\2-DataSet\\尖兵数据\\样例数据\\岱山县建筑\\模拟实验\\'

    # pan_image_path = 'F:\\2-DataSet\\20230707第一批样本\\耕地破坏\\中分\\pan_rad.tif'
    # ms_image_path = 'F:\\2-DataSet\\20230707第一批样本\\耕地破坏\\中分\\ms_rad.tif'
    # patch_save = 'F:\\2-DataSet\\20230707第一批样本\\耕地破坏\\中分\\'
    # 8波段
    # pan_image_path = 'F:\\2-DataSet\\尖兵数据\\样例数据\\越城水\\多光谱融合测试\\PAN_r1c1.tif'
    # ms_image_path = 'F:\\2-DataSet\\尖兵数据\\样例数据\\越城水\\多光谱融合测试\\MUX_r1c1.tif'
    # patch_save = 'F:\\2-DataSet\\尖兵数据\\样例数据\\越城水\\多光谱融合测试\\'

    # pan_image_path = 'D:\\2-DataSet\\尖兵数据\\样例数据\\耕地破坏\\CB04A_WPM_E120.5_N30.6_20220504_L1A0000303013\\CB04A_WPM_E120.5_N30.6_20220504_L1A0000303013-PAN.tiff'
    # ms_image_path = 'D:\\2-DataSet\\尖兵数据\\样例数据\\耕地破坏\\CB04A_WPM_E120.5_N30.6_20220504_L1A0000303013\\CB04A_WPM_E120.5_N30.6_20220504_L1A0000303013-MSS.tiff'
    # patch_save = 'F:\\'

    file_name = '\sarf_test.tif'
    lamda = 0.3
    SARF_MsPan_Fuse(pan_image_path, ms_image_path, patch_save, file_name, lamda)
