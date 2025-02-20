from io import BytesIO
from time import sleep
import time
import base64
import cv2
import numpy as np
from PIL import Image
from flask import jsonify

from sync import sync_progress, sync_finish
from A_main_Registration_matlab import MS_main
from Mosaic_Map import load_image, WriteTifImg


def image_to_base64(img):
    buffered = BytesIO()
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # 将 OpenCV 图像转为 PIL 格式
    img_pil.save(buffered, format="JPEG")  # 或者 'PNG'，根据需要
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def project1(job_id, type, img1, img2, GroundTruth1):
    # 如果没有GroundTruth1为空，则不需要输出metric
    metric = {"PSNR": 0.2, "RMSE": 0.3}

    result_pic1 = []  # 中间图像数据块或者图像对象，需要传给课题二使用
    result_pic2 = []  # 中间图像数据块或者图像对象，需要传给课题二使用
    return metric, result_pic1, result_pic2


def project1_registration(job_id, type, img1, img2, outputfile):
    result_pic1, result_pic2 = MS_main(img1, img2, outputfile)
    return result_pic1, result_pic2


def project2(job_id, pic_data1, pic_data2, GroundTruth2):
    # 如果没有GroundTruth2为空，则不需要输出accuracy
    accuracy = 0.5

    polygons = [
        [[119.53451750547801, 29.002001576760478],
         [119.6234646683748, 29.10359343672083],
         [119.88692284444929, 29.094559775595087]],
        [[120.00987921668896, 28.929837083637224],
         [119.87086154255178, 28.746504436555412],
         [119.63225625022785, 28.7523046280036],
         [119.53451750547801, 29.002001576760478]]
    ]
    return polygons, accuracy


def load_img(input_file1, input_file2):
    if input_file1.endswith('.tif' or 'tiff') and input_file2.endswith('.tif' or 'tiff'):
        # 读取图像
        img_proj1, img_geotrans1, image_1 = load_image(input_file1)
        img_proj2, img_geotrans2, image_2 = load_image(input_file2)
        if image_1.ndim == 3:
            image_1 = np.transpose(image_1, (1, 2, 0))
            image_1 = np.ascontiguousarray(image_1, dtype=np.float32)
        if image_2.ndim == 3:
            image_2 = np.transpose(image_2, (1, 2, 0))
            image_2 = np.ascontiguousarray(image_2, dtype=np.float32)
        tif_flag = True
    else:
        image_1 = cv2.imread(input_file1)
        image_2 = cv2.imread(input_file2)  # 或者新建一个图像对象
    return image_1, image_2


# 线上不允许写图，调试的时候可以测试用一下，线上得去掉,线上不要写图！！！
def write_image(img, filepath):
    return "write " + filepath + " finished!"


# 线上不允许写图，调试的时候可以测试用一下，线上得去掉,线上不要写图！！！
def write_image(pic1, polygons, filepath):
    return "write " + filepath + " finished!"


def demo_func(job_id, type, message_list, GroundTruth1, GroundTruth2, otherInfo):
    '''
    这是一份标准的计算流程，使用sleep来模拟长时间的计算
    :param job_id: 作业标识
    :param type: 任务类型 example:"耕地破坏"
    :param message_list: 卫星图片信息列表，exampel:"list": [{"pan": "","mux": """SAR": "","VN": "","SW": "",},{"pan": "","mux": ""}],"Ground truth1": "This is the ground truth 1","Ground truth2": "This is the ground truth 2",
"otherInfo":{"captureTime": …, "resolution":… , "source":… }}
    :return: 本函数不需要return，请在运算过程通过sync包中的同步函数与服务端进行同步

    同步进度使用函数：sync_progress 位于sync __init.py__
    同步结果使用函数：sync_finish 位于sync __init.py__

    '''
    print(job_id, type, message_list, GroundTruth1 is None, GroundTruth2 is None, otherInfo is None)
    start_time = time.time()
    # 读图部分，主要修改部分！！
    # img1 = load_image(message_list[0]['pan'])
    # img2 = load_image(message_list[0]['mux'])
    img1 = message_list[0]
    img2 = message_list[1]
    # img1, img2 = load_img(img1_file, img2_file)
    end_time1 = time.time()
    elapsed_time1 = end_time1 - start_time
    print(job_id, type, "part1 finished!", str(elapsed_time1))

    # 课题一， project1是课题一的主要修改部分！！！
    try:
        pic1, pic2 = project1_registration(job_id, type, img1, img2, 'image')
    except Exception as e:
        print(f"Error in project1_registration: {e}")
        pic1, pic2 = None, None
    end_time2 = time.time()
    elapsed_time2 = end_time2 - end_time1
    print(job_id, type, "part2 finished!", str(elapsed_time2))
    # 需要调试图像的时候可以用一下，线上要关掉
    # write_image(pic1, write_filepath)
    # 将图像数据转为 base64
    if img1.endswith('.tif' or 'tiff') and img2.endswith('.tif' or 'tiff'):
        # 读取图像
        img_proj1, img_geotrans1, image_1 = load_image(img1)
        img_proj2, img_geotrans2, image_2 = load_image(img2)
        if image_1.ndim == 3:
            image_1 = np.transpose(image_1, (1, 2, 0))
            image_1 = np.ascontiguousarray(image_1, dtype=np.float32)
        if image_2.ndim == 3:
            image_2 = np.transpose(image_2, (1, 2, 0))
            image_2 = np.ascontiguousarray(image_2, dtype=np.float32)
        imagere1 = image_to_base64(image_1)
        imagere2 = image_to_base64(image_2)
    else:
        imagere1 = image_to_base64(cv2.imread(img1))
        imagere2 = image_to_base64(cv2.imread(img2))
    pic1_base64 = image_to_base64(pic1)
    pic2_base64 = image_to_base64(pic2)
    # 课题二，project2是课题二的主要修改部分！！！
    polygons, accuracy = project2(job_id, pic1, pic2, GroundTruth2)
    end_time3 = time.time()
    elapsed_time3 = end_time3 - end_time2
    print(job_id, type, "part3 finished!", str(elapsed_time3))
    # 需要调试图像的时候可以用一下，线上要关掉
    # write_image(pic1,polygons, write_filepath)

    # 输出组装
    # 返回在java端实现roi区域图片的绘制与面积的计算。
    elapsed_time = {"read_time": elapsed_time1, "alg1": elapsed_time2, "alg2": elapsed_time3,
                    "all_time": elapsed_time1 + elapsed_time2 + elapsed_time3}
    annotations = [{
        # "resultimage1": pic1_base64,  # 将图像作为 base64 字符串返回
        "imagere1": imagere1,
        "imagere2": imagere2,
        "resultimage2": pic2_base64
    }]

    # print(job_id, "cal finished!", str(annotations))
    sync_finish(job_id, annotations)
    return annotations, 200
