import datetime
import os
import time

import cv2
import os

import imageio
import scipy.io
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt
import matlab.engine

from Show_Matches import Show_Matches
from Mosaic_Map import load_image, WriteTifImg


# 主函数
def MS_main(input_file1, input_file2, output_file='images'):
    # Parameters
    G_resize = matlab.double(2)  # Gaussian pyramid downsampling ratio, default: 2
    nOctaves1 = matlab.double(3)  # Gaussian pyramid octave number, default: 3
    nOctaves2 = matlab.double(3)
    G_sigma = matlab.double(1.6)  # Gaussian blurring standard deviation, default: 1.6
    nLayers = matlab.double(4)  # Gaussian pyramid layer number, default: 4
    radius = matlab.double(2)  # Local non-maximum suppression radius, default: 2
    N = matlab.double(2000)  # Keypoints number threhold
    patch_size = matlab.double(72)  # GGLOH patchsize, default: 72 or 96
    NBS = matlab.double(12)  # GGLOH localtion division, default: 12
    NBO = matlab.double(12)  # GGLOH orientation division, default: 12
    Error = matlab.double(5)  # Outlier removal pixel loss
    K = matlab.double(1)  # Rxperimental repetition times
    rotation_flag = matlab.double(1)  # 1:yes, 0:no
    trans_form = 'affine'  # similarity, affine, projective
    output_form = 'Union'  # Reference, Union, Inter

    eng = matlab.engine.start_matlab()
    eng.cd('./MS-HLMO_registration', nargout=0)

    # 读取图像
    # image_1 = cv2.imread('E:\\Image register\\Data\\pairs\\file1-3.tif')
    # image_2 = cv2.imread('E:\\Image register\\Data\\pairs\\file2-3.tif')
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
        image_2 = cv2.imread(input_file2)

    # image_1 = matlab.double(image_1.tolist())
    # image_2 = matlab.double(image_2.tolist())

    # image_1 = cv2.resize(image_1, (512, 512))
    # image_2 = cv2.resize(image_2, (512, 512))

    # # 图像预处理
    # radius = 2
    # I1_s, I1, r1, I2_s, I2, r2 = Preproscessing(image_1, image_2, radius)
    # I1_s = matlab.double(I1_s)
    # I1 = matlab.double(I1)
    # I2_s = matlab.double(I2_s)
    # I2 = matlab.double(I2)
    I1_s, I1, r1, I2_s, I2, r2 = eng.Preproscessing(image_1, image_2, radius, nargout=6)
    r1 = matlab.double(r1)
    r2 = matlab.double(r2)
    I1_s_change = np.array(I1_s)
    I2_s_change = np.array(I2_s)
    # plt.figure()
    # plt.title('Reference image')
    # plt.imshow(I1_s_change)
    # plt.pause(0.01)
    # plt.figure()
    # plt.title('Sensed Image')
    # plt.imshow(I2_s_change)
    # plt.pause(0.01)

    print("\n** Registration starts, have fun\n")

    # # Keypoints Detection
    start_Time = time.time()
    start_time = time.time()
    # keypoints_1 = Detect_Keypoint(I1, 6, r1, N, nOctaves1, G_resize)
    keypoints_1 = eng.Detect_Keypoint(I1, matlab.double(6), r1, N, nOctaves1, G_resize)
    end_time = time.time()
    str_ = f"Done: Keypoints detection of reference image, time cost: {end_time - start_time:.4f} s"
    print(str_)
    keypoints_1_change = np.array(keypoints_1)
    # plt.figure()
    # plt.imshow(I1_s_change)
    # plt.plot(keypoints_1_change[:, 0], keypoints_1_change[:, 1], 'r+')
    # plt.pause(0.01)

    start_time = time.time()
    # keypoints_2 = Detect_Keypoint(I2, 6, r2, N, nOctaves2, G_resize)
    keypoints_2 = eng.Detect_Keypoint(I2, matlab.double(6), r2, N, nOctaves2, G_resize)
    end_time = time.time()
    str_ = f"Done: Keypoints detection of sensed image, time cost: {end_time - start_time:.4f} s\n\n"
    print(str_)
    keypoints_2_change = np.array(keypoints_2)
    # plt.figure()
    # plt.imshow(I2_s_change)
    # plt.plot(keypoints_2_change[:, 0], keypoints_2_change[:, 1], 'r+')
    # plt.pause(0.01)

    cor1, cor2 = eng.Test_Registration(I1, I2, keypoints_1, keypoints_2, patch_size, NBS, NBO, nOctaves1, nOctaves2,
                                       nLayers, G_resize, G_sigma, rotation_flag, Error, K, nargout=2)

    cor1_change = np.array(cor1)
    cor2_change = np.array(cor2)

    if I1_s_change.ndim == 2:
        I1_s_change = np.expand_dims(I1_s_change, -1)
    if I2_s_change.ndim == 2:
        I2_s_change = np.expand_dims(I2_s_change, -1)

    # Show matches
    # matchment = Show_Matches(I1_s_change, I2_s_change, cor1_change, cor2_change, 1)

    # Image transformation
    start_time = time.time()

    if output_form == 'Reference':
        I2_r, I2_rs, I3, I4 = eng.Transform_ref(image_1, image_2, cor1, cor2, trans_form)
    elif output_form == 'Union':
        I1_r, I2_r, I1_rs, I2_rs, I3, I4, theta_x, theta_y, trans_points = eng.Transform_union(image_1, image_2, cor1,
                                                                                               cor2, trans_form,
                                                                                               nargout=9)
    elif output_form == 'Inter':
        I1_r, I2_r, I1_rs, I2_rs, I3, I4 = eng.Transform_inter(image_1, image_2, cor1, cor2, trans_form, nargout=6)

    end_time = time.time()
    str_ = f"Done: Image transformation, time cost: {end_time - start_time:.4f} s\n\n"
    print(str_)
    str_time = f"Done: Image Register time cost: {end_time - start_Time:.4f} s\n\n"
    print(str_time)
    # 关闭MATLAB引擎
    eng.quit()

    I1_r_change = np.array(I1_r)
    I2_r_change = np.array(I2_r)
    I1_rs_change = (np.array(I1_rs) * 255).astype('uint8')
    I2_rs_change = (np.array(I2_rs) * 255).astype('uint8')

    I3_change = (np.array(I3) * 255).astype('uint8')
    I4_change = (np.array(I4) * 255).astype('uint8')

    trans_points_c = np.array(trans_points)
    # print(trans_points_c[0][0])
    # print(trans_points_c[0][1])

    # # Display images
    # plt.figure()
    # plt.title('Fusion Form')
    # plt.imshow(I3_change, cmap='gray')
    # plt.pause(0.01)
    #
    # plt.figure()
    # plt.title('Mosaic Form')
    # plt.imshow(I4_change, cmap='gray')
    # plt.pause(0.01)

    # Save results
    # out_file = 'save_image'

    # if not os.path.exists(output_file):
    #     os.mkdir(output_file)

    date_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S__')
    correspond = [cor1_change, cor2_change]
    # correspond_path = os.path.join(output_file, f'{date_str}0_correspond.mat')
    # scipy.io.savemat(correspond_path, {'correspond': correspond})

    # if matchment:  # Assuming matchment is a valid matplotlib figure
    #     matchment_path = os.path.join(output_file, f'{date_str}0_Matching_Result.jpg')
    #     matchment.savefig(matchment_path)
    #
    # if output_form == 'Reference':
    #     ref_image_path = os.path.join(output_file, f'{date_str}1_Reference_Image.mat')
    #     save_image_path = os.path.join(output_file, f'{date_str}1_Reference_Image.jpg')
    #     scipy.io.savemat(ref_image_path, {'image_1': image_1})
    #     # cv2.imwrite(save_image_path, I1_s_change)
    #     imageio.imsave(save_image_path, I1_s_change)
    # else:
    #     reg_image_path = os.path.join(output_file, f'{date_str}1_Reference_Image.mat')
    #     save_image_path = os.path.join(output_file, f'{date_str}1_Reference_Image.jpg')
    #     scipy.io.savemat(reg_image_path, {'I1_r': I1_r_change})
    #     # cv2.imwrite(save_image_path, I1_rs_change)
    #     imageio.imsave(save_image_path, I1_r_change)
    #     if output_form == 'Union' and tif_flag:
    #         img_geotrans1_l = list(img_geotrans1)
    #         img_geotrans1_l[0] = img_geotrans1[0] - trans_points_c[0][0]  # x轴的位移
    #         img_geotrans1_l[3] = img_geotrans1[3] - trans_points_c[0][1]  # y轴的位移
    #         img_geotrans1_l[2] = img_geotrans1[2] + theta_x  # x轴的旋转
    #         img_geotrans1_l[4] = img_geotrans1[4] + theta_y  # y轴的旋转
    #         if I1_r_change.ndim == 3:
    #             I1_r_tif = np.transpose(I1_r_change, (2, 0, 1))
    #         else:
    #             I1_r_tif = I1_r_change
    #         save_tif_path = os.path.join(output_file, f'{date_str}1_Reference_Image.tif')
    #         WriteTifImg(save_tif_path, img_proj1, img_geotrans1_l, I1_r_tif)
    #
    # reg_image_path = os.path.join(output_file, f'{date_str}2_Registered_Image.mat')
    # save_image_path = os.path.join(output_file, f'{date_str}2_Registered_Image.jpg')
    # scipy.io.savemat(reg_image_path, {'I2_r': I2_r_change})
    # # cv2.imwrite(save_image_path, I2_rs_change)
    # imageio.imsave(save_image_path, I2_r_change)
    # if output_form == 'Union' and tif_flag:
    #     if I2_r_change.ndim == 3:
    #         I2_rs_tif= np.transpose(I2_r_change, (2, 0, 1))
    #     else:
    #         I2_r_tif = I2_r_change
    #     save_tif_path = os.path.join(output_file, f'{date_str}2_Registered_Image.tif')
    #     WriteTifImg(save_tif_path, img_proj2, img_geotrans1_l, I2_r_tif)

    # #save images
    # fusion_path = os.path.join(output_file, f'{date_str}3_Fusion_of_results.jpg')
    # mosaic_path = os.path.join(output_file, f'{date_str}4_Mosaic_of_results.jpg')
    # # cv2.imwrite(fusion_path, I3_change)
    # # cv2.imwrite(mosaic_path, I4_change)
    # imageio.imsave(fusion_path, I3_change)
    # imageio.imsave(mosaic_path, I4_change)

    # result_str = 'The results are saved in the save_image folder.\n\n'
    # print(result_str)

    return I1_r_change, I2_r_change  # 返回参考影像和配准后影像


if __name__ == '__main__':
    img1_file = r"C:\Users\Administrator\Pictures\DSCF5135.JPG"
    img2_file = r"C:\Users\Administrator\Pictures\DSCF5134.JPG"
    output_file = 'images'
    image1, image2 = MS_main(img1_file, img2_file, output_file)
    print(image1)
