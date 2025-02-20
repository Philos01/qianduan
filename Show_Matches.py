import math

import numpy as np
import matplotlib.pyplot as plt

def Show_Matches(I1, I2, cor1, cor2, option):
    I3, cor1, cor2 = Append_Images(I1, I2, cor1, cor2, option, 'middle')
    matchment = plt.figure()
    plt.imshow(I3)
    # 设置线的粗细
    line_width = 0.5
    # 设置点的大小，例如将点的大小设置为10
    marker_size = 2

    if option == 1:
        plt.title(f'Left is reference image --- {len(cor1)} matching pairs --- Right is sensed image')
        cols = I1.shape[1]
        for i in range(len(cor1)):
            plt.plot([cor1[i, 0], cor2[i, 0] + cols], [cor1[i, 1], cor2[i, 1]], color='y', linewidth=line_width)
            plt.plot(cor1[i, 0], cor1[i, 1], 'go', color='r',markersize=marker_size)
            plt.plot(cor2[i, 0] + cols, cor2[i, 1], 'g+', color='g',markersize=marker_size)
    elif option == 2:
        plt.title(f'Top is reference image --- {len(cor1)} matching pairs --- Bottom is sensed image')
        rows = I1.shape[0]
        for i in range(len(cor1)):
            plt.plot([cor1[i, 0], cor2[i, 0]], [cor1[i, 1], cor2[i, 1] + rows], color='y',linewidth=line_width)
            plt.plot(cor1[i, 0], cor1[i, 1], 'go', color='r',markersize=marker_size)
            plt.plot(cor2[i, 0], cor2[i, 1] + rows, 'g+', color='g',markersize=marker_size)

    plt.pause(0.01)
    return matchment

def Append_Images(I1, I2, cor1, cor2, option, pos):
    _, _, B1 = I1.shape
    _, _, B2 = I2.shape

    if B1 != 1 and B1 != 3:
        I1 = np.sum(I1, axis=2)
    I1 = Visual(I1)

    if B2 != 1 and B2 != 3:
        I2 = np.sum(I2, axis=2)
    I2 = Visual(I2)

    M1, N1, B1 = I1.shape
    M2, N2, B2 = I2.shape

    if B1 == 1 and B2 == 3:
        temp = np.copy(I1)
        I1[:, :, 0] = temp
        I1[:, :, 1] = temp
        I1[:, :, 2] = temp
    elif B1 == 3 and B2 == 1:
        temp = np.copy(I2)
        I2[:, :, 0] = temp
        I2[:, :, 1] = temp
        I2[:, :, 2] = temp

    if option == 1:
        if pos == 'top':
            if M1 < M2:
                I1[M2, 0] = 0
            else:
                I2[M1, 0] = 0
        elif pos == 'middle':
            if M1 < M2:
                dM = math.floor(abs(M1 - M2) / 2)
                temp = I1.copy()
                I1 = np.zeros((M2, N1, I1.shape[2]))
                I1[dM:dM + M1, :, :] = temp
                cor1[:, 1] += dM
            else:
                dM = math.floor(abs(M1 - M2) / 2)
                temp = I2.copy()
                I2 = np.zeros((M1, N2, I2.shape[2]))
                I2[dM:dM + M2, :, :] = temp
                cor2[:, 1] += dM
        img = np.concatenate((I1, I2), axis=1)
    elif option == 2:
        if N1 < M2:
            I1[0, N2] = 0
        else:
            I2[0, N1] = 0
        img = np.concatenate((I1, I2), axis=0)

    return img, cor1, cor2

def Visual(I_o):
    aaa = I_o[I_o > 0]
    I = I_o / np.mean(aaa) / 2.5
    return I

# 示例用法
# Show_Matches(I1, I2, cor1, cor2, option)
