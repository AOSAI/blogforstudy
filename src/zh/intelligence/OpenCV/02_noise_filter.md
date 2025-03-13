---
title: 噪声和滤波器
order: 2
author: AOSAI
date: 2024-12-15
category:
  - 图像处理入门
tag:
  - 噪声（Noise）
  - 过滤器（Filter）
  - 卷积
---

## 1. 噪声（Noise）

噪声在图像上通常表现为：引起较强视觉效果的孤立像素点/像素块。一般情况下，噪声信号与研究对象并不相关，它以无用的信息形式出现，扰乱图像的可观测信息。**通俗的说，噪声让图像变得不清楚了。**

### 1.1 噪声来源

**1. 图像获取过程中**

两种常用的图像传感器：CCD 和 CMOS，在采集图像过程中，由于受到传感器材料属性、工作环境、电子元器件和电路结构等影响，会引入各种噪声。

如，电阻引起的热噪声、场效应管的沟道热噪声、光子噪声、暗电流噪声、光响应非均匀噪声等。

**2. 图像信号传输过程中**

由于传输介质和记录设备等的不完善，数字图像在其传输记录过程中，往往会受到多种噪声的污染。另外，在图像处理的某些环节中，输入的对象并不如预想时，也会在结果图像中引入噪声。

### 1.2 椒盐噪声

椒盐噪声（Salt-and-Pepper Noise）是指随机在图像上出现黑色（椒）或白色（盐）的像素点，形成高对比的噪声。

一般是由于数据传输错误或图像采集中的传感器故障所造成的。表现为图像中的某些像素值突变为 0 或 255。使用 Python 和 OpenCV 进行噪声模拟：

![2.1 Salt-and-Pepper Noise](/img_process/02_noise&filter/01_noise1.png =560x)

```py
import cv2
import numpy as np

img = cv2.imread("img_process_1/picture.jpg");
new_size = (int(1000), int(1000 / img.shape[1] * img.shape[0]))
img1 = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)
cv2.imwrite("img_process_1/img1_1000px.png", img1)

noise1_img = np.copy(img1)
sp_rate = 0.5    # 设置椒盐噪声的黑白比例
sp_amount = 0.03    # 设置噪声图像像素的数目

# 添加salt噪声
num_salt = np.ceil(sp_amount * noise1_img.size * sp_rate)
# 设置添加噪声的坐标位置
position_salt = [np.random.randint(0, i-1, int(num_salt)) for i in noise1_img.shape]
noise1_img[position_salt[0], position_salt[1], :] = [255, 255, 255]

# 添加pepper噪声
num_pepper = np.ceil(sp_amount * noise1_img.size * (1- sp_rate))
# 设置添加噪声的坐标位置
position_pepper = [np.random.randint(0, i-1, int(num_pepper)) for i in noise1_img.shape]
noise1_img[position_pepper[0], position_pepper[1], :] = [0, 0, 0]

cv2.imwrite("img_process_1/noise1_img.png", noise1_img)
```

因为图像原本的大小很大，如果不缩小，我 2k 屏幕都放不下，所以我先 resize 了一下。

### 1.3 高斯噪声

高斯噪声（Gaussian Noise）常见于自然环境中的噪声，符合高斯分布，即噪声值服从均值为零、方差为某值的正态分布。一般是由于传感器热噪声、电子电路干扰等产生的。表现为：图像上呈现为随机分布的细小亮点或暗点。

![2.2 Gaussian Noise](/img_process/02_noise&filter/02_noise2.png =560x)

```py
import cv2
import numpy as np

noise2_img = cv2.imread("img_process_1/img1_1000px.png")
mean = 0    # 均值
sigma = 50  # 标准差（方差）

# 生成符合正态（normal）分布的随机数噪声
gauss = np.random.normal(mean, sigma, noise2_img.shape)
# 将噪声添加到 noise2_img 图像上
noise2_img = noise2_img + gauss
# 限制像素范围 [0, 255]，小于0的值会被设置为0，大于255的同理
noise2_img = np.clip(noise2_img, a_min=0, a_max=255)
cv2.imwrite("img_process_1/noise2_img.png", noise2_img)
```

### 1.4 泊松噪声

泊松噪声（Poisson Noise）与图像的信号强度相关，噪声值服从泊松分布。通常由光子的统计特性引起，如在低光条件下拍摄的照片，亮度较高的区域噪声更明显。

![2.3 Poisson Noise](/img_process/02_noise&filter/03_noise3.png =560x)

```py
import cv2
import numpy as np

noise3_img = cv2.imread("img_process_1/img1_1000px.png")

# lam 为 lambda 的缩写，在泊松分布中表示均值
noise = np.random.poisson(lam=12,size=noise3_img.shape).astype('uint8')
noise3_img = noise3_img + noise
noise3_img = np.clip(noise3_img, 0, 255)
cv2.imwrite("img_process_1/noise3_img.png", noise3_img.astype('uint8'))

# 归一化像素值，用于显示
noise3_img = noise3_img / 255
cv2.imshow("Poisson Noise Iamge", noise3_img)
cv2.waitKey()
```

### 1.5 常见噪声和处理方式

**1. 高斯噪声（Gaussian Noise）**：==高斯滤波，中值滤波，双边滤波==

**2. 椒盐噪声（Salt-and-Pepper Noise）**：==中值滤波，自适应滤波==

**3. 泊松噪声（Poisson Noise）**：==方差稳定化（如 Anscombe 变换），小波去噪==

**4. 均值噪声（Uniform Noise）**：==均值滤波，降噪算法（如小波变换）==

是指在一个固定范围内随机均匀分布的噪声。来源于量化误差或简单的随机干扰。

**5. 斑点噪声（Speckle Noise）**：==自适应滤波（如 Lee 滤波），双边滤波，尺度空间降噪==

通常出现在合成孔径雷达（SAR）图像和医学超声图像中。来源于相干信号叠加的干涉效应。表现为形成类似颗粒状的图像退化。

**6. 周期性噪声（Periodic Noise）**：==傅里叶变换滤波（频域分析），带阻滤波==

以周期性模式出现的噪声。来源于机械震动或电磁干扰。表现为图像中出现周期性的条纹或波纹。

**7. 量化噪声（Quantization Noise）**：==提高量化位深，后处理滤波==

是指图像数据在数字化过程中，由于位数限制导致的量化误差。主要因为数字图像采样或压缩时的位深度不足。表现为图像细节丢失或分辨率降低。

## 2. 滤波器（Filter）

### 2.1 平滑化与卷积

平滑化是减少图像中的噪声或细节，使其更加平滑的过程，常用于预处理阶段。主要用途有：去除噪声
、模糊化图像、提高边缘检测效果、图像重采样前的抗混叠处理。

可能大多数人都听过 CNN（convolutional neural network），它叫做卷积神经网络。卷积，我的理解是，一个灵活的工具，一种处理图像的方式，可以用来执行平滑、锐化和边缘检测等操作。

它的核心是将滤波器（卷积核）应用于图像的一种数学运算，计算图像像素与核函数的加权和。

### 2.2 均值滤波器

### 2.3 中值滤波器

### 2.4 高斯滤波器
