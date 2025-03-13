---
title: 图像基础操作
order: 1
author: AOSAI
date: 2024-12-24
category:
  - 图像处理入门
tag:
  - 图像基础
---

<style>
  @media (orientation:landscape){
    .layout{
      display:flex;
    }
  }
  @media (orientation:portrait){
    .layout{}
  }
</style>

## 1. 图像基础操作

### 1.1 读取，显示，写入

```py
# opencv 读取图像
img1 = cv2.imread(filename, flags)

# 彩色BGR转灰度
img2 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

# 灰度转彩色BGR
img3 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
```

filename 指的是文件的路径，相对路径默认为工作环境（代码项目）的根目录。flags 指的是图像的读取方式：

默认为 1（cv2.IMREAD_COLOR）, 即 BGR（blue, green, red）彩色图像格式；0（cv2.IMREAD_GRAYSCALE）的话，为单通道的灰度值图像。

除了读取时使用 flags 来决定 灰度/彩色 之外，我们还可以用 cvtColor 函数进行两者之间的转化。它这个命名还是很有意思的，灰色（gray），彩色（BGR），2 就是 to 的谐音。

但是要注意的是，灰度转彩色并不会恢复原始色彩，仅生成具有相同灰度值的三通道图像。opencv 中色彩转换的原理是，通过某种加权方法，将彩色的三通道合并为单一的通道，例如：

$$Gray=0.114×B+0.587×G+0.299×R$$

在这个过程中，大量的颜色信息（特别是色调和饱和度）丢失了，只保留了亮度信息。这是一个不可逆的过程，所以从灰度转换为彩色三通道图像时，只是将灰度值复制到 R、G、B 三个通道上，形成一个“伪彩色图像”。

```py
# opencv 显示图像
cv2.imshow(showbox_name, img_object)
# 关闭前的等待时间，0或默认，为手动关闭；
cv2.waitkey()

# matplotlib 显示图像
import matplotlib.pyplot as plt
# opencv 的图像格式是 BGR，需要转换成 RGB供 plt显示
plt_img = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
plt.imshow(plt_img)
plt.title("color_img")
plt.axis("off") # 关闭坐标轴
plt.show()

# 指定 cmap="gray" 以灰度模式显示
# plt.imshow(gray_image, cmap="gray")
```

opencv 中显示图像需要跟一个 waitkey 函数，不加的话，会一闪而过，什么都看不到。showbox_name 是指显示框的名称，img_object 就是字面意思，需要被显示的图像对象。

一个小 tag：如果需要循环输出多张图像，在 waitkey ==0 的情况下，显示框名称可以一样，因为你必须关掉上一个显示框，第二个显示框才会出现。但是如果你想连续显示很多图像，就需要让每一个的显示框名称不一样。当然，类似摄像头捕获视频流这种的，不属于我说的情况。

另外，有些人可能习惯使用 plt 显示图像，需要注意的是 plt 中的彩色图像是 RGB 格式，而 opencv 中的图像是 BGR 格式，所以显示之前，需要先进行一个转换。

```py
# opencv 存储/写入图像
cv2.imwrite(filename, img_object)
```

filename 为图像保存的路径，注意图像的后缀一定要加（.png .jpg 之类的），img_obj 还是一样，表示要储存的图像对象。

### 1.2 图像的数据类型

图像在读取过程中，可以大致分为**整型**和**浮点型**两种。在不同的编程语言下，写法大相径庭。

|     C++类型     | Python 类型（Numpy） |                 说明                  |
| :-------------: | :------------------: | :-----------------------------------: |
| CV_8U / CV_8UC1 |       np.uint8       | 单通道 8 位无符号整数图像（灰度图像） |
|     CV_8UC3     |       np.uint8       | 三通道 8 位无符号整数图像（彩色图像） |
|     CV_32F      |      np.float32      |         单通道 32 位浮点图像          |
|    CV_32FC3     |      np.float32      |         三通道 32 位浮点图像          |
|    CV_32FC2     |     np.complex64     |          双通道复数浮点图像           |

cv2.imshow()的过程中，整型和浮点类型都可以显示，但是需要注意像素值的区间，整型为 [0, 255]，浮点型为 [0, 1]。如果超出范围，是不能被正常显示的。

cv2.imwrite()的过程中，浮点类型会强制要求转换为 uint8 格式，否则保存的图片可能不能正常显示。

```py
# 限制浮点类型的范围于 0~1 之间
float_img = np.clip(float_img, 0, 1)

# 给每个像素点乘以 255，然后转换为整型
uint8_img = (float_img * 255).astype(np.uint8)

cv2.imwrite('output.jpg', uint8_img)
```

### 1.3 旋转

（1）如果是对图像进行固定角度的旋转（90 度的倍数），可以使用 ==cv2.rotate== 函数。

<div class="layout">

![1.1 旋转90度](/img_process/01_basic/02_lena_ro90.png =240x)

![1.2 旋转180度](/img_process/01_basic/03_lena_ro180.png =240x)

![1.3 旋转270度](/img_process/01_basic/04_lena_ro270.png =240x)

</div>

```py
import cv2

img = cv2.imread("img_process_0/lena.png");
rotated_90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
rotated_180 = cv2.rotate(img, cv2.ROTATE_180)
rotated_270 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

cv2.imshow('Original', img)
cv2.imshow('Rotated 90', rotated_90)
cv2.imshow('Rotated 180', rotated_180)
cv2.imshow('Rotated 270', rotated_270)
cv2.waitKey(0)
```

在编程中，大多数参数的英文命名都是直译的，比如 CLOCKWISE 就代表顺时针，加上 ROTATE_90\_，就代表顺时针方向旋转 90 度；而 COUNTERCLOCKWISE 表示逆时针，顺时针旋转 270 度，就等于逆时针旋转 90 度，一个意思。

（2）对于任意角度的旋转（比如 30 度、45 度等），需要使用 OpenCV 的 ==cv2.getRotationMatrix2D== 和 ==cv2.warpAffine== 函数。

<div class="layout">

![1.4 旋转45度，无缩放](/img_process/01_basic/05_lena_45.png =240x)

![1.5 旋转200度，缩放0.7](/img_process/01_basic/06_lena_200.png =240x)

</div>

```py
import cv2

img = cv2.imread("img_process_0/lena.png");

# 获取图像中心坐标
(h, w) = img.shape[:2]
center = (w // 2, h // 2)

# 生成旋转矩阵：旋转45度，缩放比例为1.0
angle = 45
scale = 1.0
M = cv2.getRotationMatrix2D(center, angle, scale)
M1 = cv2.getRotationMatrix2D(center, 200, 0.7)

# 进行仿射变换，输出旋转后的图像
rotated = cv2.warpAffine(img, M, (w, h))
rotated1 = cv2.warpAffine(img, M1, (w, h))

cv2.imshow('Original', img)
cv2.imshow('Rotated', rotated)
cv2.imshow('Rotated1', rotated1)
cv2.waitKey(0)
```

在 OpenCV 中，==img.shape== 函数返回一个三元组：（高度，宽度，通道数）。因此，可以使用切片 [:2] 的方法，直接对高度、宽度进行赋值。

- shape[0]：表示图像的高度（rows，即行数）。
- shape[1]：表示图像的宽度（columns，即列数）。
- shape[2]：表示图像的通道数（例如 RGB 图像有 3 个通道）。

在 Python 中，==//== 是 整数除法运算符，表示进行除法运算后，向下取整（只取结果的整数部分，不保留小数）。

（3）在旋转图像时，如果不希望缩放图像，保持原有比例 1.0，也不希望图像内容被裁剪，可以对输出图像的边界大小进行调整。

```py
M = cv2.getRotationMatrix2D(center, angle, scale)

# 调整旋转后的图像边界大小
# M[0, 0] 和 M[0, 1] 分别对应旋转矩阵中的 cos(θ) 和 sin(θ)。
# np.abs()：取绝对值，确保结果为正数。
cos = np.abs(M[0, 0])
sin = np.abs(M[0, 1])
new_w = int((h * sin) + (w * cos))
new_h = int((h * cos) + (w * sin))

# 调整旋转矩阵，考虑平移
# M[0, 2] 和 M[1, 2] 是仿射变换矩阵中的 平移项。
# 由于旋转后图像的尺寸变大，原来的中心点需要平移到新的图像中心。
M[0, 2] += (new_w / 2) - center[0]
M[1, 2] += (new_h / 2) - center[1]

# 进行仿射变换
rotated = cv2.warpAffine(img, M, (new_w, new_h))
```

### 1.4 翻转(镜像)

在 OpenCV 中，可以使用 ==cv2.flip== 函数对图像进行 **翻转（镜像）** 操作。cv2.flip 支持以下三种翻转方式：

- 水平翻转（左右镜像）：flipCode = 1
- 垂直翻转（上下镜像）：flipCode = 0
- 水平 + 垂直翻转（180° 翻转）：flipCode = -1

<div class="layout">

![1.6 水平翻转](/img_process/01_basic/07_lena_flip_h.png =240x)

![1.7 垂直翻转](/img_process/01_basic/08_lena_flip_v.png =240x)

![1.8 水平+垂直翻转](/img_process/01_basic/09_lena_flip_hv.png =240x)

</div>

```py
import cv2

img = cv2.imread("img_process_0/lena.png")

flipped_h = cv2.flip(img, 1)
flipped_v = cv2.flip(img, 0)
flipped_hv = cv2.flip(img, -1)

cv2.imshow('Original', img)
cv2.imshow('Horizontal Flip', flipped_h)
cv2.imshow('Vertical Flip', flipped_v)
cv2.imshow('Both Flip', flipped_hv)
cv2.waitKey(0)
```

### 1.5 裁剪

（1）图像裁剪的核心是==对图像的像素数组（numpy 数组）进行切片==操作。

**首先，我们要知道，图像是由一个一个的像素点组成的，而 OpenCV 读取的图像是一个 NumPy 二维数组（RGB 三通道拆分开，就是三个这样的二维数组），数组中的值就是由图像中的像素点的值构成。**

**前面使用 OpenCV 的函数，对图像进行旋转、翻转，本质上都是对像素点进行操作。P.S. 小实验：可以自己尝试用 Python 和 Numpy，手写对图像进行水平翻转的函数。**

我使用的经典 Lena 图像，图像大小是 512\*512 px，假如我想要裁剪她的眼睛部位，并且我也知道裁剪区域的坐标轴，我可以这么写：

```py
import cv2

img = cv2.imread('img_process_0/lena.png')

# 定义裁剪区域 [y1:y2, x1:x2]
x1, y1 = 200, 220   # 左上角坐标 (x1, y1)
x2, y2 = 400, 300  # 右下角坐标 (x2, y2)

# 裁剪图像
cropped_img = img[y1:y2, x1:x2]

cv2.imwrite("img_process_0/lena_crop1.png", cropped_img)
```

![1.9 像素数组切片裁剪](/img_process/01_basic/10_lena_crop1.png =240x)

但是这种方式明显不适合普通人使用啊对吧，普通人哪里知道需要裁剪的坐标轴数据是多少，那么有没有类似手机电脑端，通过框选区域进行裁剪的方法呢？OpenCV 还真有。

（2）OpenCV 支持鼠标交互，通过==监听鼠标事件==，可以选择图像中的任意区域进行裁剪。

```py
import cv2

# 初始化参数
drawing = False
ix, iy = -1, -1  # 起点坐标
rect = (0, 0, 0, 0)  # 选框坐标
cropped = None  # 保存裁剪结果

def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, rect, cropped

    if event == cv2.EVENT_LBUTTONDOWN:  # 鼠标左键按下
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:  # 鼠标拖动
        if drawing:
            img2 = img.copy()
            cv2.rectangle(img2, (ix, iy), (x, y), (0, 255, 0), 1)
            cv2.imshow('Image', img2)

    elif event == cv2.EVENT_LBUTTONUP:  # 鼠标左键释放
        drawing = False
        rect = (ix, iy, x, y)

        # 防止坐标越界或负数情况
        x1, y1 = min(ix, x), min(iy, y)
        x2, y2 = max(ix, x), max(iy, y)

        # 确保坐标合法
        if 0 <= x1 < x2 <= img.shape[1] and 0 <= y1 < y2 <= img.shape[0]:
            # 裁剪图像
            cropped = img[y1:y2, x1:x2]
            cv2.imshow('Cropped Image', cropped)
        else:
            print("选择的区域无效！")

# 加载图像
img = cv2.imread('img_process_0/lena.png')
if img is None:
    print("图像加载失败，请检查路径！")
    exit()

cv2.imshow('Image', img)
cv2.setMouseCallback('Image', draw_rectangle)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

这个简单的代码示例，可以让我们读取原始图像，通过鼠标选取想要裁剪的区域。如果搭配使用 PyQt 或 Tkinter 创建带有按钮、选择框的应用程序，可以更友好地选择裁剪区域；甚至可以做成一个软件，供自己或别人使用。

这时候可能有同学要问了：如果有很多张图像，我就想要提取一些相同的特征点进行裁剪，一个一个用鼠标交互太费劲了呀，有没有更方便的方法？

（3）==自动检测特定部位并裁剪==。OpenCV 提供内置的人脸检测器（Haar 级联分类器），可以自动识别人脸位置并进行裁剪。

```py
import cv2

# 加载 Haar 人脸检测器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 加载图像
img = cv2.imread('img_process_0/lena.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转为灰度图提高检测精度

# 检测人脸
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# 遍历检测到的人脸
for (x, y, w, h) in faces:
    # 绘制人脸矩形框
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 裁剪人脸
    face_crop = img[y:y+h, x:x+w]
    cv2.imshow('Cropped Face', face_crop)

# 显示原图
cv2.imshow('Detected Faces', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

这里面有个很有意思的事情，在加载 Haar 人脸检测器的函数中，有一个文件叫做：haarcascade_frontalface_default.xml。

在 Python 中，OpenCV 所对应的常用模型文件，是直接打包在库的安装路径中的，可以通过 cv2.data.haarcascades 动态获取这些文件所在的目录，无需手动下载或管理文件路径。

但是在 C++里，必须手动下载 Haar 特征分类器模型，将其存放在项目目录或指定路径下，然后手动提供该路径。

**detectMultiScale 方法是 OpenCV Haar 特征分类器的核心函数，用于检测人脸或其他目标**，具体的参数这里就不细讲了，关于特征检测，特征提取这方面的内容，会在之后的章节讲到。

### 1.6 调整图像大小

（1）按比例缩放 (Scale Factor)

直接通过比例因子（坐标轴百分比）调整图像大小，不关心具体像素大小。

```py
import cv2

image = cv2.imread('img_process_0/lena.png')

# 按比例缩放
scale_x = 0.5  # 宽度缩小为 50%
scale_y = 0.5  # 高度缩小为 50%
resized = cv2.resize(image, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)

original_height, original_width = image.shape[:2]
print(f"Original Image Size: {original_width}x{original_height}")

resized_height, resized_width = resized.shape[:2]
print(f"Resized Image Size: {resized_width}x{resized_height}")

# 输出结果：
# Original Image Size: 512x512
# Resized Image Size: 256x256
```

需要注意的是，resize 函数中的 interpolation 参数表示，==图像缩放时使用的插值算法==，用于计算新图像中的像素值。常见的插值方法有：

- cv2.INTER_NEAREST：最近邻插值，速度快，质量低，适合离散图像。
- cv2.INTER_LINEAR（默认）：双线性插值，适合缩小图像。
- cv2.INTER_CUBIC：双三次插值，质量较高但速度较慢。
- cv2.INTER_AREA：适合缩小图像，效果优于线性插值。
- cv2.INTER_LANCZOS4：Lanczos 插值，适合放大图像，质量高但速度慢。

（2）固定尺寸调整 (Fixed Size)

直接设置目标尺寸，而不考虑长宽比。可能会导致图像变形，因为长宽比没有保持一致。就比如 lena，从原本的 512\*512px，变成了 300\*200px。

```py
import cv2

image = cv2.imread('img_process_0/lena.png')
resized = cv2.resize(image, (300, 200), interpolation=cv2.INTER_LINEAR)

cv2.imshow('Scaled Image', resized)
cv2.waitKey(0)
```

![1.10 固定尺寸缩放可能会导致图像变形](/img_process/01_basic/11_lena_scale.png =240x)

虽然说，一些算法对图像的大小尺寸有明确的需求，例如一些神经网络算法。我们可以使用这种固定尺寸的办法进行缩放，但是更推荐边缘填充这种方式。

（3）保持比例，但宽/高有一方是固定像素大小

比如宽度（width）要缩放到 1000px（new_width），那么 new_width / width = new_height / height，所以 new_height = new_width \* height / width。

```py
image = cv2.imread('img_process_0/lena.png')
original_height, original_width = image.shape[:2]

width = 1000
height = 1000 * original_height / original_width
resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
```

（4）裁剪与调整大小 (Crop and Resize)

裁剪图像的中心区域或特定区域，然后调整为指定大小。一般用于图像识别任务或面部识别任务，避免图像变形。

```py
# 裁剪中间区域
crop = image[50:250, 50:250]  # 取中心区域
resized = cv2.resize(crop, (300, 300), interpolation=cv2.INTER_LINEAR)
```

（5）边缘填充 (Padding)

将不同尺寸的图像调整到相同大小，同时保持内容比例，适合神经网络等需要固定尺寸输入的场合。

众所周知，我使用的经典 lena 图像是 512\*512 的，目标大小和（2 Fixed Size）一样，还是（300，200），怎么缩放，怎么填充？

```py
import cv2

image = cv2.imread('img_process_0/lena.png')
h, w = image.shape[:2]

# 目标尺寸
target_w = 300
target_h = 200

# 缩放图像
scale = 200 / max(h, w)
resized = cv2.resize(image, (int(w * scale), int(h * scale)))

# 计算边距
delta_w = target_w - resized.shape[1]
delta_h = target_h - resized.shape[0]
top, bottom = delta_h // 2, delta_h - delta_h // 2
left, right = delta_w // 2, delta_w - delta_w // 2

# 填充
padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

cv2.imshow('Padded Image', padded)
cv2.waitKey(0)
```

![1.11 边缘填充](/img_process/01_basic/12_lena_padding.png =240x)

函数 cv2.copyMakeBorder 中 borderType, value=None 的注解：

**borderType**：指定边框的类型。

- cv2.BORDER_CONSTANT: 添加常数值的边框，value 参数指定边框颜色或填充值。
- cv2.BORDER_REPLICATE: 复制图像的边缘像素作为边框。
- cv2.BORDER_REFLECT: 反射图像的边缘，边缘像素镜像反射。
- cv2.BORDER_DEFAULT（cv2.BORDER_REFLECT_101）: 与 BORDER_REFLECT 相似，但边缘的反射不包括边缘像素。
- cv2.BORDER_WRAP: 环绕图像边缘像素。

**value**：指定边框颜色或填充值。

仅当 borderType 为 cv2.BORDER_CONSTANT 时才有效。

## 2. 图像压缩

仍旧记得，在我刚开始写博客的时候，因为有很多图像需要插入网页，我就想做一个编写的批量压缩图像的小软件。但是那个时候，我 OpenCV 用不明白，只能用 Pillow 进行压缩操作。所以，我来弥补遗憾了！

当然了，并不是说 OpenCV 做压缩就一定更好，它只支持 JPG、PNG、BMP、WebP 等主流格式，而 Pillow 支持更多格式，包括 TIFF、GIF、ICO 等。对比 OpenCV 和 Pillow：

- 如果需要更高性能和复杂图像处理任务，比如物体检测或图像分割，选择 OpenCV。
- 如果主要是格式转换、批量处理或 Web 图片压缩，选择 Pillow 更轻便。

如果是高性能需求，可以在 OpenCV 中处理图像，再使用 Pillow 保存成其他格式或更进一步优化。

### 2.1 imwrite 中的压缩参数

```py
import cv2

image = cv2.imread('img_process_0/lena.png')

cv2.imwrite('img_process_0/compressed1.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 50])
cv2.imwrite('img_process_0/compressed2.png', image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
cv2.imwrite('img_process_0/compressed3.webp', image, [cv2.IMWRITE_WEBP_QUALITY, 50])
```

- **IMWRITE_JPEG_QUALITY**：设置 JPEG 的压缩质量，范围为 0 - 100（默认值 95）。值越大，图像质量越高，压缩率越低，文件越大。值越小，图像质量下降，压缩率越高，文件越小。

- **IMWRITE_PNG_COMPRESSION**：设置 PNG 图像的压缩等级，0 - 9（默认值为 3）。值越大，压缩率越高，文件越小，但压缩速度越慢。值为 0 时不压缩，为 9 时压缩率最高但速度最慢。

- **IMWRITE_WEBP_QUALITY**：设置 WebP 图像的压缩质量，0 - 100（默认值为 75）。支持有损和无损压缩。值越大，质量越高，文件越大；值越小，质量越低，文件越小。

OpenCV 中，压缩相关的较全的图像格式，以及压缩参数，见下表：

| 格式 | 参数名称                               | 压缩类型  | 参数范围   | 默认值 |
| ---- | -------------------------------------- | --------- | ---------- | ------ |
| JPEG | cv2.IMWRITE_JPEG_QUALITY               | 有损压缩  | 0 - 100    | 95     |
| PNG  | cv2.IMWRITE_PNG_COMPRESSION            | 无损压缩  | 0 - 9      | 3      |
| WebP | cv2.IMWRITE_WEBP_QUALITY               | 有损/无损 | 0 - 100    | 75     |
| TIFF | cv2.IMWRITE_TIFF_COMPRESSION           | 无损/有损 | 1, 2, 3, 5 | 无     |
| PXM  | cv2.IMWRITE_PXM_BINARY                 | 无损      | 0 或 1     | 1      |
| EXR  | cv2.IMWRITE_EXR_TYPE                   | 无损压缩  | 0 - 5      | 无     |
| JP2  | cv2.IMWRITE_JPEG2000_COMPRESSION_X1000 | 有损压缩  | 0 - 1000   | 无     |

### 2.2 使用对象保存压缩数据

使用对象保存压缩数据，储存到内存缓冲区，可以用 imshow 读取，也可以进行网络传输。

```py
import cv2
import numpy as np

# 根据实际大小自动选择合适的单位
def format_size(size_bytes):
    if size_bytes < 1024:  # 小于 1KB
        return f"{size_bytes} B"
    elif size_bytes < 1024**2:  # 小于 1MB
        return f"{size_bytes / 1024:.2f} KB"
    else:  # 大于 1MB
        return f"{size_bytes / (1024**2):.4f} MB"

# 读取原始图像
image = cv2.imread('img_process_0/lena.png')

# 将图像压缩为 JPEG 格式并保存到内存缓冲区
encode_param = [cv2.IMWRITE_JPEG_QUALITY, 50]  # 设置压缩质量为50%
result, buffer = cv2.imencode('.jpg', image, encode_param)

# 确保编码成功
if result:
    byte_size = len(buffer)  # 获取字节大小
    print(f"Compressed size: {format_size(len(buffer))}")
else:
    print("Compression failed")

# 可以将 buffer 转换为字节流
# 用于 网络传输、数据库存储、实时处理（比如imdecode）
compressed_bytes = buffer.tobytes()

# 从压缩的字节数据中重新加载图像
compressed_image = cv2.imdecode(np.frombuffer(compressed_bytes, np.uint8), cv2.IMREAD_COLOR)

# 显示解码后的图像
cv2.imshow('Compressed Image', compressed_image)
cv2.waitKey(0)
```

**（1）result, buffer = cv2.imencode(ext, img, params=None)**

1. 返回值 result：布尔（bool）类型，表示编码是否成功。如果成功，返回 True，否则返回 False。
2. 返回值 buffer：numpy.ndarray 类型，表示编码后的二进制数据，存储在一个字节数组中，可以直接用于传输或写入文件。

- 参数 ext：文件扩展名（例如：.jpg, .png, .webp），确定使用哪种编码格式。
- 参数 img：输入的图像数据（通常为 numpy 数组类型）。
- 参数 params：编码参数列表，用于控制压缩质量或算法选项。

**（2）retval = cv2.imdecode(buf, flags)**

1. 返回值 retval：解码后的图像，类型为 numpy.ndarray。如果解码失败，返回 None。

- 参数 buf：numpy.ndarray 或 bytes 格式数据。
- 参数 flags：控制解码后的图像格式和颜色空间，常用值有：

| Flag                               | 用途                                                    | 使用频率 |
| ---------------------------------- | ------------------------------------------------------- | -------- |
| cv2.IMREAD_COLOR                   | 彩色图像解码（默认方式，无透明度）                      | 高       |
| cv2.IMREAD_GRAYSCALE               | 灰度图像解码                                            | 高       |
| cv2.IMREAD_UNCHANGED               | 保持原始图像格式，包括透明度                            | 较高     |
| cv2.IMREAD_ANYDEPTH                | 处理高精度位深度的图像，如 16 位或 32 位                | 中等     |
| cv2.IMREAD_REDUCED_COLOR_2/4/8     | 彩色图像按比例缩小，适合大图像加载优化（1/2，1/4，1/8） | 中等     |
| cv2.IMREAD_REDUCED_GRAYSCALE_2/4/8 | 彩色图像按比例缩小，适合大图像加载优化（1/2，1/4，1/8） | 中等     |

### 2.3 压缩算法简介

OpenCV 中 ==JPEG== 压缩算法的核心技术是：==DCT (离散余弦变换)==：

1. 将图像从空间域转换到频率域，通过抛弃高频信息（细节）来减少数据量。
2. 量化步骤会进一步降低精度并减少存储需求。
3. 使用熵编码（Huffman 编码）最终压缩数据。

OpenCV 中 ==PNG== 压缩算法的核心技术是：==DEFLATE 算法==：

1. 使用无损压缩，包括 LZ77 和哈夫曼编码的结合。
2. 数据不会丢失，但文件大小会更大。

OpenCV 中 ==WebP== 压缩算法的核心技术是：==预测编码 + 熵编码==：

1. 使用预测编码来减少冗余数据，然后使用 VP8 编码进一步压缩。
2. 支持有损和无损模式，适合现代网页和移动端应用。

## 3. 色彩空间解构方式

上一小节我们已经知道了基于 RGB 三原色通道的色彩空间，以及 Grayscale 灰度值下的单通道色彩空间，除此之外还有一些其他的色彩空间构成，比如 RGBA，引入了透明度信息；HSV，用色相、饱和度、明度来定义图像；……

| 色彩空间    | 特点                 | 典型应用               |
| ----------- | -------------------- | ---------------------- |
| RGB         | 最常用，直观但不独立 | 图像显示、简单处理     |
| Grayscale   | 简化为单通道         | 特征检测、边缘检测     |
| HSV / HLS   | 色调和饱和度独立     | 目标检测、颜色过滤     |
| YUV / YCbCr | 分离亮度和色度       | 视频编码、人脸检测     |
| Lab         | 与人眼感知接近       | 图像增强、颜色分割     |
| CMYK        | 针对打印优化         | 图像输出到打印设备     |
| XYZ / LUV   | 高精度颜色管理和分析 | 专业颜色匹配、颜色测量 |

### 3.1 RGB 色彩空间

（1） 使用 OpenCV 分离 RGB 通道

```py
import cv2

img = cv2.imread('img_process_0/lena.png')
b, g, r = cv2.split(img)  # 分离通道

# 显示各通道
cv2.imshow('Red Channel', r)   # 红色通道
cv2.imshow('Green Channel', g) # 绿色通道
cv2.imshow('Blue Channel', b)  # 蓝色通道
cv2.waitKey(0)
```

cv2.split(img)：将 RGB 图像拆分为 3 个单通道图像，分别对应 B、G、R 通道。要注意的是，这三个图像都是灰度图像，每个通道显示的是其强度值（0-255）：

- R 通道： 红色的强度值显示为白色，其他地方为黑色。
- G 通道： 绿色的强度值显示为白色，其他地方为黑色。
- B 通道： 蓝色的强度值显示为白色，其他地方为黑色。

由此可见，这种分离方式并不包含原始颜色信息，只是直观地表示每个通道的像素强度，而不是实际的颜色。单看代码和解释肯定很懵逼，又到了经典的 lena 时刻：

<div class="layout">

![1.12 Red通道](/img_process/01_basic/13_lena_R.png =240x)

![1.13 Green通道](/img_process/01_basic/14_lena_G.png =240x)

![1.14 Blue通道](/img_process/01_basic/15_lena_B.png =240x)

</div>

（2）通过 NumPy 提取通道的像素数组

```py
import cv2
img = cv2.imread('img_process_0/lena.png')

blue = img[:, :, 0]   # B 通道
green = img[:, :, 1]  # G 通道
red = img[:, :, 2]    # R 通道

print("Blue Channel:\n", blue)
print("Green Channel:\n", green)
print("Red Channel:\n", red)

# 部分结果
# Red Channel:
#  [[226 226 223 ... 230 221 200]
#  [226 226 223 ... 230 221 200]
#  [226 226 223 ... 230 221 200]
#  ...
#  [ 84  84  92 ... 173 172 177]
#  [ 82  82  96 ... 179 181 185]
#  [ 82  82  96 ... 179 181 185]]
```

（3）合并通道显示特定色相

合并通道使用 cv2.merge() 将某个通道的值保留，同时将其他通道的值设为 0，重新组合成一个新的 RGB 图像。

对比来看，分离通道直接显示灰度强度值，而合并通道用颜色直观展示实际色相，所以视觉效果差别明显。合并通道比单独通道灰度显示更直观，适合用于展示特定颜色成分。

```py
import cv2
import numpy as np

# 读取图像
img = cv2.imread('img_process_0/lena.png')

# 创建空白通道
zero_channel = np.zeros(img.shape[:2], dtype='uint8')

# 保留各通道的颜色
blue = cv2.merge([img[:, :, 0], zero_channel, zero_channel]) # 只保留蓝色
green = cv2.merge([zero_channel, img[:, :, 1], zero_channel]) # 只保留绿色
red = cv2.merge([zero_channel, zero_channel, img[:, :, 2]])  # 只保留红色

# 显示结果
cv2.imshow('Blue', blue)
cv2.imshow('Green', green)
cv2.imshow('Red', red)
cv2.waitKey(0)
```

<div class="layout">

![1.15 Red合并通道](/img_process/01_basic/16_lena_R.png =240x)

![1.16 Green合并通道](/img_process/01_basic/17_lena_G.png =240x)

![1.17 Blue合并通道](/img_process/01_basic/18_lena_B.png =240x)

</div>

（4）使用 Matplotlib 显示单通道图像

```py
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('img_process_0/lena.png')
b, g, r = cv2.split(img)

# plt显示部分
plt.figure(figsize=(10, 5))
plt.subplot(131), plt.imshow(b, cmap="Blues"), plt.title('Blue Channel')
plt.subplot(132), plt.imshow(g, cmap="Greens"), plt.title('Green Channel')
plt.subplot(133), plt.imshow(r, cmap="Reds"), plt.title('Red Channel')
plt.show()
```

首先，plt 的自动颜色映射（cmap="XXX"），让我感觉锐化很严重。但是 Chat-gpt 告诉我，这是因为梯度颜色映射会放大像素之间的细微差异，从而使边缘和纹理更明显，看起来像是经过了“锐化处理”。我更偏爱 imshow 的视觉观感。

![1.18 plt中显示单通道-1](/img_process/01_basic/19_plt1.png =560x)

另外，这标题明明是 “显示单通道图像” 对吧，怎么就变成 “合并通道” 的样子了。原因还是因为 plt 的自动颜色映射机制，我们只需要把 cmap 的值全部改为 gray 就可以了。

![1.19 plt中显示单通道-2](/img_process/01_basic/20_plt2.png =560x)

### 3.2 HSV 色彩空间

（1）什么是 HSV 色彩空间？

HSV（Hue, Saturation, Value）是基于人类视觉感知的色彩模型，比 RGB 色彩空间更适合处理颜色检测和分割任务。

- ==Hue(色调)==: 表示颜色的类型，以角度(0°-360°)表示，例如：红色为 0°，绿色为 120°，蓝色为 240°。
- ==Saturation(饱和度)==: 表示颜色的纯度，范围为 0%到 100%。
- ==Value(亮度)==: 表示颜色的亮度，范围为 0%到 100%。

<div class="layout">

![1.20 HSV图解1](/img_process/01_basic/21_hsv.webp =240x)

![1.21 HSV图解2](/img_process/01_basic/22_hsv1.png =240x)

</div>

（2）RGB 与 HSV 的相互转换

```py
import cv2
import matplotlib.pyplot as plt

# 读取图像并转换为HSV
img = cv2.imread('img_process_0/lena.png')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 显示图像
plt.figure(figsize=(10, 5))
plt.subplot(121), plt.imshow(img_rgb), plt.title('Original RGB')
plt.subplot(122), plt.imshow(hsv), plt.title('HSV')
plt.show()
```

![1.22 plt 中 RGB 和 HSV 的对比](/img_process/01_basic/23_hsv1.png =560x)

（3）常见颜色 HSV 范围参考表

| 颜色          | H (色调)  | S (饱和度) | V (亮度)  |
| ------------- | --------- | ---------- | --------- |
| 红色 (低范围) | 0 - 10    | 100 - 255  | 100 - 255 |
| 红色 (高范围) | 160 - 180 | 100 - 255  | 100 - 255 |
| 橙色          | 11 - 25   | 100 - 255  | 100 - 255 |
| 黄色          | 26 - 35   | 100 - 255  | 100 - 255 |
| 绿色          | 36 - 85   | 100 - 255  | 100 - 255 |
| 青色          | 86 - 100  | 100 - 255  | 100 - 255 |
| 蓝色          | 101 - 130 | 100 - 255  | 100 - 255 |
| 紫色          | 131 - 160 | 100 - 255  | 100 - 255 |

红色是一个比较特殊的范围，虽然我们看色环的时候，是收尾相接的，但是在 HSV 中并不能，如果需要同时提取高范围和低范围的红色，需要写两次范围分别提取。

另外，饱和度和亮度在实际的计算中取的是百分比，就比如 120/255=47%，中等程度的饱和度。之所以写的是 100 - 255，而不是 0 - 255，是因为我们在处理色彩提取的时候，通常只关心高饱和度和高亮度的范围。

（4）提取特定颜色范围

```py
import cv2
import numpy as np

# 读取图像并转换到HSV
img = cv2.imread('img_process_0/lena.png')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 定义紫色的HSV范围
lower_red = np.array([131, 120, 70])
upper_red = np.array([160, 255, 255])

# 提取紫色区域
mask = cv2.inRange(hsv, lower_red, upper_red)
result = cv2.bitwise_and(img, img, mask=mask)

# 显示结果
cv2.imshow('Original', img)
cv2.imshow('Mask', mask)
cv2.imshow('Result', result)
cv2.waitKey(0)
```

- cv2.inRange() 生成掩码，将 HSV 图像中特定范围的颜色提取出来。
- cv2.bitwise_and() 按位操作提取掩码指定的区域。

<div class="layout">

![1.22 HSV提取特定颜色1](/img_process/01_basic/24_lena1.png =240x)

![1.23 HSV提取特定颜色2](/img_process/01_basic/24_lena2.png =240x)

</div>

（5）HSV 色彩通道可视化

```py
import cv2
import matplotlib.pyplot as plt

# 读取图像并转换到HSV
img = cv2.imread('img_process_0/lena.png')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 分离HSV通道
h, s, v = cv2.split(hsv)

# 显示各通道
plt.figure(figsize=(12, 4))
plt.subplot(131), plt.imshow(h, cmap='hsv'), plt.title('Hue')
plt.subplot(132), plt.imshow(s, cmap='gray'), plt.title('Saturation')
plt.subplot(133), plt.imshow(v, cmap='gray'), plt.title('Value')
plt.show()
```

- Hue 通道显示色相信息，颜色范围对应色谱环。
- Saturation 和 Value 通道使用灰度显示数值强度。

![1.24 HSV三通道可视化](/img_process/01_basic/25_hsv.png =560x)
