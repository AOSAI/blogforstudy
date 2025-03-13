---
title: 图像处理入门指南
order: 0
author: AOSAI
date: 2024-10-30
category:
  - 图像处理入门
tag:
  - OpenCV安装
---

## 1. 序言

这个系列内容的名字我纠结了好久，**计算机视觉（CV）**，**图像处理**，**OpenCV**。其实叫什么无所谓，本质上就是对图像操作的学习。

记录这个专题的起因是，我这学期选的三门课：“计算机视觉”、“医疗画像处理”、“人工智能”，都是在讲 CV 的内容。“计算机视觉”偏实战，“医疗画像处理”偏模型理论，“人工智能”偏科普。

整体学下来，对计算机视觉的认知清晰了不少，所以趁热打铁，写一波博客。希望整理的过程中，再深入以及扩展一些内容。

## 2. C++安装使用 OpenCV

### 2.1 安装 Visual Studio 2022

1. 打开 [Visual Studio 官网](https://visualstudio.microsoft.com/zh-hans/vs/)，在 **下载** 下拉框中选择 **社区版 2022（Community 2022）**。

![1.1 下载 Visual Studio](/img_process/00_guide/00-01.png)

2. 打开下载的 exe 文件，根据你的开发需求选择安装。一般来说，如果只是为了用 C++结合 OpenCV 写点东西，选择一个 **使用 C++的桌面开发** 就够了，右边栏里的选项默认即可。

![1.2 安装C++](/img_process/00_guide/00-02.png)

3. 选择安装路径时，需要注意两点：（1）最好不要安装在 C 盘，并且不要出现中文路径。（2）缓存文件和安装文件不能放在同一个文件夹里，最简单的办法就是，只改变根路径（比如 C:\Program Files\ ......），子路径保持不变（变为 D:\VS\ ......）。

![1.3 选择安装路径](/img_process/00_guide/00-03.png)

4. 安装好之后，登录与否看你心情。创建一个新项目 --> 选择 **空项目** --> 设置项目路径 --> 在文件结构中选择 **源文件**，右键创建一个 cpp 文件，输入以下代码，测试是否能够成功运行。

```C++
#include <stdio.h>
using namespace std;

int main()
{
    printf("Hello World");
    return 0;
}
```

### 2.2 下载 OpenCV

打开 C++所使用的 OpenCV 资源网站：[OpenCV 下载中心](https://opencv.org/releases/)，下载一个你需要的版本。没有具体要求，就下载倒数第二新的版本，防止出错。

![1.4 安装OpenCV](/img_process/00_guide/00-04.png)

就比如这里，最新的是 4.10.0 版本，我们就下载 4.9.0 版本的 windows 安装包就行。在运行下载好的 opencv-4.9.0-windows.exe 文件时，除了选择安装位置外，其余都默认。==假设我们安装在 D:\OpenCV 路径下==

### 2.3 IDE 项目中配置 OpenCV

1. 点击刚才创建好的项目，再点击上方的 **小扳手** 按钮，打开项目的属性页。

![1.5 安装OpenCV](/img_process/00_guide/00-05.png)

2. 找到 **CV++目录** 中的 **外部包含目录** 和 **库目录**。将 OpenCV 安装路径下的这两个路径，分别写入。

- D:\OpenCV\opencv\build\include --> 外部包含目录
- D:\OpenCV\opencv\build\x64\vc16\lib --> 库目录

![1.6 C++项目配置OpenCV1](/img_process/00_guide/00-06.png)

以第一个为例，**外部包含目录**的最右侧有一个下拉框 --> 点击 **编辑** --> 粘贴**D:\OpenCV\opencv\build\include** --> 点击 **确认按钮**，完成操作。

3. 找到 **链接器** 中的 **输入**，第一项为 **附加默认项**，同样的操作，把 OpenCV 刚才 \build\x64\vc16\lib 路径下的 opencv_world4100d.lib 文件路径，复制粘贴进去。

- "D:\OpenCV\opencv\build\x64\vc16\lib\opencv_world4100d.lib"

==需要注意的是：==

- opencv_world4100d.lib 文件表示在 Debug 模式下运行
- opencv_world4100.lib 文件表示在 Release 模式下运行

我在图 1.5 和图 1.6 中已经用橙色的方框圈起来了，VS 中默认运行的是 Debug 模式。如果你需要在 Release 模式下运行，请在属性页，配置处，修改为 Release，然后重复上述步骤。

4. 复制粘贴以下代码，如果不报错，并且能成功运行，即为该项目配置 OpenCV 成功。

```C++
#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main()
{
    // 输出 OpenCV 版本
    printf(CV_VERSION);
    return 0;
}
```

## 3. Python 安装使用 OpenCV

1. 下载安装一个 VS code。

2. 安装 Anaconda3 或者 只安装 Python 基本包。[《Anaconda 下载地址》](https://www.anaconda.com)

3. 安装 OpenCV 只有一句话：（要注意的是，即使是 Anaconda，也是使用 pip，而不是用 conda 去下载）

```py
pip install opencv-python
```

4. 验证安装是否成功：新建一个 python 文件，输入以下代码，运行。

```py
import cv2
print(cv2.__version__)
```
