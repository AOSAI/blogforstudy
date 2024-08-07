---
title: Numpy之拜入宗门
order: 0
author: AOSAI
date: 2023-11-13
category:
  - 机器学习
tag:
  - 机器学习
  - Numpy
---

## 序言

Numpy 的内容整理是基于 [Numpy 官方中文文档](https://www.numpy.org.cn/)完成的。为什么写这个，其一是我在看官方文档的时候，可能很多内容是收藏的一些博主的，所以看的过程中，内容有些冗余；其二就是我需要复习，整理有利于梳理逻辑。希望也能帮助到看到我博客的朋友。

## 简介

Numpy 是一个功能强大的 Python 库，主要用于对多维数组（矩阵）执行计算。Numpy 这个词来源于两个单词 -- Numerical（数字的、以数字表示的）和 Python。这类数值计算广泛用于以下任务：

- **机器学习模型：** 在编写机器学习算法时，需要对矩阵进行各种数值计算。例如矩阵乘法、换位、加法等。NumPy 提供了一个非常好的库，用于简单(在编写代码方面)和快速(在速度方面)计算。NumPy 数组用于存储训练数据和机器学习模型的参数。

- **图像处理和计算机图形学：** 计算机中的图像表示为多维数字数组。NumPy 成为同样情况下最自然的选择。实际上，NumPy 提供了一些优秀的库函数来快速处理图像。例如，镜像图像、按特定角度旋转图像等。

- **数学任务：** NumPy 对于执行各种数学任务非常有用，如数值积分、微分、内插、外推等。因此，当涉及到数学任务时，它形成了一种基于 Python 的 MATLAB 的快速替代。

## 安装

情况一：电脑中单独装了 Python，并且配置了相关环境，这时候这样安装。

```py
pip install numpy
```

情况二：电脑中安装了 Anaconda，并配置了相关环境。再往前不知道，但是 23 年这两次 Anaconda 的库中直接内置了 Numpy 和其相关的库，配置好环境后可以直接查看其版本。

```py
# 列出所有安装好的包名
pip list  # == conda list

# 模糊查询，列出包含 关键词xxx 的所有包名
pip list xxx  # == conda list xxx
```

情况三：项目需要统一 Python 和 Numpy 库的版本，需要指定版本安装。

```py
# version为需要被指定的版本号
pip install numpy=version
```
