# TFLite-iOS(由于保密这里上传部分code)
TFLite模型在iOS平台上的实践
## 机打数字号码模型训练
用Tensorflow在服务器上训练个机打手机号码识别的模型

### 数据集准备
数据集为使用的**Chars74K**公开数据集和通过**sketch**工具人工制作的图片

### 模型选型
选择合适的深度学习模型来训练机打号码识别，训练好的模型为**TF**格式。
### 模型转换
上面我们已经获得了TensorFlow训练好的模型了，这个格式的模型是不能直接在移动端平台使用的，这里就需要将其进行转换了。详情可以参照[TFLite官方文档](https://tensorflow.google.cn/lite)这里有很详细的介绍如何使用该工具。以及压缩和量化操作。

## iOS平台的开发
### OpenCV库的开发
这里可以通过C++直接调用OpenCV库，然后在根据所需要的方法进行裁剪。用OC的方式调用OpenCV库，那速度不知比Android快多少的。

### iOS相关的开发
可以工程中的具体相关代码，这里包含了数据的预处理以及识别结果的处理。

## 安装清单
##### 环境
* keras >= 1.4.0
* TensorFlow >= 1.12.0
* OpenCV
* Swift
* C++

