//
//  PhoneNumberRecognizer.cpp
//  TFLiteDemo
//
//  Created by JiongXing on 2019/5/16.
//  Copyright © 2019 LJX. All rights reserved.
//

#include "PhoneNumberRecognizer.hpp"
#import "tensorflow/lite/kernels/register.h"
#import "tensorflow/lite/op_resolver.h"
#import "tensorflow/lite/string_util.h"

void PhoneNumberRecognizer::Start(const char *modelPath) {
    this->model = tflite::FlatBufferModel::BuildFromFile(modelPath);
    this->model->error_reporter();
    printf("TFLite模型加载成功：%s\n", modelPath);
}

PhoneNumberRecognizerResult PhoneNumberRecognizer::Recognize(cv::Mat image) {
    PhoneNumberRecognizerResult result = PhoneNumberRecognizerResult();
    result.success = false;
    result.meanConfidence = 0;
    result.minConfidence = 0;
    // 灰度
    cv::Mat grayMat;
    if (image.channels() == 3) {
        cv::cvtColor(image, grayMat, CV_RGB2GRAY);
    } else if (image.channels() == 4) {
        cv::cvtColor(image, grayMat, CV_RGBA2GRAY);
    } else if (image.channels() == 1) {
        grayMat = image.clone();
    } else {
        return result;
    }
    //高斯模糊
//    cv::Mat gaussianMat;
//    cv::GaussianBlur(grayMat, gaussianMat, cv::Size(5, 5), 2);
//    // 二值化

    cv::Mat inversedThresholdMat;
    cv::adaptiveThreshold(grayMat, inversedThresholdMat, 255,
                          CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY_INV, 49, 13);
//    cv::inRange(inversedThresholdMat, cv::Scalar(0,0,0), cv::Scalar(85,85,85), inversedThresholdMat);

    // 二值化
//    cv::Mat thresholdMat;
//    cv::threshold(grayMat, thresholdMat, 0, 255, CV_THRESH_OTSU|CV_THRESH_BINARY_INV);
    // 去噪
    cv::Mat openedMat;
    cv::morphologyEx(inversedThresholdMat, openedMat, cv::MORPH_OPEN,
                     getStructuringElement(cv::MORPH_RECT, cv::Size(1, 1)),
                     cv::Point(-1, -1), 2);
    //横向腐蚀，隔开字符
    cv::Mat processingMat;
    cv::dilate(openedMat, processingMat, getStructuringElement(cv::MORPH_RECT, cv::Size(1, 1)),cv::Point(-1,-1), 1);
    // 水平连接字符
    cv::Mat closedMat;
    cv::morphologyEx(processingMat, closedMat, cv::MORPH_CLOSE,
                     getStructuringElement(cv::MORPH_RECT, cv::Size(20, 1)));
    
    // 检测疑似号码的区域
    std::vector<cv::Rect> rects = this->FindNumberAreas(closedMat);
    printf("检测手机号区域：%lu\n", rects.size());
    if (rects.size() <= 0) {
        return result;
    }
    // 记录最准确的识别结果
    float bestMeanConfidence = 0.0;
    float bestMinConfidence = 0.0;
    std::string bestText = "";
    cv::Mat bestArea;
    std::vector<cv::Mat> bestNumberMats;
    for (auto rect : rects) {
        // 切割出数字串区域
        cv::Mat numbers = processingMat(rect);
        // 扩充号串区域边界
        cv::Mat borderNumbers = this->MakeBorder(numbers, 2, false);
        // 切割字符
        std::vector<cv::Mat> numberMats = this->SplitNumbers(borderNumbers);
        printf("分割字符：%lu\n", numberMats.size());
        if (numberMats.size() < this->requireNumberCount / 2
            || numberMats.size() > this->requireNumberCount * 2) {
            continue;
        }
        std::vector<cv::Mat> effectedNumberMats;
        // 整串字符可信度均值
        float meanConfidence = 0.0;
        // 单字符可信度最低值
        float minConfidence = 0.0;
        std::string text;
        for (auto mat : numberMats) {
            //先将图片做腐蚀运算然后再添加边界
            cv::dilate(mat, mat,
                       getStructuringElement(cv::MORPH_RECT, cv::Size(1, 1)),cv::Point(-1,-1), 2);
            // 切割完后，数字会贴边，故扩充边界
            cv::Mat borderedNumber = this->MakeBorder(mat, 1, false);
            // 转为模型识别要求的尺寸
            cv::Mat squaredNumber = this->SquareImage(borderedNumber, 28, false);
            // 转为模型要求的白底黑字
            cv::Mat number;
            cv::bitwise_not(squaredNumber, number);
            // 识别
            Prediction pred = this->Predict(number);
            // 丢弃识别不到的结果，以及达不到最低可信度的字符
            if (pred.text.length() > 0 && pred.confidence > this->minSingleConfidence) {
                effectedNumberMats.push_back(number);
                meanConfidence += pred.confidence;
                if (minConfidence < 0.1 || pred.confidence < minConfidence) {
                    minConfidence = pred.confidence;
                }
                text.append(pred.text);
            }
        }
//        if (effectedNumberMats.size() != this->requireNumberCount) {
//            continue;
//        }
        meanConfidence = meanConfidence / (float)(effectedNumberMats.size());
        // 保存最优结果
        if (meanConfidence > bestMeanConfidence) {
            bestMeanConfidence = meanConfidence;
            bestMinConfidence = minConfidence;
            bestText = text;
            bestArea = borderNumbers;
            bestNumberMats = effectedNumberMats;
        }
    }
    result.success = bestMinConfidence > this->minSingleConfidence ? true : false;
    if (result.success) {
        result.meanConfidence = bestMeanConfidence;
        result.minConfidence = bestMinConfidence;
        // 特殊处理：允许第一个数字识别不出来，第一个数字固定为1
        result.text = bestText;
        result.numberArea = bestArea.clone();
        result.numberMerged = this->MergeNumbers(bestNumberMats).clone();
    }
    return result;
}

/// 找出符合数字串特征的区域
std::vector<cv::Rect> PhoneNumberRecognizer::FindNumberAreas(cv::Mat mat) {
    // 存储所有检测到的轮廓
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(mat, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    
    // 取矩形，大致筛选出号码区域。号码区域特征：长宽比大于4，小于12
    std::vector<cv::Rect> rects;
    for (int index = 0; index < contours.size(); index ++) {
        cv::Rect rect = cv::boundingRect(contours[index]);
        if (rect.x > 0 && rect.y > 0 && rect.height > 15
            && rect.width > rect.height * 5
            && rect.width < rect.height * 13) {
            rects.push_back(rect);
        }
    }
    return rects;
}

/// 分割出字符
std::vector<cv::Mat> PhoneNumberRecognizer::SplitNumbers(cv::Mat mat) {
    // 竖向膨胀
    cv::Mat dilateMat;
    cv::dilate(mat, dilateMat, getStructuringElement(cv::MORPH_RECT, cv::Size(1, 9)));
    // 存储所有检测到的轮廓
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    // 检测
    cv::findContours(dilateMat, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    std::vector<cv::Rect> rects;
    for (int index = 0; index < contours.size(); index ++) {
        cv::Rect rect = cv::boundingRect(contours[index]);
        rects.push_back(rect);
    }
    // 排序
    std::sort(rects.begin(), rects.end(), [](cv::Rect r1, cv::Rect r2) {
        return r1.x < r2.x;
    });
    // 切割字符
    std::vector<cv::Mat> numberMats;
    for (auto rect : rects) {
        numberMats.push_back(mat(rect));
    }
    return numberMats;
}

/// 添加外边界
cv::Mat PhoneNumberRecognizer::MakeBorder(cv::Mat mat, int pixel, bool white) {
    cv::Mat expandMat;
    cv::copyMakeBorder(mat.clone(), expandMat,
                       pixel, pixel, pixel, pixel,
                       cv::BORDER_CONSTANT, cv::Scalar(white ? 255 : 0));
    return expandMat;
}

/// 取固定尺寸的方形图片
cv::Mat PhoneNumberRecognizer::SquareImage(cv::Mat img, int size, bool white) {
    // 画布
    cv::Mat square(size, size, CV_8UC1, cv::Scalar(white ? 255 : 0));
    
    int width = img.cols;
    int height = img.rows;
    int max_dim = (width >= height) ? width : height;
    float scale = ((float)size) / max_dim;
    
    cv::Rect roi;
    if (width >= height) {
        roi.width = size;
        roi.x = 0;
        roi.height = height * scale;
        roi.y = (size - roi.height) / 2;
    } else {
        roi.y = 0;
        roi.height = size;
        roi.width = width * scale;
        roi.x = (size - roi.width) / 2;
    }
    cv::resize(img, square(roi), roi.size());
    return square;
}

/// 把多张字符图合并到一张输出。要求输入为单通道；其输出结果也是单通道合成图
cv::Mat PhoneNumberRecognizer::MergeNumbers(std::vector<cv::Mat> numbers) {
    // 画布大小
    int cols = 0;
    int rows = 0;
    // 给每个字符图片加1像素边框
    std::vector<cv::Mat> borderNumbers;
    for (auto number : numbers) {
        cv::Mat borderMat = this->MakeBorder(number, 1, false);
        borderNumbers.push_back(borderMat);
        // 累加总宽高
        cols += borderMat.cols;
        if (borderMat.rows > rows) {
            rows = borderMat.rows;
        }
    }
    // 画布
    cv::Mat merged(rows, cols, CV_8UC1, cv::Scalar(255));
    int widthOffset = 0;
    for (auto number : borderNumbers) {
        cv::Mat roi = merged(cv::Rect(widthOffset, 0, number.cols, number.rows));
        number.copyTo(roi);
        widthOffset += number.cols;
    }
    return merged;
}

/// 返回值Prediction：text为nil表示识别不到数字或星号
PhoneNumberRecognizer::Prediction PhoneNumberRecognizer::Predict(cv::Mat mat) {
    // Create an interpreter
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder(*(this->model), resolver)(&interpreter);
    assert(interpreter);
    interpreter->SetNumThreads(1);
    
    // Obtains the input buffer from the interpreter
    int inputTensorIndex = interpreter->inputs()[0];
    int imageWidth = 28;
    int imageHeight = 28;
    std::vector<int> sizes = {1, imageHeight, imageWidth, 1};
    interpreter->ResizeInputTensor(inputTensorIndex, sizes);
    
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        printf("Failed to allocate tensors.");
        return PhoneNumberRecognizer::Prediction();
    }
    
    // Get the pointer to the input buffer.
    float* inputBuffer = interpreter->typed_tensor<float>(inputTensorIndex);
    // Feed data.归一化
    uchar* imageData = mat.data;
    int numPixel = imageWidth * imageHeight;
    for (int i = 0; i < numPixel; i++) {
        inputBuffer[i] = (float)(imageData[i]) / 255.0;
    }
    
    // 运行模型
    interpreter->Invoke();
    
    // Get the index of first output tensor.
    const int outputTensorIndex = interpreter->outputs()[0];
    // Get the pointer to the output buffer.
    float* outputBuffer = interpreter->typed_tensor<float>(outputTensorIndex);
    
    // 类别数量，0-9对应0至9，10-*，11-其它
//    int numClass = 12;
    int numClass = 11;
    // 选出一行中的最大值
    float maxValue = 0.0;
    // 最大值所在的Index
    int maxValueIndex = -1;
    for (int index = 0; index < numClass; index ++) {
        float value = outputBuffer[index];
        printf("%.3f ", value);
        if (value > maxValue) {
            maxValue = value;
            maxValueIndex = index;
        }
    }
    printf("-> [%d, %.3f]\n", maxValueIndex, maxValue);
    PhoneNumberRecognizer::Prediction pred = PhoneNumberRecognizer::Prediction();
    pred.confidence = maxValue;
    if (maxValueIndex == 10) {
        pred.text = "*";
    } else if (maxValueIndex >= 0 && maxValueIndex <= 9) {
        pred.text = std::to_string(maxValueIndex);
    }
    return pred;
}
