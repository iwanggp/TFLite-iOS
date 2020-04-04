//
//  PhoneNumberRecognizer.hpp
//  TFLiteDemo
//
//  Created by JiongXing on 2019/5/16.
//  Copyright © 2019 LJX. All rights reserved.
//

#ifndef PhoneNumberRecognizer_hpp
#define PhoneNumberRecognizer_hpp

#import <stdio.h>
#import "opencv2/imgproc/types_c.h"
#import "tensorflow/lite/model.h"
#import "PhoneNumberRecognizerResult.hpp"

class PhoneNumberRecognizer {
public:
    /// 期望数字个数
    int requireNumberCount = 11;
    
    /// 单字符识别可信度。小于此可信度的识别结果将被丢弃
    float minSingleConfidence = 0.7;
    
    /// 启动实例，传入模型文件路径
    void Start(const char *modelPath);
    
    /// 识别图片
    PhoneNumberRecognizerResult Recognize(cv::Mat image);
    
private:
    struct Prediction {
        std::string text = "";
        float confidence = 0.0;
    };
    
    std::unique_ptr<tflite::FlatBufferModel> model;
    
    std::vector<cv::Rect> FindNumberAreas(cv::Mat mat);
    std::vector<cv::Mat> SplitNumbers(cv::Mat mat);
    cv::Mat MakeBorder(cv::Mat mat, int pixel, bool white);
    cv::Mat SquareImage(cv::Mat img, int size,  bool white);
    cv::Mat MergeNumbers(std::vector<cv::Mat> numbers);
    PhoneNumberRecognizer::Prediction Predict(cv::Mat mat);
};

#endif /* PhoneNumberRecognizer_hpp */
