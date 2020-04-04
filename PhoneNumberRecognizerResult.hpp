//
//  PhoneNumberRecognizerResult.hpp
//  TFLiteDemo
//
//  Created by JiongXing on 2019/5/20.
//  Copyright © 2019 LJX. All rights reserved.
//

#ifndef PhoneNumberRecognizerResult_hpp
#define PhoneNumberRecognizerResult_hpp

#import <stdio.h>
#import "opencv2/imgproc/types_c.h"

class PhoneNumberRecognizerResult {
public:
    bool success = false;
    float meanConfidence = 0;
    float minConfidence = 0;
    std::string text = "";
    cv::Mat numberArea;
    cv::Mat numberMerged;
};

#endif /* PhoneNumberRecognizerResult_hpp */
