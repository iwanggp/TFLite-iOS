//
//  PhoneNumberRecognizerBridge.m
//  TFLiteDemo
//
//  Created by JiongXing on 2019/5/17.
//  Copyright © 2019 LJX. All rights reserved.
//

#import "PhoneNumberRecognizerBridge.h"
#import "PhoneNumberRecognizer.hpp"
#import "opencv2/imgcodecs/ios.h"

@implementation PhoneNumberRecognizerBridge {
    // C++对象
    PhoneNumberRecognizer _recognizer;
}

- (void)start {
    _recognizer = PhoneNumberRecognizer();
    NSString *modelPath = [[NSBundle mainBundle] pathForResource:@"model_v10" ofType:@"tflite"];
    _recognizer.Start([modelPath UTF8String]);
}

- (PhoneNumberRecognizerBridgeResult *)recognize:(UIImage *)image {
    cv::Mat inputMat;
    UIImageToMat(image, inputMat);
    PhoneNumberRecognizerResult result = _recognizer.Recognize(inputMat);
    PhoneNumberRecognizerBridgeResult *bridgeResult = [[PhoneNumberRecognizerBridgeResult alloc] init];
    bridgeResult.success = result.success;
    if (bridgeResult.success == false) {
        return bridgeResult;
    }
    bridgeResult.text = [NSString stringWithUTF8String:result.text.c_str()];
    bridgeResult.meanConfidence = result.meanConfidence;
    bridgeResult.minConfidence = result.minConfidence;
    bridgeResult.numberArea = MatToUIImage(result.numberArea);
    bridgeResult.numberMerged = MatToUIImage(result.numberMerged);
    return bridgeResult;
}

@end
