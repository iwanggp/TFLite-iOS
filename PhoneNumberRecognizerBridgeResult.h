//
//  PhoneNumberRecognizerBridgeResult.h
//  TFLiteDemo
//
//  Created by JiongXing on 2019/5/20.
//  Copyright © 2019 LJX. All rights reserved.
//

#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>

NS_ASSUME_NONNULL_BEGIN

@interface PhoneNumberRecognizerBridgeResult : NSObject

@property (nonatomic, assign) BOOL success;
@property (nonatomic, assign) float meanConfidence;
@property (nonatomic, assign) float minConfidence;
@property (nonatomic, copy) NSString *text;
@property (nonatomic, copy) UIImage *numberArea;
@property (nonatomic, copy) UIImage *numberMerged;

@end

NS_ASSUME_NONNULL_END
