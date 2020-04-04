//
//  PhoneNumberRecognizerBridge.h
//  TFLiteDemo
//
//  Created by JiongXing on 2019/5/17.
//  Copyright © 2019 LJX. All rights reserved.
//

#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>
#import "PhoneNumberRecognizerBridgeResult.h"

NS_ASSUME_NONNULL_BEGIN

@interface PhoneNumberRecognizerBridge : NSObject

- (void)start;

- (PhoneNumberRecognizerBridgeResult *)recognize:(UIImage *)image;

@end

NS_ASSUME_NONNULL_END
