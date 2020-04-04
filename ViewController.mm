//
//  ViewController.m
//  TFLiteDemo
//
//  Created by JiongXing on 2019/3/22.
//  Copyright © 2019 LJX. All rights reserved.
//

#import "ViewController.h"

#include <stdio.h>
#include <pthread.h>
#include <unistd.h>
#include <fstream>
#include <iostream>
#include <queue>
#include <sstream>
#include <string>

#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/op_resolver.h"
#include "tensorflow/lite/string_util.h"

#import "ios_image_load.h"

#import <opencv2/imgproc/types_c.h>
#import <opencv2/imgcodecs/ios.h>

using namespace cv;

@interface ViewController ()

@property (weak, nonatomic) IBOutlet UIImageView *imageView;

@end

@implementation ViewController {
//    std::unique_ptr<tflite::Interpreter> _interpreter;
}

- (void)viewDidLoad {
    [super viewDidLoad];
    self.imageView.layer.borderWidth = 1 / UIScreen.mainScreen.scale;
    self.imageView.layer.borderColor = UIColor.redColor.CGColor;
    
    UIBarButtonItem *resetItem = [[UIBarButtonItem alloc] initWithTitle:@"重置" style:UIBarButtonItemStylePlain target:self action:@selector(onReset)];
    UIBarButtonItem *fullItem = [[UIBarButtonItem alloc] initWithTitle:@"完整" style:UIBarButtonItemStylePlain target:self action:@selector(onFull)];
    UIBarButtonItem *grayItem = [[UIBarButtonItem alloc] initWithTitle:@"灰度" style:UIBarButtonItemStylePlain target:self action:@selector(onGray)];
    UIBarButtonItem *threshold = [[UIBarButtonItem alloc] initWithTitle:@"二值" style:UIBarButtonItemStylePlain target:self action:@selector(onThreshold)];
    UIBarButtonItem *erodeItem = [[UIBarButtonItem alloc] initWithTitle:@"腐蚀" style:UIBarButtonItemStylePlain target:self action:@selector(onErode)];
    UIBarButtonItem *dilateItem = [[UIBarButtonItem alloc] initWithTitle:@"膨胀" style:UIBarButtonItemStylePlain target:self action:@selector(onDilate)];
    UIBarButtonItem *contourItem = [[UIBarButtonItem alloc] initWithTitle:@"边缘" style:UIBarButtonItemStylePlain target:self action:@selector(findContours)];
    
    self.navigationItem.leftBarButtonItems = @[fullItem, grayItem, threshold];
    self.navigationItem.rightBarButtonItems = @[contourItem, dilateItem, erodeItem];
    
//    [self initResources];
    [self onReset];
}

- (void)onReset {
    UIImage *imgOri = [UIImage imageNamed:@"pic1"];
    self.imageView.image = imgOri;
}

- (void)onFull {
    NSTimeInterval begin = [[NSDate date] timeIntervalSince1970];
    [self onGray];
    [self onThreshold];
    [self onErode];
    [self onDilate];
    [self findContours];
    NSLog(@"耗时：%.0f", ([[NSDate date] timeIntervalSince1970] - begin) * 1000);
}

- (void)onGray {
    UIImage *imgOri = self.imageView.image;
    
    // UIImage -> 矩阵
    Mat mat_image;
    UIImageToMat(imgOri, mat_image);
    if (mat_image.empty()) {
        NSLog(@"Mat Empty!");
        return;
    }
    NSLog(@"Channels is %@ !", @(mat_image.channels()));
    if (mat_image.channels() < 3) {
        NSLog(@"已是灰图");
        return;
    }
    
    // 灰度图片
    // 参数：数据源，目标数据，转换类型
    Mat mat_image_dst;
    cvtColor(mat_image, mat_image_dst, CV_BGR2GRAY);
    
    // 矩阵 -> UIImage
    self.imageView.image = MatToUIImage(mat_image_dst);
    NSLog(@"灰度完成。");
}

- (void)onThreshold {
    UIImage *imgOri = self.imageView.image;
    
    // UIImage -> 矩阵
    Mat mat_image;
    UIImageToMat(imgOri, mat_image);
    if (mat_image.empty()) {
        NSLog(@"mat_image empty!");
        return;
    }
    NSLog(@"Channels is %@ !", @(mat_image.channels()));
    if (mat_image.channels() > 1) {
        NSLog(@"要求是单通道图像！");
        return;
    }
    
    Mat mat_image_dst;
    adaptiveThreshold(mat_image, mat_image_dst, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY_INV, 25, 25);
    if (mat_image_dst.empty()) {
        NSLog(@"mat_image_dst empty!");
        return;
    }
    
    // 矩阵 -> UIImage
    self.imageView.image = MatToUIImage(mat_image_dst);
    NSLog(@"二值化完成。");
}

- (void)onErode {
    UIImage *imgOri = self.imageView.image;
    
    // UIImage -> 矩阵
    Mat mat_image;
    UIImageToMat(imgOri, mat_image);
    if (mat_image.empty()) {
        NSLog(@"mat_image empty!");
        return;
    }
    NSLog(@"Channels is %@ !", @(mat_image.channels()));
    if (mat_image.channels() > 1) {
        NSLog(@"要求是单通道图像！");
        return;
    }
    
    Mat mat_image_dst;
    Mat erodeKernel = getStructuringElement(MORPH_RECT, cv::Size(2, 2));
    erode(mat_image, mat_image_dst, erodeKernel);
    
    // 矩阵 -> UIImage
    self.imageView.image = MatToUIImage(mat_image_dst);
    NSLog(@"腐蚀完成。");
}

- (void)onDilate {
    UIImage *imgOri = self.imageView.image;
    
    // UIImage -> 矩阵
    Mat mat_image;
    UIImageToMat(imgOri, mat_image);
    if (mat_image.empty()) {
        NSLog(@"mat_image empty!");
        return;
    }
    NSLog(@"Channels is %@ !", @(mat_image.channels()));
    if (mat_image.channels() > 1) {
        NSLog(@"要求是单通道图像！");
        return;
    }
    
    Mat mat_image_dst;
    Mat erodeKernel = getStructuringElement(MORPH_RECT, cv::Size(2, 2));
    dilate(mat_image, mat_image_dst, erodeKernel);
    
    // 矩阵 -> UIImage
    self.imageView.image = MatToUIImage(mat_image_dst);
    NSLog(@"膨胀完成。");
}

- (void)findContours {
    UIImage *imgOri = self.imageView.image;
    
    // UIImage -> 矩阵
    Mat mat_image;
    UIImageToMat(imgOri, mat_image);
    if (mat_image.empty()) {
        NSLog(@"mat_image empty!");
        return;
    }
    NSLog(@"Channels is %@ !", @(mat_image.channels()));
    if (mat_image.channels() > 1) {
        NSLog(@"要求是单通道图像！");
        return;
    }
    
    // 存储所有检测到的轮廓
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    
    findContours(mat_image, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    
    printf("contours.size: %lu\n", contours.size());
    
    if (contours.size() < 6) {
        NSLog(@"识别数字太少");
        return;
    }
    
    std::vector<cv::Rect> rects;
    for (int index = 0; index < contours.size(); index ++) {
        cv::Rect rect = cv::boundingRect(contours[index]);
        printf("Rect x:%d y:%d w:%d h:%d\n", rect.x, rect.y, rect.width, rect.height);
        rects.push_back(rect);
        if (rect.x == 0 && rect.y == 0) {
            continue;
        }
        UIImageView *view = [[UIImageView alloc] initWithFrame:CGRectMake(rect.x, rect.y, rect.width, rect.height)];
        view.layer.borderColor = UIColor.blueColor.CGColor;
        view.layer.borderWidth = 1 / UIScreen.mainScreen.scale;
        [self.imageView addSubview:view];
        
        Mat singleNum = mat_image(rect);
        UIImage *singleImg = MatToUIImage(singleNum);
        view.image = singleImg;
    }
    
    printf("\n----- Sort -----\n");
    std::sort(rects.begin(), rects.end(), ASCSort);
    for (auto rect : rects) {
        printf("Rect x:%d y:%d w:%d h:%d\n", rect.x, rect.y, rect.width, rect.height);
        
        Mat singleNum = mat_image(rect);
        Mat resizeMat = GetSquareImage(singleNum);
        Mat mat_inv;
        cv::bitwise_not(resizeMat, mat_inv);
        [self predict:mat_inv];
    }
}

bool ASCSort(cv::Rect r1, cv::Rect r2) {
    return r1.x < r2.x;
}

cv::Mat GetSquareImage( const cv::Mat& img, int target_width = 28 )
{
    int width = img.cols,
    height = img.rows;
    
    cv::Mat square = cv::Mat::zeros( target_width, target_width, img.type() );
    
    int max_dim = ( width >= height ) ? width : height;
    float scale = ( ( float ) target_width ) / max_dim;
    cv::Rect roi;
    if ( width >= height )
    {
        roi.width = target_width;
        roi.x = 0;
        roi.height = height * scale;
        roi.y = ( target_width - roi.height ) / 2;
    }
    else
    {
        roi.y = 0;
        roi.height = target_width;
        roi.width = width * scale;
        roi.x = ( target_width - roi.width ) / 2;
    }
    
    cv::resize( img, square( roi ), roi.size() );
    
    return square;
}

std::vector<uint8_t> MatToVector(cv::Mat mat) {
    std::vector<uint8_t> array;
    if (mat.isContinuous()) {
        array.assign((uint8_t*)mat.datastart, (uint8_t*)mat.dataend);
    } else {
        for (int i = 0; i < mat.rows; ++i) {
            array.insert(array.end(), mat.ptr<uint8_t>(i), mat.ptr<uint8_t>(i)+mat.cols);
        }
    }
    return array;
}

- (void)predict:(cv::Mat)mat {
    NSString* graph = @"only_dig";
    const int num_threads = 1;
    std::string input_layer_type = "float";
    std::vector<int> sizes = {1, 28, 28, 1};
    
    const NSString* graph_path = FilePathForResourceName(graph, @"tflite");
    
    std::unique_ptr<tflite::FlatBufferModel> model(tflite::FlatBufferModel::BuildFromFile([graph_path UTF8String]));
    if (!model) {
        NSLog(@"Failed to mmap model %@.", graph);
        exit(-1);
    }
    NSLog(@"Loaded model %@.", graph);
    model->error_reporter();
    NSLog(@"Resolved reporter.");
    
#ifdef TFLITE_CUSTOM_OPS_HEADER
    tflite::MutableOpResolver resolver;
    RegisterSelectedOps(&resolver);
#else
    tflite::ops::builtin::BuiltinOpResolver resolver;
#endif
    
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    if (!interpreter) {
        NSLog(@"Failed to construct interpreter.");
        exit(-1);
    }
    
    if (num_threads != -1) {
        interpreter->SetNumThreads(num_threads);
    }
    
    int input = interpreter->inputs()[0];
    
    if (input_layer_type != "string") {
        interpreter->ResizeInputTensor(input, sizes);
    }
    
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        NSLog(@"Failed to allocate tensors.");
        exit(-1);
    }
    
//    cv::imwrite("/Users/jiongxing/Desktop/TFLiteDemo/outputs/img.png", mat);
    
    int image_width = 28;
    int image_height = 28;
    int image_channels = 1;
    std::vector<uint8_t> image_data = MatToVector(mat);
//    LoadImageFromFile("/Users/jiongxing/Desktop/TFLiteDemo/outputs/img.png", &image_width, &image_height, &image_channels);
    
    const int wanted_width = 28;
    const int wanted_height = 28;
    const int wanted_channels = 1;
    const float input_mean = 127.5f;
    const float input_std = 127.5f;
    assert(image_channels >= wanted_channels);
    uint8_t* in = image_data.data();
    float* out = interpreter->typed_tensor<float>(input);
    for (int y = 0; y < wanted_height; ++y) {
        const int in_y = (y * image_height) / wanted_height;
        uint8_t* in_row = in + (in_y * image_width * image_channels);
        float* out_row = out + (y * wanted_width * wanted_channels);
        for (int x = 0; x < wanted_width; ++x) {
            const int in_x = (x * image_width) / wanted_width;
            uint8_t* in_pixel = in_row + (in_x * image_channels);
            float* out_pixel = out_row + (x * wanted_channels);
            for (int c = 0; c < wanted_channels; ++c) {
                out_pixel[c] = (in_pixel[c] - input_mean) / input_std;
            }
        }
    }
    
    if (interpreter->Invoke() != kTfLiteOk) {
        NSLog(@"Failed to invoke!");
        exit(-1);
    }
    
    float* output = interpreter->typed_output_tensor<float>(0);
    const int output_size = 10;
    const int kNumResults = 1;
    const float kThreshold = 0.1f;
    std::vector<std::pair<float, int> > top_results;
    GetTopN(output, output_size, kNumResults, kThreshold, &top_results);
    
    std::stringstream ss;
    ss.precision(3);
    for (const auto& result : top_results) {
        const float confidence = result.first;
        const int index = result.second;
        
        ss << index << " " << confidence << "  ";
        
        ss << "\n";
    }
    
    std::string predictions = ss.str();
    NSString* result = @"";
    result = [NSString stringWithFormat:@"%@ - %s", result, predictions.c_str()];
    NSLog(@"Predictions: %@", result);
}

NSString* FilePathForResourceName(NSString* name, NSString* extension) {
    NSString* file_path = [[NSBundle mainBundle] pathForResource:name ofType:extension];
    if (file_path == NULL) {
        NSLog(@"Couldn't find '%@.%@' in bundle.", name, extension);
        exit(-1);
    }
    return file_path;
}

// Returns the top N confidence values over threshold in the provided vector,
// sorted by confidence in descending order.
static void GetTopN(const float* prediction, const int prediction_size, const int num_results,
                    const float threshold, std::vector<std::pair<float, int> >* top_results) {
    // Will contain top N results in ascending order.
    std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int> >,
    std::greater<std::pair<float, int> > >
    top_result_pq;
    
    const long count = prediction_size;
    for (int i = 0; i < count; ++i) {
        const float value = prediction[i];
        
        // Only add it if it beats the threshold and has a chance at being in
        // the top N.
        if (value < threshold) {
            continue;
        }
        
        top_result_pq.push(std::pair<float, int>(value, i));
        
        // If at capacity, kick the smallest value out.
        if (top_result_pq.size() > num_results) {
            top_result_pq.pop();
        }
    }
    
    // Copy to output vector and reverse into descending order.
    while (!top_result_pq.empty()) {
        top_results->push_back(top_result_pq.top());
        top_result_pq.pop();
    }
    std::reverse(top_results->begin(), top_results->end());
}

NSString* RunInferenceOnImage() {
    NSString* graph = @"digital";
    const int num_threads = 1;
    std::string input_layer_type = "float";
    std::vector<int> sizes = {1, 28, 28, 1};
    
    const NSString* graph_path = FilePathForResourceName(graph, @"tflite");
    
    std::unique_ptr<tflite::FlatBufferModel> model(tflite::FlatBufferModel::BuildFromFile([graph_path UTF8String]));
    if (!model) {
        NSLog(@"Failed to mmap model %@.", graph);
        exit(-1);
    }
    NSLog(@"Loaded model %@.", graph);
    model->error_reporter();
    NSLog(@"Resolved reporter.");
    
#ifdef TFLITE_CUSTOM_OPS_HEADER
    tflite::MutableOpResolver resolver;
    RegisterSelectedOps(&resolver);
#else
    tflite::ops::builtin::BuiltinOpResolver resolver;
#endif
    
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    if (!interpreter) {
        NSLog(@"Failed to construct interpreter.");
        exit(-1);
    }
    
    if (num_threads != -1) {
        interpreter->SetNumThreads(num_threads);
    }
    
    int input = interpreter->inputs()[0];
    
    if (input_layer_type != "string") {
        interpreter->ResizeInputTensor(input, sizes);
    }
    
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        NSLog(@"Failed to allocate tensors.");
        exit(-1);
    }
    
    // Read the Grace Hopper image.
    NSString* image_path = FilePathForResourceName(@"img008-00008", @"png");
    int image_width;
    int image_height;
    int image_channels;
    std::vector<uint8_t> image_data =
    LoadImageFromFile([image_path UTF8String], &image_width, &image_height, &image_channels);
    const int wanted_width = 28;
    const int wanted_height = 28;
    const int wanted_channels = 1;
    const float input_mean = 127.5f;
    const float input_std = 127.5f;
    assert(image_channels >= wanted_channels);
    uint8_t* in = image_data.data();
    float* out = interpreter->typed_tensor<float>(input);
    for (int y = 0; y < wanted_height; ++y) {
        const int in_y = (y * image_height) / wanted_height;
        uint8_t* in_row = in + (in_y * image_width * image_channels);
        float* out_row = out + (y * wanted_width * wanted_channels);
        for (int x = 0; x < wanted_width; ++x) {
            const int in_x = (x * image_width) / wanted_width;
            uint8_t* in_pixel = in_row + (in_x * image_channels);
            float* out_pixel = out_row + (x * wanted_channels);
            for (int c = 0; c < wanted_channels; ++c) {
                out_pixel[c] = (in_pixel[c] - input_mean) / input_std;
            }
        }
    }
    
    if (interpreter->Invoke() != kTfLiteOk) {
        NSLog(@"Failed to invoke!");
        exit(-1);
    }
    
    float* output = interpreter->typed_output_tensor<float>(0);
    const int output_size = 10;
    const int kNumResults = 1;
    const float kThreshold = 0.1f;
    std::vector<std::pair<float, int> > top_results;
    GetTopN(output, output_size, kNumResults, kThreshold, &top_results);
    
    std::stringstream ss;
    ss.precision(3);
    for (const auto& result : top_results) {
        const float confidence = result.first;
        const int index = result.second;
        
        ss << index << " " << confidence << "  ";
        
        ss << "\n";
    }
    
    std::string predictions = ss.str();
    NSString* result = @"";
    result = [NSString stringWithFormat:@"%@ - %s", result, predictions.c_str()];
    NSLog(@"Predictions: %@", result);
    return result;
}


@end
