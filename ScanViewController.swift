//
//  ScanViewController.swift
//  TFLiteDemo
//
//  Created by JiongXing on 2019/4/1.
//  Copyright © 2019 LJX. All rights reserved.
//

import UIKit
import AVFoundation
import CoreGraphics

class ScanViewController: UIViewController {
    
    lazy var captureDevice = AVCaptureDevice.default(for: AVMediaType.video)
    
    /// 视频帧率
    private var timescale: Int32 = 30
    /// 帧计数
    private var sampleCount = -1
    /// 每秒采样数
    private var samplePerSecond = 10
    /// 每多少个结果输出一次最佳值
    private var bestResultDuration = 3
    /// 结果计数
    private var resultCount = -1
    /// 最佳结果保存
    private var bestResult = PhoneNumberRecognizerBridgeResult()
    
    private let lock = NSLock()
    
    private let imageView = UIImageView()
    private let imageView2 = UIImageView()
    
    private let label = UILabel()
    
    private let metricsLabel = UILabel()
    
    private let button = UIButton(type: .custom)
    
    private var previewLayer: AVCaptureVideoPreviewLayer?
    
    private var focusFrame: CGRect?
    
    private let ciContext = CIContext()
    
    private let recognizer = PhoneNumberRecognizerBridge()
    
    private var stop = true

    override func viewDidLoad() {
        super.viewDidLoad()
        
        let width = focusRect(for: view.bounds).size.width + 30
        let x = (view.bounds.width - width) / 2
        imageView.frame = CGRect(x: x, y: 30, width: width, height: 50)
        imageView.contentMode = .scaleAspectFit
        imageView.backgroundColor = .white
        view.addSubview(imageView)
        
        imageView2.frame = CGRect(x: x, y: imageView.frame.maxY,
                                  width: width, height: 40)
        imageView2.contentMode = .scaleAspectFit
        imageView2.backgroundColor = .white
        view.addSubview(imageView2)
        
        label.frame = CGRect(x: x, y: imageView2.frame.maxY,
                             width: width, height: 30)
        label.font = UIFont.systemFont(ofSize: 16)
        label.textColor = UIColor.black
        label.textAlignment = .center
        label.backgroundColor = UIColor.white
        label.numberOfLines = 0
        view.addSubview(label)
        
        metricsLabel.frame = CGRect(x: x, y: label.frame.maxY,
                                    width: width, height: 30)
        metricsLabel.font = UIFont.systemFont(ofSize: 16)
        metricsLabel.textColor = UIColor.black
        metricsLabel.textAlignment = .center
        metricsLabel.backgroundColor = UIColor.white
        metricsLabel.adjustsFontSizeToFitWidth = true
        view.addSubview(metricsLabel)
        
        // 暂停按钮
        button.setTitleColor(.blue, for: .normal)
        button.titleLabel?.font = UIFont.boldSystemFont(ofSize: 17)
        button.addTarget(self, action: #selector(onButton), for: .touchUpInside)
        button.frame.size = CGSize(width: 100, height: 40)
        view.addSubview(button)
        onButton()

        recognizer.start()
        do {
            try setupScan()
        } catch {
            print(error)
        }
    }
    
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        navigationController?.setNavigationBarHidden(true, animated: true)
    }
    
    @objc private func onButton() {
        stop = !stop
        button.setTitle(stop ? "【开 始】": "【暂 停】", for: .normal)
        button.frame.origin.x = (view.bounds.width - button.frame.width) / 2.0
        button.frame.origin.y = view.bounds.height / 4 * 3
    }
    
    private func setupScan() throws {
        guard let device = AVCaptureDevice.default(for: AVMediaType.video) else {
            return
        }
        // 帧率
        do {
            try device.lockForConfiguration()
            device.activeVideoMinFrameDuration = CMTime.init(value: 1, timescale: timescale)
            device.unlockForConfiguration()
        } catch {
            print("device lock error:")
            print(error)
        }
        
        // 输入
        let input = try AVCaptureDeviceInput.init(device: device)
        
        // 输出
        let output = AVCaptureVideoDataOutput.init()
        output.alwaysDiscardsLateVideoFrames = true
        output.setSampleBufferDelegate(self, queue: DispatchQueue.global())
        output.videoSettings = [
            (kCVPixelBufferPixelFormatTypeKey as String): kCVPixelFormatType_32BGRA
        ]
        if let connection = output.connection(with: AVMediaType.video) {
            if connection.isVideoOrientationSupported {
                connection.videoOrientation = .portrait
            }
        }
        
        // 会话
        let session = AVCaptureSession.init()
        if session.canSetSessionPreset(.high) {
            session.sessionPreset = .high
        }
        if session.canAddInput(input) {
            session.addInput(input)
        }
        if session.canAddOutput(output) {
            session.addOutput(output)
        }
        
        let fdesc = device.activeFormat.formatDescription
        let dims = CMVideoFormatDescriptionGetDimensions(fdesc)
        NSLog("%d x %d", dims.width, dims.height)
        
        // 预览层
        let previewLayer = AVCaptureVideoPreviewLayer.init(session: session)
        previewLayer.frame = view.bounds
        previewLayer.videoGravity = AVLayerVideoGravity.resizeAspectFill
        view.layer.insertSublayer(previewLayer, at: 0)
        let focusFrame = addFocusLayer(to: previewLayer)
        print("focusFrame:\(focusFrame)")
        self.previewLayer = previewLayer
        self.focusFrame = focusFrame
        // 开启会话
        session.startRunning()
    }
    
    private func addFocusLayer(to layer: CALayer) -> CGRect {
        let focusLayer = CALayer.init()
        focusLayer.borderWidth = 1
        focusLayer.borderColor = UIColor.blue.cgColor
        focusLayer.backgroundColor = UIColor.clear.cgColor
        focusLayer.frame = focusRect(for: layer.bounds)
        layer.addSublayer(focusLayer)
        
        let line = CALayer.init()
        line.backgroundColor = UIColor.yellow.cgColor
        let width: CGFloat = focusLayer.frame.width * 0.7
        let x: CGFloat = focusLayer.frame.midX - (width / 2)
        let y: CGFloat = focusLayer.frame.midY
        line.frame = CGRect(x: x, y: y, width: width, height: 1)
        layer.addSublayer(line)
        return focusLayer.frame
    }
}

extension ScanViewController: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        if stop {
            return
        }
        sampleCount += 1
        if (sampleCount > timescale) {
            sampleCount = 1
        }
        // 采样条件
        guard sampleCount % samplePerSecond == 0 else {
            return
        }
        if connection.isVideoOrientationSupported {
            connection.videoOrientation = .portrait
        }
        let beginTime = Date().timeIntervalSince1970
        guard let imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return
        }
        let ciImg = CIImage.init(cvPixelBuffer: imageBuffer)
        if ciImg.extent.size == CGSize.zero {
            return
        }
        let fRect = focusRect(for: ciImg.extent, cg: true)
        
        guard let cgImg = ciContext.createCGImage(ciImg, from: fRect) else {
            return
        }
        let cropImg = UIImage(cgImage: cgImg)
        // 识别图片
        let result = self.recognizer.recognize(cropImg)
        DispatchQueue.main.async { [weak self] in
            self?.handleResult(beginTime: beginTime, result: result)
        }
    }
    
    private func handleResult(beginTime: TimeInterval, result: PhoneNumberRecognizerBridgeResult) {
        lock.lock()
        imageView.image = result.numberArea
        imageView2.image = result.numberMerged
        label.text = result.text
        metricsLabel.text = String(
            format: "最低:%.2f 均值:%.2f 耗时:%0.2fms",
            result.minConfidence, result.meanConfidence,
            (Date().timeIntervalSince1970 - beginTime) * 1000)
        lock.unlock()
    }
    
    private func handleTimesResult(beginTime: TimeInterval, result: PhoneNumberRecognizerBridgeResult) {
        lock.lock()
        resultCount += 1
        if resultCount < bestResultDuration {
            // 更新最佳结果
            if result.success && result.meanConfidence > bestResult.minConfidence {
                bestResult = result
            }
        } else if resultCount == bestResultDuration {
            // 到达周期统计点
            imageView.image = bestResult.numberArea
            imageView2.image = bestResult.numberMerged
            label.text = result.text
            metricsLabel.text = String(
                format: "最低:%.2f 均值:%.2f 耗时:%0.2fms",
                bestResult.minConfidence, bestResult.meanConfidence,
                (Date().timeIntervalSince1970 - beginTime) * 1000)
        } else {
            // 开始新周期
            resultCount = 0
            bestResult = PhoneNumberRecognizerBridgeResult()
        }
        lock.unlock()
    }
    
    private func focusRect(for bounds: CGRect, cg: Bool = false) -> CGRect {
        let width: CGFloat = 240 * bounds.width / 375
        let height: CGFloat = 50 * bounds.height / 667
        let x: CGFloat = (bounds.width - width) / 2
        var y: CGFloat = bounds.height / 3
        if cg {
            // 坐标系关于x轴对称，y值从底部往上算
            y = bounds.height - (y + height)
        }
        return CGRect(x: x, y: y, width: width, height: height)
    }
}
