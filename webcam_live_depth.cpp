// ================================================================
// Simple Webcam Depth Estimation
// ================================================================
// Efficient real-time depth estimation from webcam using ONNX model
// Shows original camera feed and depth map in separate windows
// 
// Usage: ./webcam_depth [model_path.onnx]
// Controls: Press 'q' to quit
// ================================================================

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <memory>

// Default model path
#define DEFAULT_MODEL_PATH "D:/Comp_Vision/ObjUltralytics_py/vits_qint8_sim_OP15.onnx"

// Simple depth estimation class
class DepthEstimator {
public:
    DepthEstimator(const std::string &modelPath);
    cv::Mat predict(const cv::Mat &image);

private:
    Ort::Env env{ORT_LOGGING_LEVEL_ERROR, "DepthEstimator"};
    Ort::SessionOptions sessionOptions;
    std::unique_ptr<Ort::Session> session;
    cv::Size inputSize;
    std::vector<const char*> inputNames;
    std::vector<const char*> outputNames;

    cv::Mat preprocess(const cv::Mat &image);
    cv::Mat postprocess(const cv::Mat &depthOutput, const cv::Size &originalSize);
};

DepthEstimator::DepthEstimator(const std::string &modelPath) {
    // Setup session options
    sessionOptions.SetIntraOpNumThreads(1);
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    
    // Try CUDA if available
    try {
        OrtCUDAProviderOptions cudaOptions;
        sessionOptions.AppendExecutionProvider_CUDA(cudaOptions);
    } catch (...) {
        // Fallback to CPU
    }

    // Create session
    session = std::make_unique<Ort::Session>(env, 
        std::wstring(modelPath.begin(), modelPath.end()).c_str(), sessionOptions);

    // Get input/output info
    Ort::AllocatorWithDefaultOptions allocator;
    
    // Input name
    char* inputName = session->GetInputName(0, allocator);
    inputNames.push_back(inputName);
    
    // Output name  
    char* outputName = session->GetOutputName(0, allocator);
    outputNames.push_back(outputName);

    // Input shape
    auto inputTypeInfo = session->GetInputTypeInfo(0);
    auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
    auto inputShape = inputTensorInfo.GetShape();
    
    if (inputShape.size() >= 4) {
        inputSize = cv::Size(518, 518); // Default size
    }
}

cv::Mat DepthEstimator::preprocess(const cv::Mat &image) {
    cv::Mat resized;
    cv::resize(image, resized, inputSize);
    
    cv::Mat floatImage;
    resized.convertTo(floatImage, CV_32FC3, 1.0f / 255.0f);
    
    // Normalize (ImageNet normalization)
    cv::Scalar mean(0.485, 0.456, 0.406);
    cv::Scalar stddev(0.229, 0.224, 0.225);
    floatImage = (floatImage - mean) / stddev;
    
    return floatImage;
}

cv::Mat DepthEstimator::postprocess(const cv::Mat &depthOutput, const cv::Size &originalSize) {
    cv::Mat depth;
    cv::resize(depthOutput, depth, originalSize);
    return depth;
}

cv::Mat DepthEstimator::predict(const cv::Mat &image) {
    // Preprocess
    cv::Mat processed = preprocess(image);
    
    // Convert to blob
    std::vector<cv::Mat> channels;
    cv::split(processed, channels);
    
    std::vector<float> blob;
    for (auto &channel : channels) {
        blob.insert(blob.end(), (float*)channel.datastart, (float*)channel.dataend);
    }
    
    // Create input tensor
    std::vector<int64_t> inputShape = {1, 3, inputSize.height, inputSize.width};
    Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memInfo, blob.data(), blob.size(), inputShape.data(), inputShape.size());
    
    // Run inference
    auto outputTensors = session->Run(Ort::RunOptions{nullptr}, 
        inputNames.data(), &inputTensor, 1, outputNames.data(), 1);
    
    // Get output
    const float* outputData = outputTensors[0].GetTensorData<float>();
    auto outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
    
    cv::Mat depthMap(outputShape[1], outputShape[2], CV_32FC1, (void*)outputData);
    return postprocess(depthMap.clone(), image.size());
}

int main(int argc, char* argv[]) {
    std::string modelPath = DEFAULT_MODEL_PATH;
    if (argc > 1) {
        modelPath = argv[1];
    }

    try {
        // Initialize depth estimator
        DepthEstimator depthEstimator(modelPath);
        
        // Open camera
        cv::VideoCapture cap(0);
        if (!cap.isOpened()) {
            std::cerr << "Error: Cannot open camera" << std::endl;
            return -1;
        }
        
        // Set camera resolution for performance
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
        
        // Create windows
        cv::namedWindow("Raw Depth", cv::WINDOW_AUTOSIZE);
        cv::namedWindow("Depth", cv::WINDOW_AUTOSIZE);
        
        cv::Mat frame, depthMap;
        
        while (true) {
            // Capture frame
            cap >> frame;
            if (frame.empty()) break;
            
            // Estimate depth
            depthMap = depthEstimator.predict(frame);
            
            // Normalize depth for display
            cv::Mat depthDisplay;
            cv::normalize(depthMap, depthDisplay, 0, 255, cv::NORM_MINMAX, CV_8U);
            cv::applyColorMap(depthDisplay, depthDisplay, cv::COLORMAP_JET);

            // Show the raw depth map
            cv::imshow("Raw Depth", depthMap);

            // Display both windows
            // cv::imshow("Webcam", frame);
            cv::imshow("Depth", depthDisplay);
            
            // Exit on 'q' key
            if (cv::waitKey(1) == 'q') break;
        }
        
        cv::destroyAllWindows();
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}