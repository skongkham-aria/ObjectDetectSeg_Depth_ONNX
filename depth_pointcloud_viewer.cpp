// ================================================================
// Simple Point Cloud Viewer (OpenCV Basic)
// ================================================================
// Converts depth maps to point cloud data and displays as 2D projections
// Alternative to full 3D visualization when OpenCV Viz is not available
// 
// Features:
// - Real-time depth map to point cloud conversion
// - Multiple 2D projection views (Top, Side, Front)
// - Depth-based coloring
// 
// Usage: ./simple_pointcloud_viewer [model_path.onnx]
// Controls: 'q' to quit, '1'/'2'/'3' to switch views
// ================================================================

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <memory>

// Default model path
#define DEFAULT_MODEL_PATH "D:/Comp_Vision/ObjUltralytics_py/vits_qint8_sim_OP15.onnx"

// Forward declaration
void addGrid(cv::Mat& image, const std::string& title);

// Camera intrinsics (adjust these based on your camera)
struct CameraIntrinsics {
    float fx = 525.0f;  // Focal length X
    float fy = 525.0f;  // Focal length Y  
    float cx = 320.0f;  // Principal point X
    float cy = 240.0f;  // Principal point Y
    float scale = 1000.0f; // Depth scale factor
};

// Simple depth estimation class (reused from main file)
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
        inputSize = cv::Size(518, 518);
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

// Point Cloud 2D Projection Generator Class
class PointCloudProjector {
public:
    PointCloudProjector(const CameraIntrinsics& intrinsics) : intrinsics_(intrinsics) {}
    
    // Generate top-down view (X-Z plane)
    cv::Mat generateTopView(const cv::Mat& depthMap, int viewSize = 400) {
        cv::Mat topView = cv::Mat::zeros(viewSize, viewSize, CV_8UC3);
        
        // Find depth range for scaling
        double minDepth, maxDepth;
        cv::minMaxLoc(depthMap, &minDepth, &maxDepth);
        
        float scale = viewSize / 10.0f; // 10 meter range
        int centerX = viewSize / 2;
        int centerZ = viewSize / 2;
        
        for (int y = 0; y < depthMap.rows; y += 2) {
            for (int x = 0; x < depthMap.cols; x += 2) {
                float depth = depthMap.at<float>(y, x);
                if (depth <= 0 || depth > 10.0f) continue;
                
                // Convert to world coordinates
                float worldX = (x - intrinsics_.cx) * depth / intrinsics_.fx;
                float worldZ = depth;
                
                // Project to top view
                int projX = centerX + static_cast<int>(worldX * scale);
                int projZ = centerZ + static_cast<int>(worldZ * scale);
                
                if (projX >= 0 && projX < viewSize && projZ >= 0 && projZ < viewSize) {
                    cv::Vec3b color = getDepthColor(depth, minDepth, maxDepth);
                    topView.at<cv::Vec3b>(projZ, projX) = color;
                }
            }
        }
        
        return topView;
    }
    
    // Generate side view (Y-Z plane)
    cv::Mat generateSideView(const cv::Mat& depthMap, int viewSize = 400) {
        cv::Mat sideView = cv::Mat::zeros(viewSize, viewSize, CV_8UC3);
        
        double minDepth, maxDepth;
        cv::minMaxLoc(depthMap, &minDepth, &maxDepth);
        
        float scale = viewSize / 10.0f; // 10 meter range
        int centerY = viewSize / 2;
        int centerZ = viewSize / 2;
        
        for (int y = 0; y < depthMap.rows; y += 2) {
            for (int x = 0; x < depthMap.cols; x += 2) {
                float depth = depthMap.at<float>(y, x);
                if (depth <= 0 || depth > 10.0f) continue;
                
                // Convert to world coordinates
                float worldY = (y - intrinsics_.cy) * depth / intrinsics_.fy;
                float worldZ = depth;
                
                // Project to side view (flip Y for display)
                int projY = centerY - static_cast<int>(worldY * scale);
                int projZ = centerZ + static_cast<int>(worldZ * scale);
                
                if (projY >= 0 && projY < viewSize && projZ >= 0 && projZ < viewSize) {
                    cv::Vec3b color = getDepthColor(depth, minDepth, maxDepth);
                    sideView.at<cv::Vec3b>(projY, projZ) = color;
                }
            }
        }
        
        return sideView;
    }
    
    // Generate front view (X-Y plane)
    cv::Mat generateFrontView(const cv::Mat& depthMap, int viewSize = 700) {
        cv::Mat frontView = cv::Mat::zeros(viewSize, viewSize, CV_8UC3);
        
        double minDepth, maxDepth;
        cv::minMaxLoc(depthMap, &minDepth, &maxDepth);
        
        float scale = viewSize / 5.0f; // 5 meter range for X-Y
        int centerX = viewSize / 2;
        int centerY = viewSize / 2;
        
        for (int y = 0; y < depthMap.rows; y += 2) {
            for (int x = 0; x < depthMap.cols; x += 2) {
                float depth = depthMap.at<float>(y, x);
                if (depth <= 0 || depth > 10.0f) continue;
                
                // Convert to world coordinates
                float worldX = (x - intrinsics_.cx) * depth / intrinsics_.fx;
                float worldY = (y - intrinsics_.cy) * depth / intrinsics_.fy;

                // Project to front view (flip Y to show correct orientation)
                int projX = centerX + static_cast<int>(worldX * scale);
                int projY = centerY + static_cast<int>(worldY * scale);  // Changed from - to + to flip Y
                
                if (projX >= 0 && projX < viewSize && projY >= 0 && projY < viewSize) {
                    cv::Vec3b color = getDepthColor(depth, minDepth, maxDepth);
                    frontView.at<cv::Vec3b>(projY, projX) = color;
                }
            }
        }
        
        return frontView;
    }

private:
    CameraIntrinsics intrinsics_;
    
    cv::Vec3b getDepthColor(float depth, double minDepth, double maxDepth) {
        float normalizedDepth = (depth - minDepth) / (maxDepth - minDepth);
        return getJetColor(normalizedDepth);
    }
    
    cv::Vec3b getJetColor(float value) {
        value = std::max(0.0f, std::min(1.0f, value));
        
        float r, g, b;
        if (value < 0.25f) {
            r = 0; g = 4 * value; b = 1;
        } else if (value < 0.5f) {
            r = 0; g = 1; b = 1 - 4 * (value - 0.25f);
        } else if (value < 0.75f) {
            r = 4 * (value - 0.5f); g = 1; b = 0;
        } else {
            r = 1; g = 1 - 4 * (value - 0.75f); b = 0;
        }
        
        return cv::Vec3b(static_cast<uint8_t>(b * 255), 
                        static_cast<uint8_t>(g * 255), 
                        static_cast<uint8_t>(r * 255));
    }
};

int main(int argc, char* argv[]) {
    std::string modelPath = DEFAULT_MODEL_PATH;
    if (argc > 1) {
        modelPath = argv[1];
    }

    try {
        // Initialize depth estimator
        DepthEstimator depthEstimator(modelPath);
        
        // Camera intrinsics
        CameraIntrinsics intrinsics;
        intrinsics.fx = 525.0f;
        intrinsics.fy = 525.0f;
        intrinsics.cx = 320.0f;
        intrinsics.cy = 240.0f;
        
        // Initialize point cloud projector
        PointCloudProjector projector(intrinsics);
        
        // Open camera
        cv::VideoCapture cap(0);
        if (!cap.isOpened()) {
            std::cerr << "Error: Cannot open camera" << std::endl;
            return -1;
        }
        
        // Set camera resolution
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
        
        // Create windows
        cv::namedWindow("Camera", cv::WINDOW_AUTOSIZE);
        cv::namedWindow("Depth", cv::WINDOW_AUTOSIZE);
        // cv::namedWindow("Top View (X-Z)", cv::WINDOW_AUTOSIZE);
        // cv::namedWindow("Side View (Y-Z)", cv::WINDOW_AUTOSIZE);
        cv::namedWindow("Front View (X-Y)", cv::WINDOW_AUTOSIZE);
        
        std::cout << "Point Cloud 2D Projections Started" << std::endl;
        std::cout << "Controls:" << std::endl;
        std::cout << "  - 'q': Quit application" << std::endl;
        std::cout << "  - Views: Top (bird's eye), Side (profile), Front (face-on)" << std::endl;
        
        cv::Mat frame, depthMap;
        int frameCount = 0;
        
        while (true) {
            // Capture frame
            cap >> frame;
            if (frame.empty()) break;
            
            // Estimate depth
            depthMap = depthEstimator.predict(frame);
            
            // Generate projections (every few frames for performance)
            if (frameCount % 2 == 0) {
                try {
                    // cv::Mat topView = projector.generateTopView(depthMap);
                    // cv::Mat sideView = projector.generateSideView(depthMap);
                    cv::Mat frontView = projector.generateFrontView(depthMap);
                    
                    // Add grid lines for reference
                    // addGrid(topView, "Top View (X-Z)");
                    // addGrid(sideView, "Side View (Y-Z)");
                    addGrid(frontView, "Front View (X-Y)");
                    
                    // cv::imshow("Top View (X-Z)", topView);
                    // cv::imshow("Side View (Y-Z)", sideView);
                    cv::imshow("Front View (X-Y)", frontView);
                    
                } catch (const std::exception& e) {
                    std::cerr << "Projection error: " << e.what() << std::endl;
                }
            }
            
            // Display reference images
            cv::Mat depthDisplay;
            cv::normalize(depthMap, depthDisplay, 0, 255, cv::NORM_MINMAX, CV_8U);
            cv::applyColorMap(depthDisplay, depthDisplay, cv::COLORMAP_JET);
            
            cv::imshow("Camera", frame);
            cv::imshow("Depth", depthDisplay);
            
            // Handle events
            char key = cv::waitKey(1) & 0xFF;
            if (key == 'q' || key == 27) { // 'q' or ESC
                break;
            }
            
            frameCount++;
        }
        
        cv::destroyAllWindows();
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}

// Helper function to add grid lines
void addGrid(cv::Mat& image, const std::string& title) {
    int step = image.rows / 10;
    
    // Draw grid lines
    for (int i = step; i < image.rows; i += step) {
        cv::line(image, cv::Point(0, i), cv::Point(image.cols, i), cv::Scalar(128, 128, 128), 1);
    }
    for (int i = step; i < image.cols; i += step) {
        cv::line(image, cv::Point(i, 0), cv::Point(i, image.rows), cv::Scalar(128, 128, 128), 1);
    }
    
    // Add center cross
    cv::line(image, cv::Point(image.cols/2, 0), cv::Point(image.cols/2, image.rows), cv::Scalar(255, 255, 255), 2);
    cv::line(image, cv::Point(0, image.rows/2), cv::Point(image.cols, image.rows/2), cv::Scalar(255, 255, 255), 2);
    
    // Add title
    cv::putText(image, title, cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
}