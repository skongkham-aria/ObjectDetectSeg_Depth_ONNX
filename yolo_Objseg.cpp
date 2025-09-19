#include <iostream>
#include <vector>
#include <string>
#include <numeric>
#include <algorithm>
#include <random>

// ADDED: For std::exception
#include <stdexcept>

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

// Class names from utils.py
const std::vector<std::string> CLASS_NAMES = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"};

// Utility function to convert xywh to xyxy format
cv::Rect xywh2xyxy(float x, float y, float w, float h)
{
    int x1 = static_cast<int>(x - w / 2);
    int y1 = static_cast<int>(y - h / 2);
    int x2 = static_cast<int>(x + w / 2);
    int y2 = static_cast<int>(y + h / 2);
    return cv::Rect(x1, y1, x2 - x1, y2 - y1);
}

// Sigmoid function
float sigmoid(float x)
{
    return 1.0f / (1.0f + exp(-x));
}

class YOLOSeg
{
public:
    YOLOSeg(const std::string &modelPath, float confThreshold, float iouThreshold);
    ~YOLOSeg();
    void detect(cv::Mat &image);
    const std::vector<cv::Mat>& getMasks() const { return maskMaps; }

private:
    void initializeModel(const std::string &modelPath);
    void preprocess(cv::Mat &image);
    void postprocess(cv::Mat &image, std::vector<Ort::Value> &outputs);
    void drawDetections(cv::Mat &image);

    Ort::Env env;
    Ort::Session session{nullptr};
    Ort::AllocatorWithDefaultOptions allocator;

    std::vector<char*> allocatedInputNames;
    std::vector<char*> allocatedOutputNames;

    std::vector<const char *> inputNames;
    std::vector<const char *> outputNames;
    int64_t inputWidth, inputHeight;
    cv::Size2f modelInputShape;
    std::vector<float> inputTensor;
    cv::Mat originalImage;
    float confThreshold;
    float iouThreshold;
    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> classIds;
    std::vector<cv::Mat> maskMaps;
    std::vector<cv::Scalar> colors;
};

YOLOSeg::YOLOSeg(const std::string &modelPath, float confThresh, float iouThresh)
    : env(ORT_LOGGING_LEVEL_WARNING, "YOLOSeg"), confThreshold(confThresh), iouThreshold(iouThresh)
{
    initializeModel(modelPath);
    std::mt19937 rng(3);
    std::uniform_int_distribution<int> dist(0, 255);
    for (size_t i = 0; i < CLASS_NAMES.size(); ++i)
    {
        colors.emplace_back(dist(rng), dist(rng), dist(rng));
    }
}

YOLOSeg::~YOLOSeg()
{
    for (auto name : allocatedInputNames) allocator.Free(name);
    for (auto name : allocatedOutputNames) allocator.Free(name);
}

void YOLOSeg::initializeModel(const std::string &modelPath)
{
    Ort::SessionOptions sessionOptions;
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    #ifdef _WIN32
        const std::wstring modelPathW = std::wstring(modelPath.begin(), modelPath.end());
        session = Ort::Session(env, modelPathW.c_str(), sessionOptions);
    #else
        session = Ort::Session(env, modelPath.c_str(), sessionOptions);
    #endif

    char* inputName = session.GetInputName(0, allocator);
    allocatedInputNames.push_back(inputName);
    inputNames.push_back(inputName);

    Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(0);
    auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
    auto inputDims = inputTensorInfo.GetShape();
    inputHeight = inputDims[2];
    inputWidth = inputDims[3];
    modelInputShape = cv::Size2f(static_cast<float>(inputWidth), static_cast<float>(inputHeight));

    size_t numOutputNodes = session.GetOutputCount();
    for (size_t i = 0; i < numOutputNodes; i++)
    {
        char* outputName = session.GetOutputName(i, allocator);
        allocatedOutputNames.push_back(outputName);
        outputNames.push_back(outputName);
    }
}

void YOLOSeg::preprocess(cv::Mat &image)
{
    originalImage = image.clone();
    cv::Mat resizedImage;
    cv::cvtColor(image, resizedImage, cv::COLOR_BGR2RGB);
    cv::resize(resizedImage, resizedImage, cv::Size(static_cast<int>(inputWidth), static_cast<int>(inputHeight)));
    resizedImage.convertTo(resizedImage, CV_32F, 1.0 / 255.0);

    std::vector<cv::Mat> channels(3);
    cv::split(resizedImage, channels);
    
    inputTensor.assign((float*)channels[0].datastart, (float*)channels[0].dataend);
    inputTensor.insert(inputTensor.end(), (float*)channels[1].datastart, (float*)channels[1].dataend);
    inputTensor.insert(inputTensor.end(), (float*)channels[2].datastart, (float*)channels[2].dataend);
}

// Post processing Mask map
void YOLOSeg::postprocess(cv::Mat &image, std::vector<Ort::Value> &outputs)
{
    boxes.clear(); scores.clear(); classIds.clear(); maskMaps.clear();

    const float *boxOutput = outputs[0].GetTensorData<float>();
    auto boxOutputShape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    int numClasses = static_cast<int>(boxOutputShape[1]) - 4 - 32;
    int numDetections = static_cast<int>(boxOutputShape[2]);

    std::vector<int> tempClassIds;
    std::vector<float> tempScores;
    std::vector<cv::Rect> tempBoxes;
    std::vector<cv::Mat> tempMaskPreds;

    for (int i = 0; i < numDetections; ++i)
    {
        const float* p_class = boxOutput + 4 * numDetections + i;
        int class_id = 0;
        float max_score = 0;
        for (int j = 0; j < numClasses; ++j) {
            if (p_class[j * numDetections] > max_score) {
                max_score = p_class[j * numDetections];
                class_id = j;
            }
        }

        if (max_score > confThreshold)
        {
            tempScores.push_back(max_score);
            tempClassIds.push_back(class_id);

            float x = boxOutput[0 * numDetections + i];
            float y = boxOutput[1 * numDetections + i];
            float w = boxOutput[2 * numDetections + i];
            float h = boxOutput[3 * numDetections + i];
            
            float gain_w = (float)image.cols / inputWidth;
            float gain_h = (float)image.rows / inputHeight;
            x *= gain_w; y *= gain_h; w *= gain_w; h *= gain_h;
            
            tempBoxes.push_back(xywh2xyxy(x, y, w, h));
            
            cv::Mat maskPred(1, 32, CV_32F);
            for(int j=0; j < 32; ++j) {
                maskPred.at<float>(0, j) = boxOutput[(4 + numClasses + j) * numDetections + i];
            }
            tempMaskPreds.push_back(maskPred);
        }
    }
    
    std::vector<int> indices;
    cv::dnn::NMSBoxes(tempBoxes, tempScores, confThreshold, iouThreshold, indices);
    if (indices.empty()) return;

    const float *maskOutput = outputs[1].GetTensorData<float>();
    auto maskOutputShape = outputs[1].GetTensorTypeAndShapeInfo().GetShape();
    int maskHeight = static_cast<int>(maskOutputShape[2]);
    int maskWidth = static_cast<int>(maskOutputShape[3]);
    cv::Mat maskProto(32, maskHeight * maskWidth, CV_32F, (void*)maskOutput);

    for (int idx : indices)
    {
        boxes.push_back(tempBoxes[idx]);
        scores.push_back(tempScores[idx]);
        classIds.push_back(tempClassIds[idx]);

        cv::Mat matmulResult = tempMaskPreds[idx] * maskProto;
        for (int j = 0; j < matmulResult.cols; ++j) matmulResult.at<float>(0, j) = sigmoid(matmulResult.at<float>(0, j));

        cv::Mat finalMask = matmulResult.reshape(1, maskHeight);
        
        cv::Rect original_box = tempBoxes[idx];

        float scale_w = (float)maskWidth / image.cols;
        float scale_h = (float)maskHeight / image.rows;
        cv::Rect scaled_box;
        scaled_box.x = static_cast<int>(original_box.x * scale_w);
        scaled_box.y = static_cast<int>(original_box.y * scale_h);
        scaled_box.width = static_cast<int>(original_box.width * scale_w);
        scaled_box.height = static_cast<int>(original_box.height * scale_h);
        scaled_box.x = std::max(0, std::min(maskWidth - 1, scaled_box.x));
        scaled_box.y = std::max(0, std::min(maskHeight - 1, scaled_box.y));
        scaled_box.width = std::max(1, std::min(maskWidth - scaled_box.x, scaled_box.width));
        scaled_box.height = std::max(1, std::min(maskHeight - scaled_box.y, scaled_box.height));

        cv::Mat cropped_low_res_mask = finalMask(scaled_box);
        cv::Mat upscaled_cropped_mask;
        cv::resize(cropped_low_res_mask, upscaled_cropped_mask, original_box.size(), 0, 0, cv::INTER_CUBIC);

        cv::Mat mask_map = cv::Mat::zeros(image.rows, image.cols, CV_8UC1);
        cv::Rect safe_box = original_box & cv::Rect(0, 0, image.cols, image.rows);
        
        if (safe_box.width > 0 && safe_box.height > 0) {
            cv::Mat dest_roi = mask_map(safe_box);
            cv::Mat source_roi = upscaled_cropped_mask(cv::Rect(0,0, safe_box.width, safe_box.height));
            
            // --- START OF CHANGED LOGIC ---

            // 1. Create an 8-bit binary mask from the float mask by applying the threshold.
            //    Pixels > 0.5 will become 255 (white), others will be 0 (black).
            cv::Mat binary_mask = source_roi > 0.5;

            // 2. Copy this new binary mask to the destination. This avoids the float-to-int truncation.
            binary_mask.copyTo(dest_roi);

            // --- END OF CHANGED LOGIC ---
        }

        maskMaps.push_back(mask_map);
    }
}




void YOLOSeg::drawDetections(cv::Mat &image)
{
    // Clone the image to draw on
    cv::Mat maskedImage = image.clone();

    // The main alpha value for blending the masks
    const float mask_alpha = 0.5;

    // Draw the masks for each detection
    for (size_t i = 0; i < boxes.size(); ++i)
    {
        // Ensure the mask is valid
        if (maskMaps[i].size() != image.size() || maskMaps[i].type() != CV_8UC1) continue;

        // Get the color for the current class
        cv::Scalar color = colors[classIds[i]];
        
        // Find all non-zero pixels in the mask
        cv::Mat mask_pixels;
        cv::findNonZero(maskMaps[i], mask_pixels);

        // --- START OF CHANGED LOGIC ---

        // Iterate through the matrix of points using a standard for loop
        for (int j = 0; j < mask_pixels.rows; ++j) {
            // Get the point at the current row
            cv::Point p = mask_pixels.at<cv::Point>(j);
            
            // Get the original pixel color
            cv::Vec3b& image_pixel = image.at<cv::Vec3b>(p);
            
            // Use 'double' to match cv::Scalar's type and resolve warnings
            double mask_b = color[0];
            double mask_g = color[1];
            double mask_r = color[2];

            // Apply the alpha blending formula: blended = original * (1-alpha) + overlay * alpha
            maskedImage.at<cv::Vec3b>(p)[0] = cv::saturate_cast<uchar>(image_pixel[0] * (1 - mask_alpha) + mask_b * mask_alpha);
            maskedImage.at<cv::Vec3b>(p)[1] = cv::saturate_cast<uchar>(image_pixel[1] * (1 - mask_alpha) + mask_g * mask_alpha);
            maskedImage.at<cv::Vec3b>(p)[2] = cv::saturate_cast<uchar>(image_pixel[2] * (1 - mask_alpha) + mask_r * mask_alpha);
        }

        // --- END OF CHANGED LOGIC ---
    }

    // Draw the bounding boxes and labels on top of the masked image
    for (size_t i = 0; i < boxes.size(); ++i)
    {
        cv::rectangle(maskedImage, boxes[i], colors[classIds[i]], 2);

        std::string label = CLASS_NAMES[classIds[i]] + " " + std::to_string(static_cast<int>(scores[i] * 100)) + "%";
        int baseLine;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        cv::rectangle(maskedImage,
                      cv::Point(boxes[i].x, boxes[i].y - labelSize.height - baseLine),
                      cv::Point(boxes[i].x + labelSize.width, boxes[i].y),
                      colors[classIds[i]], cv::FILLED);
        
        cv::putText(maskedImage, label, cv::Point(boxes[i].x, boxes[i].y - baseLine),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    }
    
    // Replace the original image with the final result
    image = maskedImage;
}



void YOLOSeg::detect(cv::Mat &image)
{
    preprocess(image);
    std::array<int64_t, 4> inputShape = {1, 3, inputHeight, inputWidth};
    auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value inputOrtTensor = Ort::Value::CreateTensor<float>(memoryInfo, inputTensor.data(), inputTensor.size(), inputShape.data(), inputShape.size());
    auto outputTensors = session.Run(Ort::RunOptions{nullptr}, inputNames.data(), &inputOrtTensor, 1, outputNames.data(), outputNames.size());
    postprocess(originalImage, outputTensors);
    drawDetections(originalImage);
    image = originalImage;
}

// ======================================================================================
// MAIN FUNCTION - UPDATED WITH ERROR HANDLING
// ======================================================================================
// int main(int argc, char **argv)
// int main(int argc, char **argv)
int main()
{
    // // --- 1. Argument Parsing ---
    // if (argc < 2)
    // {
    //     std::cerr << "Usage: " << argv[0] << " <path_to_onnx_model>" << std::endl;
    //     return -1;
    // }
    // std::string modelPath = argv[1];
    std::string modelPath = "D:/Comp_Vision/ObjUltralytics_py/yolo11m-seg.onnx";
    std::cout << "DEBUG: Using model path: " << modelPath << std::endl;

    // --- 2. Main Try-Catch Block ---
    // This will catch any major errors during initialization or the main loop.
    try
    {
        // --- 3. Model Initialization ---
        std::cout << "DEBUG: Initializing YOLO Segmentation model..." << std::endl;
        YOLOSeg yoloseg(modelPath, 0.3f, 0.5f);
        std::cout << "DEBUG: Model loaded successfully." << std::endl;

        // --- 4. Webcam Initialization ---
        std::cout << "DEBUG: Opening webcam..." << std::endl;
        cv::VideoCapture cap(0);
        if (!cap.isOpened())
        {
            // This is a critical error, so we throw an exception.
            throw std::runtime_error("FATAL: Could not open webcam.");
        }
        std::cout << "DEBUG: Webcam opened successfully." << std::endl;

        // --- 5. Main Processing Loop ---
        cv::Mat frame;
        int frame_count = 0;
        while (true)
        {
            cap.read(frame);
            if (frame.empty())
            {
                std::cerr << "WARN: Blank frame grabbed. End of stream?" << std::endl;
                break;
            }

            // Perform detection on the frame
            yoloseg.detect(frame);

            const auto& masks = yoloseg.getMasks();
            if (!masks.empty()) {
                // Show mask directly.
                cv::imshow("Segmented Mask (Debug)", masks[0]);
            }
            
            // Display results
            cv::imshow("YOLO Segmentation - C++", frame);
            
            if (cv::waitKey(1) == 27) // Exit on ESC key
            {
                std::cout << "DEBUG: ESC key pressed. Exiting." << std::endl;
                break;
            }
            frame_count++;
        }
        cap.release();
        cv::destroyAllWindows();
    }
    catch (const Ort::Exception& e)
    {
        // This will catch errors specifically from the ONNX Runtime library.
        // The most likely cause is a problem with the model file.
        std::cerr << "\nONNX RUNTIME ERROR: " << e.what() << std::endl;
        std::cerr << "Please check if the model path is correct and the ONNX model file is valid." << std::endl;
        return -1;
    }
    catch (const std::exception& e)
    {
        // This will catch other standard errors, like the webcam failing to open.
        std::cerr << "\nERROR: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}