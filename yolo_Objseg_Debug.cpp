#include <iostream>
#include <vector>
#include <numeric>
#include <iomanip> // For std::fixed and std::setprecision
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

// COCO class labels
const std::vector<std::string> CLASS_NAMES = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
};

// --- NEW: Function to visualize the 32 mask prototypes ---
void visualize_prototypes(const float* mask_data, const std::vector<int64_t>& mask_shape) {
    int num_prototypes = static_cast<int>(mask_shape[1]);
    int mask_height = static_cast<int>(mask_shape[2]);
    int mask_width = static_cast<int>(mask_shape[3]);

    // Create a canvas to tile the prototypes
    int grid_cols = 8;
    int grid_rows = (num_prototypes + grid_cols - 1) / grid_cols;
    cv::Mat canvas = cv::Mat::zeros(grid_rows * mask_height, grid_cols * mask_width, CV_8UC1);

    for (int i = 0; i < num_prototypes; ++i) {
        // Create a Mat header for the current prototype
        cv::Mat prototype(mask_height, mask_width, CV_32F, (void*)(mask_data + i * mask_height * mask_width));
        
        // Normalize the prototype to 0-255 to make it visible
        cv::Mat normalized_prototype;
        cv::normalize(prototype, normalized_prototype, 0, 255, cv::NORM_MINMAX);
        
        cv::Mat uchar_prototype;
        normalized_prototype.convertTo(uchar_prototype, CV_8U);

        // Copy it to the correct position on the canvas
        int row = i / grid_cols;
        int col = i % grid_cols;
        uchar_prototype.copyTo(canvas(cv::Rect(col * mask_width, row * mask_height, mask_width, mask_height)));
    }

    cv::imshow("Raw Mask Prototypes", canvas);
}

void printTensorData(float* data, size_t size, const std::string& label) {
    std::cout << label << " values: ";
    for (size_t i = 0; i < size; ++i) {
        std::cout << data[i] << " ";
    }
    std::cout << std::endl;
}


int main() {
    try {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "YOLO_Segmentation");
        Ort::SessionOptions session_options;
        const wchar_t* model_path = L"D:/Comp_Vision/ObjUltralytics_py/yolo11m-seg.onnx"; 
        Ort::Session session(env, model_path, session_options);

        cv::VideoCapture cap(0);
        if (!cap.isOpened()) {
            std::cerr << "ERROR: Could not open webcam." << std::endl;
            return -1;
        }

        cv::Mat frame;
        while (true) {
            if (!cap.read(frame) || frame.empty()) {
                std::cerr << "ERROR: Could not read frame from webcam." << std::endl;
                break;
            }

            // --- Pre-processing & Inference ---
            cv::Mat blob;
            cv::dnn::blobFromImage(frame, blob, 1./255., cv::Size(640, 640), cv::Scalar(), true, false);
            std::vector<int64_t> input_shape = {1, 3, 640, 640};
            auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, (float*)blob.data, blob.total(), input_shape.data(), input_shape.size());
            const char* input_names[] = {"images"};
            const char* output_names[] = {"output0", "output1"}; 
            auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 2);

            // --- Post-processing ---
            // Get raw data pointers
            float* detection_data = output_tensors[0].GetTensorMutableData<float>();
            float* mask_data = output_tensors[1].GetTensorMutableData<float>();

            // Get tensor shapes
            std::vector<int64_t> detection_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
            std::vector<int64_t> mask_shape = output_tensors[1].GetTensorTypeAndShapeInfo().GetShape();

            
            // --- NEW: Visualize the raw mask prototypes ---
            visualize_prototypes(mask_data, mask_shape);

            const size_t num_classes = 80;
            const size_t mask_coeffs_size = detection_shape[1] - 4 - num_classes;
            const size_t num_proposals = detection_shape[2];
            cv::Mat detection_output(static_cast<int>(detection_shape[1]), static_cast<int>(num_proposals), CV_32F, detection_data);
            cv::transpose(detection_output, detection_output);
            
            std::vector<cv::Rect> boxes;
            std::vector<int> class_ids;
            std::vector<float> confidences;
            std::vector<cv::Mat> mask_coefficients;

            for (int i = 0; i < detection_output.rows; i++) {
                cv::Mat row = detection_output.row(i);
                cv::Mat classes_scores = row.colRange(4, 4 + static_cast<int>(num_classes));
                cv::Point class_id_point;
                double max_class_score;
                cv::minMaxLoc(classes_scores, 0, &max_class_score, 0, &class_id_point);
                if (max_class_score > 0.5) { // Use a slightly higher threshold to reduce noise
                    float cx = row.at<float>(0, 0);
                    float cy = row.at<float>(0, 1);
                    float w = row.at<float>(0, 2);
                    float h = row.at<float>(0, 3);
                    float x_factor = frame.cols / 640.0f;
                    float y_factor = frame.rows / 640.0f;
                    int left = static_cast<int>((cx - 0.5 * w) * x_factor);
                    int top = static_cast<int>((cy - 0.5 * h) * y_factor);
                    int width = static_cast<int>(w * x_factor);
                    int height = static_cast<int>(h * y_factor);
                    boxes.push_back(cv::Rect(left, top, width, height));
                    confidences.push_back((float)max_class_score);
                    class_ids.push_back(class_id_point.x);
                    mask_coefficients.push_back(row.colRange(4 + static_cast<int>(num_classes), 4 + static_cast<int>(num_classes) + static_cast<int>(mask_coeffs_size)));
                }
            }

            std::vector<int> indices;
            cv::dnn::NMSBoxes(boxes, confidences, 0.5f, 0.4f, indices);
            
            // --- NEW: Print raw coefficients for the first detected object ---
            bool first_detection_printed = false;
            if (!indices.empty()) {
                int first_idx = indices[0];
                cv::Mat first_coeffs = mask_coefficients[first_idx];
                
                std::cout << "--- Raw Data for First Detected Object (Frame) ---" << std::endl;
                std::cout << "Class: " << CLASS_NAMES[class_ids[first_idx]] << ", Confidence: " << confidences[first_idx] << std::endl;
                std::cout << "Mask Coefficients (recipe): ";
                for (int i = 0; i < first_coeffs.cols; ++i) {
                    std::cout << std::fixed << std::setprecision(4) << first_coeffs.at<float>(0, i) << " ";
                }
                std::cout << "\n----------------------------------------------------" << std::endl;
                first_detection_printed = true;
            }


            // We keep the main window to see what the current rendering shows
            cv::imshow("Original Feed", frame);

            if (cv::waitKey(1) == 27) { // 'ESC' key to exit
                break;
            }
        }
        cap.release();
        cv::destroyAllWindows();
    }
    catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}