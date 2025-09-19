#include <iostream>
#include <vector>
#include <numeric>
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

int main() {
    try {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "Yolo");
        Ort::SessionOptions session_options;
        const wchar_t* model_path = L"D:/Comp_Vision/ObjUltralytics_py/yolo11l.onnx";
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

            // --- NEW: CORRECT PRE-PROCESSING ---
            cv::Mat blob;
            // This single function handles resizing, normalization, BGR->RGB, and HWC->NCHW conversion.
            cv::dnn::blobFromImage(frame, blob, 1./255., cv::Size(640, 640), cv::Scalar(), true, false);

            // --- Create ONNX Runtime Tensor ---
            std::vector<int64_t> input_shape = {1, 3, 640, 640};
            auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, (float*)blob.data, blob.total(), input_shape.data(), input_shape.size());

            // --- Inference ---
            const char* input_names[] = {"images"};
            const char* output_names[] = {"output0"};
            auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
            
            // --- Post-processing ---
            Ort::TensorTypeAndShapeInfo output_shape_info = output_tensors[0].GetTensorTypeAndShapeInfo();
            std::vector<int64_t> output_shape = output_shape_info.GetShape();
            float* output_data = output_tensors[0].GetTensorMutableData<float>();
            cv::Mat output_buffer(output_shape[1], output_shape[2], CV_32F, output_data);
            cv::transpose(output_buffer, output_buffer); // Transpose to [8400, 84]

            std::vector<cv::Rect> boxes;
            std::vector<int> class_ids;
            std::vector<float> confidences;

            for (int i = 0; i < output_buffer.rows; i++) {
                cv::Mat classes_scores = output_buffer.row(i).colRange(4, 84);
                cv::Point class_id_point;
                double max_class_score;
                cv::minMaxLoc(classes_scores, 0, &max_class_score, 0, &class_id_point);

                if (max_class_score > 0.25) { // Use a reasonable threshold
                    float cx = output_buffer.at<float>(i, 0);
                    float cy = output_buffer.at<float>(i, 1);
                    float w = output_buffer.at<float>(i, 2);
                    float h = output_buffer.at<float>(i, 3);

                    // Scale bounding boxes back to the original frame size
                    float x_factor = frame.cols / 640.0;
                    float y_factor = frame.rows / 640.0;
                    
                    int left = int((cx - 0.5 * w) * x_factor);
                    int top = int((cy - 0.5 * h) * y_factor);
                    int width = int(w * x_factor);
                    int height = int(h * y_factor);

                    boxes.push_back(cv::Rect(left, top, width, height));
                    confidences.push_back((float)max_class_score);
                    class_ids.push_back(class_id_point.x);
                }
            }

            std::vector<int> indices;
            cv::dnn::NMSBoxes(boxes, confidences, 0.5f, 0.4f, indices); // Restore good NMS thresholds

            for (int idx : indices) {
                cv::rectangle(frame, boxes[idx], cv::Scalar(0, 255, 0), 2);
                std::string label = CLASS_NAMES[class_ids[idx]] + " " + cv::format("%.2f", confidences[idx]);
                cv::putText(frame, label, cv::Point(boxes[idx].x, boxes[idx].y - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
            }

            cv::imshow("YOLO Webcam", frame);
            if (cv::waitKey(1) == 27) {
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