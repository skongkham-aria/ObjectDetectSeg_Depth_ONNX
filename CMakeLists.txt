cmake_minimum_required(VERSION 3.10)
project(YoloWebcam)

# --- OpenCV ---
set(OpenCV_DIR "D:/opencv/build") 
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# --- ONNX Runtime ---
# Set this to the path where you extracted the ONNX Runtime release
set(ONNXRUNTIME_ROOTDIR "D:/onnxruntime-win-x64-1.10.0") 

find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
    HINTS ${ONNXRUNTIME_ROOTDIR}/include)

find_library(ONNXRUNTIME_LIBRARY
    NAMES onnxruntime
    HINTS ${ONNXRUNTIME_ROOTDIR}/lib)

include_directories(${ONNXRUNTIME_INCLUDE_DIR})

# # --- Executable ---
add_executable(yolo_webcam main.cpp)
# --- Linking ---
target_link_libraries(yolo_webcam ${OpenCV_LIBS} ${ONNXRUNTIME_LIBRARY})

# --- Executable ---
add_executable(yolo_segmentation yolo_Objseg.cpp)
# --- Linking ---
target_link_libraries(yolo_segmentation ${OpenCV_LIBS} ${ONNXRUNTIME_LIBRARY})

# # --- Executable ---
add_executable(seg_debug yolo_Objseg_Debug.cpp)
# --- Linking ---
target_link_libraries(seg_debug ${OpenCV_LIBS} ${ONNXRUNTIME_LIBRARY})

# # --- Executable ---
add_executable(webcam_live_depth webcam_live_depth.cpp)
# --- Linking ---
target_link_libraries(webcam_live_depth ${OpenCV_LIBS} ${ONNXRUNTIME_LIBRARY})

# --- Point Cloud Visualization Executable ---
add_executable(depth_pointcloud_viewer depth_pointcloud_viewer.cpp)
# --- Linking ---
target_link_libraries(depth_pointcloud_viewer ${OpenCV_LIBS} ${ONNXRUNTIME_LIBRARY})