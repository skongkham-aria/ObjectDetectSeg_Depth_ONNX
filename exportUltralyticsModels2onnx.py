from ultralytics import YOLO

# Load the YOLO model
# model = YOLO("yolo11l.pt")  # detection

# # Export the model to ONNX format
# export_path = model.export(format="onnx", opset=10)
# # Results saved to D:\Comp_Vision\ObjUltralytics_py
# # Predict:         yolo predict task=detect model=yolo11l.onnx imgsz=640  
# # Validate:        yolo val task=detect model=yolo11l.onnx imgsz=640 data=/ultralytics/ultralytics/cfg/datasets/coco.yaml  
# # Visualize:       https://netron.app
# # Model exported to yolo11l.onnx

model = YOLO("yolo11m-seg.pt")  # segmentation
# what is opset? Operation Set - a versioning system for operators in ONNX
export_path = model.export(format="onnx", opset=10)
# export_path = model.export(format="onnx")
print(f"ONNX Model exported to {export_path}")

# export model to tflite
## The ai_edge_litert package required for TFLite export is currently only available on Linux-based systems
# # use this in google colab
# export_path = model.export(format="tflite")
# print(f"TFLITE Model exported to {export_path}")