from cvu.detector.yolov5 import Yolov5

model = Yolov5(
    classes="menu",
    backend="onnx",
    weight="detect_models/best.onnx",
    device="cpu",
    input_shape=640,
)
