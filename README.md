# flask-menu-element-detection

Web-интерфейс: http://igorobed.pythonanywhere.com/

Собранный датасет: https://drive.google.com/drive/folders/1VXX9DOlfOyB7F9xYFmsneDD6NkDogzaA?usp=sharing

Модель: YOLOv5s

Для обучения модели использовался следующий инструментарий: https://github.com/ultralytics/yolov5?ysclid=lf6pl35aub526802110

Также использовались такие инструменты, как ONNX, ONNXRuntime и библиотека CVU(https://github.com/BlueMirrors/cvu?ysclid=lf6pp2udxk110825426) для обработки инференса модели

Предобученная модель yolov5s дообучалась на собранном мною наборе данных с размером батча - 16 и количеством эпох - 35
