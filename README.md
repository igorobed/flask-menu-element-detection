# flask-menu-element-detection

## Web-интерфейс: http://igorobed.pythonanywhere.com/

Задание: https://docs.google.com/document/d/1dVOdpIeV7DJfANbHMHOBI-0oln2Th5TtndimZW18YEg/edit

Собранный датасет: https://drive.google.com/drive/folders/1VXX9DOlfOyB7F9xYFmsneDD6NkDogzaA?usp=sharing

Обучающий набор данных состоит из 176 изображений, в валидационном и тестовом по 20 изображений

Датасет размечался с помощью CVAT(https://www.cvat.ai/). Данные разбиты на train/val/test. При обучении модель валидировалась на данных из val

Модель: YOLOv5s

Для обучения модели использовался следующий инструментарий: https://github.com/ultralytics/yolov5?ysclid=lf6pl35aub526802110

Также использовались такие инструменты, как ONNX, ONNXRuntime и библиотека CVU(https://github.com/BlueMirrors/cvu?ysclid=lf6pp2udxk110825426) для обработки инференса модели

Предобученная модель yolov5s дообучалась на собранном мною наборе данных с размером батча - 16 и количеством эпох - 35. Результаты обучения(веса) и полученные метрики доступны по следующей ссылке - https://drive.google.com/drive/folders/1B9Ms2Y435UhyzkJyF323h1Jg9jbYMhg8?usp=sharing

Оценка раюоты модели на валидационном наборе данных:

![image](https://user-images.githubusercontent.com/43452966/224967713-77ecf768-7c6a-40fc-b289-e60daeb45ebe.png)

Оценка работы модели на тестовом наборе данных:

![image](https://user-images.githubusercontent.com/43452966/224969274-bf0aff64-3958-4913-8207-ebdce44c8fde.png)

## Запуск

docker build . -t flask_detect:latest

docker run -p 5000:5000 flask_detect:latest

# Пример работы:

## Начало работы

![photo_5222210977150387696_y](https://user-images.githubusercontent.com/43452966/224692379-f375e0af-898b-4b2a-9bb2-722efadda095.jpg)

## Выбранный скриншот

![photo_5222210977150387698_y](https://user-images.githubusercontent.com/43452966/224692673-2332b61e-ae5f-40db-a19d-3ac316947038.jpg)

## Загрузка выбранного изображения

![photo_5222210977150387699_y](https://user-images.githubusercontent.com/43452966/224692885-b18551b0-33a8-406a-ab5e-da19abc71cea.jpg)

## После нажатия кнопки распознать

![photo_5222210977150387697_y (1)](https://user-images.githubusercontent.com/43452966/224693040-8f052282-5fc4-46d0-9cbc-6564e3480eb0.jpg)
