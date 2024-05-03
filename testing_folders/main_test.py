from api import yolo_model

model_path = 'yolov8_model/result/weights/last.pt'

result = yolo_model(model_full_path=model_path, image='https://m.media-amazon.com/images/I/61+r3+JstZL.jpg')

print(result)