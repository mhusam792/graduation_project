import os
from ultralytics import YOLO
import torch
from shutil import copy2, rmtree

class ImagePredictor:
    def __init__(self, model_full_path):
        self.model_full_path = model_full_path

    def model_result(self, model_path, img_path):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Using Device: ', device)
        model = YOLO(model_path)
        results = model.predict(img_path, device=device,
                                show_conf=False, conf=.45, project='folders/runs/detect/prediction')
        return results, model

    def return_cls(self, model_result, trained_model):
        clist = model_result[0].boxes.cls
        cls = [trained_model.names[int(cno)] for cno in clist]
        return cls

    def count_cls(self, input_list):
        label_counts = {}
        for label in input_list:
            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts[label] = 1
        return label_counts

    def copy_images_and_remove_folder(self, source_folder, destination_folder):
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)
        files = os.listdir(source_folder)
        for file_name in files:
            source_path = os.path.join(source_folder, file_name)
            destination_path = os.path.join(destination_folder, file_name)
            copy2(source_path, destination_path)
        rmtree(source_folder)
