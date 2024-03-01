from fastapi import UploadFile
from face_recognition_utils import find_similar_faces
import cv2
import numpy as np
import glob
import os
import face_recognition

def get_image_name(image_path):
    path_components = image_path.split("/")
    image_name_with_extension = path_components[-1]
    image_name = os.path.splitext(image_name_with_extension)[0]
    return image_name

class FaceAPI:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.known_face_encodings = []
        self.known_face_paths = []

    def load_known_faces(self):
        img_paths = glob.glob(os.path.join(self.folder_path, "*.*"))

        for img_path in img_paths:
            img = cv2.imread(img_path)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img_encoding = face_recognition.face_encodings(rgb_img)[0]
            self.known_face_encodings.append(img_encoding)
            self.known_face_paths.append(img_path)

    async def find_similar_faces_api(self, photo: UploadFile):
        content = await photo.read()
        nparr = np.frombuffer(content, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        photo_encoding = face_recognition.face_encodings(rgb_img)[0]

        similar_faces = find_similar_faces(photo_encoding, self.known_face_encodings, self.known_face_paths)

        # Extracting the name of the matched faces
        names = [get_image_name(path) for path in similar_faces]

        return {"similar_faces": similar_faces, "Name": names}
