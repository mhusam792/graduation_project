from DeepImageSearch import Load_Data, Search_Setup
from fastapi import HTTPException
from PIL import Image
import os

class SearchEngine:
    def __init__(self):
        self.st = None
        self.metadata = None

    def create_search_engine(self, folder_path: str):
        image_list = Load_Data().from_folder([folder_path])
        self.st = Search_Setup(image_list=image_list, model_name='vgg19', 
                               pretrained=True, image_count=len(image_list))
        self.st.run_index()
        self.metadata = self.st.get_image_metadata_file()

    def get_similar_images_path(self, image_path, num_images=5):
        if self.st is None:
            raise ValueError("Search engine is not initialized. Call create_search_engine first.")

        similar_image_path = self.st.get_similar_images(image_path=image_path, number_of_images=num_images)
        # Convert numpy.int64 to standard Python int for serialization
        similar_image_path = {str(key): str(value) for key, value in similar_image_path.items()}
        return similar_image_path

    def add_images_to_index(self, image_paths: list):
        if self.st is None:
            raise HTTPException(status_code=400, detail="Search engine not initialized.")
        self.st.add_images_to_index(image_paths)
        return {"message": "Images added to index successfully."}

