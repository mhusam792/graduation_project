from fastapi import HTTPException
from PIL import Image
from io import BytesIO
import os
import requests
from DeepImageSearch import Load_Data, Search_Setup
from typing import List


def read_image_from_url(url: str) -> Image.Image:
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check for errors in the HTTP response
        image = Image.open(BytesIO(response.content))
        return image

    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error fetching image from URL: {str(e)}")
    
# Function to extract base name
def extract_basename(file_path):
    return os.path.basename(file_path)

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

    def get_similar_images_path(self, image_path, num_images=2):
        if self.st is None:
            raise ValueError("Search engine is not initialized. Call create_search_engine first.")

        # Assuming image_path is a file path, convert it to the actual image object
        img_obj = read_image_from_url(url=image_path)

        # Convert the image to RGB mode
        img_obj = img_obj.convert('RGB')
        
        # Save the image to a temporary file and pass the file path
        temp_image_path = "temp_image.jpg"
        img_obj.save(temp_image_path)

        def extract_filename(file_path):
            # Use os.path.basename to get the filename from the file path
            filename = os.path.basename(file_path)
            return filename

        similar_image_path = self.st.get_similar_images(image_path=temp_image_path, number_of_images=num_images)
        # Convert numpy.int64 to standard Python int for serialization
        similar_image_path = {extract_filename(str(value)) for key, value in similar_image_path.items()}

        # Remove the temporary image file
        os.remove(temp_image_path)

        return similar_image_path

    def add_images_to_index(self, image_urls: List[str]):
        if self.st is None:
            raise HTTPException(status_code=400, detail="Search engine not initialized.")

        image_paths = []
        try:
            for url in image_urls:
                image = read_image_from_url(url)
                filename = url.split("/")[-1]  # Extract filename from URL
                path = os.path.join("folders", "search_by_image", filename)
                image.save(path)
                image_paths.append(path)
        except Exception as e:
            return {"error": f"Error downloading or saving images: {str(e)}"}

        try:
            result = self.st.add_images_to_index(image_paths)
        except Exception as e:
            return {"error": f"Error updating index: {str(e)}"}

        return {"message": "Images added to index successfully."}
