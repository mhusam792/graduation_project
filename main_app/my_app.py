from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from object_finder import ObjectFinder
from image_predictor import ImagePredictor
from search_similar_images import SearchEngine
from id import ImageProcessor
from tamween import IDExtractor
from face_api import FaceAPI
from license import get_final_data
from egy_plate_recognition import get_Letters_and_nums
from image_enhacment import *
from read_any_image import data_corection_, get_data_
from PIL import Image
import requests
from io import BytesIO
import os
from typing import List
from bs4 import BeautifulSoup
import csv
import numpy as np
from urllib.parse import urlparse

# from text_to_image_genarate import create_img


def download_image(url, folder_path):
    try:
        # Create the folder if it doesn't exist
        os.makedirs(folder_path, exist_ok=True)

        # Extract filename from the URL
        filename = os.path.basename(urlparse(url).path)

        # Construct the full path to save the image
        file_path = os.path.join(folder_path, filename)

        # Make a request to the URL
        response = requests.get(url)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Save the image
            with open(file_path, 'wb') as file:
                file.write(response.content)

            print(f"Image downloaded successfully and saved at: {file_path}")
        else:
            print(f"Failed to download the image. Status code: {response.status_code}")

    except Exception as e:
        print(f"An error occurred: {e}")


def resize_image(input_path, output_path, new_size):
    try:
        # Read the image
        image = cv2.imread(input_path)

        # Check if the image is not empty
        if image is not None:
            # Resize the image
            resized_image = cv2.resize(image, new_size)

            # Save the resized image
            cv2.imwrite(output_path, resized_image)

            print(f"Image resized successfully and saved at: {output_path}")
        else:
            print("Error: Input image is empty.")

    except Exception as e:
        print(f"An error occurred: {e}")

def read_image_from_url(url: str) -> Image.Image:
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check for errors in the HTTP response

        image = Image.open(BytesIO(response.content))
        return image

    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error fetching image from URL: {str(e)}")
    

def extract_image_info(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table')

    image_info = []
    if table:
        base_url = url.rstrip('/')  # Remove trailing slash from base URL
        rows = table.find_all('tr')
        for row in rows[1:]:  # Skip the header row
            columns = row.find_all('td')
            image_filename = columns[0].text.strip()
            full_image_url = f"{base_url}/{image_filename}"
            image_info.append({'filename': image_filename, 'url': full_image_url})

    return image_info

def download_images(image_info, download_folder):
    os.makedirs(download_folder, exist_ok=True)

    for info in image_info:
        response = requests.get(info['url'])
        image_filename = os.path.join(download_folder, info['filename'])

        with open(image_filename, 'wb') as image_file:
            image_file.write(response.content)

def save_to_csv(image_info, csv_filename):
    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Image Filename', 'Image URL'])  # Write header
        for info in image_info:
            csv_writer.writerow([info['filename'], info['url']])

def update_images_folder(image_info, download_folder):
    existing_images = set(os.listdir(download_folder))

    for info in image_info:
        image_filename = info['filename']
        image_path = os.path.join(download_folder, image_filename)

        if image_filename not in existing_images:
            # Download the image if it doesn't exist in the folder
            response = requests.get(info['url'])
            with open(image_path, 'wb') as image_file:
                image_file.write(response.content)
        else:
            # Remove the image from the folder if it's not in image_info
            existing_images.remove(image_filename)

    # Delete images in the folder that are not present in image_info
    for obsolete_image in existing_images:
        obsolete_image_path = os.path.join(download_folder, obsolete_image)
        os.remove(obsolete_image_path)


class FindObjectsRequest(BaseModel):
    find_object: str
    founded_objects: list

class FindObjectsResponse(BaseModel):
    top_matches: list
   


class ImageRequest(BaseModel):
    image_path: str

class SimilarImagesResponse(BaseModel):
    similar_images: dict



class MyApp(FastAPI):
    def __init__(self, model_full_path, upload_folder, source_folder_path, destination_folder_path, host, port):
        super().__init__()
        self.model_full_path = model_full_path
        self.image_predictor = ImagePredictor(model_full_path)
        self.upload_folder = upload_folder
        self.source_folder_path = source_folder_path
        self.destination_folder_path = destination_folder_path
        self.object_finder = ObjectFinder()
        self.search_engine_instance = SearchEngine()
        self.processor = ImageProcessor()

        self.host = host
        self.port = port

    def setup_routes(self):
        # chang dir
        # @self.post("/predict")
        @self.get("/predict", tags=["Model"])
        async def predict_image(file: str):
            try:
                # file_path = os.path.join("/tmp", file.filename)  # Using a temporary folder instead
                # with open(file_path, "wb") as image_file:
                #     image_file.write(file.file.read())

                prediction_result = self.image_predictor.model_result(
                    model_path=self.model_full_path,
                    img_path=read_image_from_url(file)
                )
                
                cls = self.image_predictor.return_cls(
                    model_result=prediction_result[0],
                    trained_model=prediction_result[1]
                )
                dict_cls = self.image_predictor.count_cls(cls)

                if len(dict_cls) == 0:
                    founded_objects = "Sorry, We can't figure out what type of item it is?! \n Please choose from this list the type of item in the image."
                else:
                    founded_objects = dict_cls

                if not os.path.exists(self.source_folder_path):
                    os.makedirs(self.source_folder_path)

                self.image_predictor.copy_images_and_remove_folder(
                    self.source_folder_path, self.destination_folder_path
                )

                path_all_images = 'folders/runs/detect/prediction/' ###################
                image_full_path = os.path.join(path_all_images, file)
                file_name_without_extension = os.path.splitext(os.path.basename(image_full_path))[0]
                
                data_from_image = {"objects": founded_objects}

                return JSONResponse(content=data_from_image)

            except Exception as e:
                return JSONResponse(content={"error": str(e)}, status_code=500)
        
        # compare text description
        # change dir
        @self.post("/find_objects", response_model=FindObjectsResponse, tags=['Comparing Text'])
        async def find_objects(request: FindObjectsRequest):
            print(request.dict())  # Add this line for debugging
            try:
                if not request.find_object:
                    raise HTTPException(status_code=400, detail="Please provide a description")

                # en_find_object, en_founded_objects = self.object_finder.translate_to_english(
                #     find_object=request.find_object, founded_objects=request.founded_objects
                # )
                top_matches = self.object_finder.compare_objects_similarity(request.find_object, request.founded_objects)

                response = {
                    "top_matches": [{"description": obj, "similarity_score": score} for obj, score in top_matches]
                }

                return response
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        # Search Engine
        @self.post("/create_search_engine", tags=["Search Engine"])
        def create_search_engine_api(folder_path: str="https://wdw888lb-7075.uks1.devtunnels.ms/resources/"):
            url = folder_path
            image_info = extract_image_info(url)

            # # Save the image info to a CSV file
            # csv_filename = "folders/image_info.csv"
            # save_to_csv(image_info, csv_filename)

            # Update the images in the "search_engine_images" folder
            download_folder = "folders/search_by_image"
            update_images_folder(image_info, download_folder)

            self.search_engine_instance.create_search_engine('folders/search_by_image')
            return JSONResponse(content={"message": "Search engine created successfully."})

        @self.get("/get_similar_images", tags=["Search Engine"])
        def get_similar_images_api(img: str):
            try:
                img_url = img  # Assuming img parameter is the URL of the image
                similar_image_path = self.search_engine_instance.get_similar_images_path(img_url)
                return {"similar_images": similar_image_path}
            except HTTPException as e:
                return {"error": str(e)}

        @self.post("/add_images_to_index", tags=["Search Engine"])
        def add_images_to_index_api(image_urls: List[str]):
            try:
                # Download images and add them to the search engine index
                result = self.search_engine_instance.add_images_to_index(image_urls)
                return JSONResponse(content=result)
            except HTTPException as e:
                return JSONResponse(content={"error": str(e)})

        # Similar Faces
        @self.get("/find_similar_faces", tags=["Find Similar Faces"])
        async def find_similar_faces_api(photo: str, folder_path: str = "folders/face_identifier/known_face_images"): ############# change path
            face_api = FaceAPI(folder_path)
            face_api.load_known_faces()
            result = await face_api.find_similar_faces_api(photo)
            return result


        # Card
        @self.get("/id", tags=["Card"])
        async def process_image(photo: str):
            try:
                # # Save the uploaded image temporarily
                # with open("temp_image.jpg", "wb") as temp_image:
                #     temp_image.write(photo.file.read())

                # Process the image using the ImageProcessor
                url = photo
                folder_path = "folders/ids"
                download_image(url, folder_path)
                
                # Extract filename from the URL
                filename = os.path.basename(urlparse(url).path)
                print(filename)
                # # Construct the full path to save the image
                # file_path = os.path.join(folder_path, filename)
                # print(file_path)
                fully_path = "/home/hossam/python_projects/final_app_2/final_app/folders/ids/"
                img = fully_path + filename
                # input_path = "path/to/your/input/image.jpg"
                # output_path = "path/to/your/output/resized_image.jpg"
                new_size = (700, 480)  # Replace width and height with your desired dimensions
                print(img)
                resize_image(img, img, new_size)
                result = self.processor.get_final_data(img)

                return JSONResponse(content=result, status_code=200)

            except Exception as e:
                # Handle exceptions
                return HTTPException(detail=str(e), status_code=500)

        @self.get("/tamween_card", tags=["Card"])
        async def extract_id(image_path: str):
            # # Process the image using the ImageProcessor
            # url = image_path
            # folder_path = "folders/tamween_images"
            # download_image(url, folder_path)
            
            # # Extract filename from the URL
            # filename = os.path.basename(urlparse(url).path)
            # print(filename)
            # # # Construct the full path to save the image
            # # file_path = os.path.join(folder_path, filename)
            # # print(file_path)
            # fully_path = "/home/hossam/python_projects/final_app_2/final_app/folders/tamween_images/"
            # img = fully_path + filename
            # # input_path = "path/to/your/input/image.jpg"
            # # output_path = "path/to/your/output/resized_image.jpg"
            # new_size = (700, 480)  # Replace width and height with your desired dimensions
            # print(img)
            # resize_image(img, img, new_size)

            id_extractor = IDExtractor(image_path)
            final_data = id_extractor.get_final_data()
            return final_data
        
        @self.get("/license", tags=["Card"])
        async def process_image(file: str):
            # file_path = f"/tmp/{file.filename}"
            # with open(file_path, "wb") as buffer:
            #     buffer.write(file.file.read())

            try:
                url = file
                folder_path = "folders/license"
                download_image(url, folder_path)
                
                # Extract filename from the URL
                filename = os.path.basename(urlparse(url).path)
                print(filename)
                # # Construct the full path to save the image
                # file_path = os.path.join(folder_path, filename)
                # print(file_path)
                fully_path = "/home/hossam/python_projects/final_app_2/final_app/folders/license/"
                img = fully_path + filename
                # input_path = "path/to/your/input/image.jpg"
                # output_path = "path/to/your/output/resized_image.jpg"
                new_size = (700, 480)  # Replace width and height with your desired dimensions
                print(img)
                resize_image(img, img, new_size)
                data = get_final_data(img)
                return JSONResponse(content=data)
            except Exception as e:
                return JSONResponse(content={"error": str(e)}, status_code=500)
            
        @self.get("/get_plate_num", tags=["Model"])
        async def get_plate(file_path):
            url = file_path
            folder_path = "folders/car_plate"
            download_image(url, folder_path)
            
            # Extract filename from the URL
            filename = os.path.basename(urlparse(url).path)
            print(filename)
            # # Construct the full path to save the image
            # file_path = os.path.join(folder_path, filename)
            # print(file_path)
            fully_path = "/home/hossam/python_projects/final_app_2/final_app/folders/car_plate/"
            img = fully_path + filename
            # input_path = "path/to/your/input/image.jpg"
            # output_path = "path/to/your/output/resized_image.jpg"
            new_size = (700, 480)  # Replace width and height with your desired dimensions
            print(img)
            resize_image(img, img, new_size)
            return get_Letters_and_nums(img)
        

        # Image enhancement
        @self.get("/apply_median_filter",tags=["Image optimization"])
        async def median_filter(file_path):
            base_name=os.path.basename(file_path)
            (file_Name,ext)=os.path.splitext(base_name)
            apply_median_filter(file_path)
            out_put_path=f"folders/image_enhancment/output/median_filter/{file_Name}.png"
            return {'output_img_path':out_put_path}

        @self.get("/get_card",tags=["Card"])
        async def get_Card(file_path):
            base_name=os.path.basename(file_path)
            (file_Name,ext)=os.path.splitext(base_name)
            get_card(file_path)
            out_put_path=f"folders/image_enhancment/output/cards/{file_Name}.png"
            return {'output_img_path':out_put_path}

        @self.get("/resize",tags=["Image optimization"])
        async def Resize(file_path,width:int,height:int):
            base_name=os.path.basename(file_path)
            (file_Name,ext)=os.path.splitext(base_name)
            resize(file_path,width,height)
            out_put_path=f"folders/image_enhancment/output/resized_image/{file_Name}.png"
            return {'output_img_path':out_put_path}

        @self.get("/rotate",tags=["Image optimization"])
        async def Rotate(file_path,scale:float,direction:str,angle_direction:str,angle:int):
            base_name=os.path.basename(file_path)
            (file_Name,ext)=os.path.splitext(base_name)
            rotate(file_path,scale,direction,angle_direction,angle)
            out_put_path=f"folders/image_enhancment/output/rotated_image/{file_Name}.png"
            return {'output_img_path':out_put_path}

        @self.get("/remove_noise",tags=["Image optimization"])
        async def remove_Noise(file_path,method):
            base_name=os.path.basename(file_path)
            (file_Name,ext)=os.path.splitext(base_name)
            remove_noise(file_path,method)#avg_blur gauss_blur bilateral
            out_put_path=f"folders/image_enhancment/output/DeNoisesImage/{method}/{file_Name}.png"
            return {'output_img_path':out_put_path}

        @self.get("/bright_an_img",tags=["Image optimization"])
        async def bright_An_Img(file_path):
            base_name=os.path.basename(file_path)
            (file_Name,ext)=os.path.splitext(base_name)
            bright_an_img(file_path)
            out_put_path=f"folders/image_enhancment/outputbrightness_images/{file_Name}.png"
            return {'output_img_path':out_put_path}

        @self.get("/light_image",tags=["Image optimization"])
        async def light_Image(file_path):
            base_name=os.path.basename(file_path)
            (file_Name,ext)=os.path.splitext(base_name)
            light_image(file_path)
            out_put_path=f"folders/image_enhancment/output/light_images/{file_Name}.png"
            return {'output_img_path':out_put_path}

        @self.get("/sharpe_image",tags=["Image optimization"])
        async def sharpe_Image(file_path):
            base_name=os.path.basename(file_path)
            (file_Name,ext)=os.path.splitext(base_name)
            sharpe_image(file_path)
            out_put_path=f"folders/image_enhancment/output/sharpe_images/{file_Name}.png"
            return {'output_img_path':out_put_path}

        @self.get("/blur_image_card",tags=["Card"])
        async def blur_image(file_path,type_of_card,card):
            if card =="yes":
                blur_card2(file_path,type_of_card)
                base_name=os.path.basename(file_path)
                (file_Name,ext)=os.path.splitext(base_name)
                out_put_path=f"folders/image_enhancment/output/blured_image/{file_Name}_blured_image.png"
            elif card =="no":
                blur_image(file_path)
                base_name=os.path.basename(file_path)
                (file_Name,ext)=os.path.splitext(base_name)
                out_put_path=f"folders/image_enhancment/output/blured_image/{file_Name}_full_blured_image.png"
            return {'output_img_path':out_put_path}
        
        @self.get("/get_data",tags=["Read Data"])
        async def get_data(file_path):
            data=data_corection_(get_data_(file_path))
            return {"data":data}

        # @self.get("/create_img")
        # async def image_genarate(method,text,img_number=6):
        #     path=''
        #     keyword=create_img(method,text,int(img_number))
        #     if method== 'google':
        #         path=f'/content/simple_images/{keyword}'
        #     elif method== 'AI':
        #         path=f"/content/genarated_images/{keyword}.png"
        #     return {'output_img_path':path}
