from PIL import Image
import requests
from io import BytesIO
from fastapi import HTTPException
from main_app.image_predictor import ImagePredictor


model_path = 'yolov8_model/result/weights/last.pt'
image_predictor = ImagePredictor(model_full_path=model_path)

def read_image_from_url(url: str) -> Image.Image:
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check for errors in the HTTP response

        image = Image.open(BytesIO(response.content))
        return image

    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error fetching image from URL: {str(e)}")
    
def get_objects_and_type(result):
    objects_and_type = {}

    if 'objects' in result:
        objects_and_type['objects'] = list(result['objects'].keys())

    if 'type' in result:
        objects_and_type['type'] = result['type']

    return objects_and_type

# Categories
def categorize_items(item_dict):
    type_list = []

    if len(item_dict) == 0:
        return type_list

    categories = dict(
        bags = ["Backpack", "Briefcase", "Handbag", "Suitcase"],
        electronics = ["Computer keyboard", "Computer monitor", "Computer mouse", "Tablet computer", "Calculator", "Camera", "Tripod", "Laptop", "Mobile phone", "Headphones", "Flashlight"],
        animals = ["Cat", "Dog"],
        accessories = ["Fashion accessory", "Earrings", "Necklace"],
        hats = ["Hat", "Fedora", "helmet", "Bicycle helmet", "Sun hat", "Swim cap"],
        glasses = ["Glasses", "Sunglasses", "Binoculars"],
        toys = ["Doll", "Football", "Ball", "Volleyball (Ball)"],
        personal_purposes = ["Watch", "wallet", "Bottle", "Mug", "Pencil case", "Book", "Umbrella", "Envelope"],
        clothes = ["Tie", "Belt"]
    )

    type_list = []

    for item, _ in item_dict.items():
        found = False
        for category, items in categories.items():
            if item in items:
                type_list.append(category)
                found = True
                break
        if not found:
            type_list.append("others")

    return type_list


def yolo_model(model_full_path, image: str):
    # try:
    prediction_result = image_predictor.model_result(
        model_path=model_full_path,
        img_path=read_image_from_url(image)
    )
    
    cls = image_predictor.return_cls(
        model_result=prediction_result[0],
        trained_model=prediction_result[1]
    )
    dict_cls = image_predictor.count_cls(cls)

    if len(dict_cls) == 0:
        print("Sorry, We can't figure out what type of item it is?! \n Please choose from this list the type of item in the image.")
        founded_objects = dict_cls
    else:
        founded_objects = dict_cls

    type_list = categorize_items(founded_objects)
    
    data_from_image = {"objects": founded_objects, "type": type_list}

    data_from_image = get_objects_and_type(data_from_image)
    
    return data_from_image

