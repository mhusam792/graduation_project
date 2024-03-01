import cv2
import numpy as np
import os
from rembg import remove
from PIL import Image, ImageEnhance, ImageOps
from scipy.ndimage import rotate as rotate_image

parent_folder_path = "folders/image_enhancment/output"

os.makedirs(parent_folder_path, exist_ok=True)

subfolder_names = ["DeNoisesImage", "brightness_images", "cards","light_images", 
                   "median_filter", "resized_image","rotated_image","sharpe_images"]

for subfolder_name in subfolder_names:
    subfolder_path = os.path.join(parent_folder_path, subfolder_name)
    os.makedirs(subfolder_path, exist_ok=True)

def apply_median_filter(image_path:str, kernel_size: int = 3) :
    """
  A Function that apply median filter on image to use it on Many purposes, such as eliminating noises.

  Parameters:
  - A image_path {[string]}.
  - A Kernel size {[int]}. Kernel size should be an odd number


    """
    image = cv2.imread(image_path)
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size should be an odd number")

    padded_image = cv2.copyMakeBorder(image, kernel_size // 2, kernel_size // 2,
                                       kernel_size // 2, kernel_size // 2, cv2.BORDER_CONSTANT)


    output_image = np.zeros_like(image)

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):

            roi = padded_image[y:y+kernel_size, x:x+kernel_size]

            output_image[y, x] = np.median(roi)

    base_name=os.path.basename(image_path)
    (file_Name,ext)=os.path.splitext(base_name)
    cv2.imwrite(f"folders/image_enhancment/output/median_filter/{file_Name}.png",output_image)

def remove_img_bg(input_img_path:str,output_path:str):
  """
        remove image back Ground and corp the fore ground of the image

        Arguments:
            input_img_path {string} -- image Path
            output_path {string} -- where is the new image will be saved
  """
  input=Image.open(input_img_path)
  output=remove(input)
  output.save(output_path)

def cut_black_regions(image_path:str) -> np.array:
    """
   A Function that remove black parts from images that have a black background.

   Parameters:
   - A image_path {string}.

   Returns:
   - it return a new cropped image with removed black parts from it.

    """

    img = Image.open(image_path)

    width, height = img.size

    # Find the bounding box of non-black regions
    bbox = img.getbbox()

    # Crop the image to the bounding box
    cropped_img = img.crop(bbox)
    im2arr = np.array(cropped_img)

    return im2arr

def get_card(img_path):
  """
  A function named 'get_card' designed to process images by removing black parts from those with a black background.

  Parameters:
   - img_path {string}: The path to the input image that requires processing.

  Note: This function relies on two additional functions, 'remove_img_bg' and 'cut_black_regions,' which should be defined elsewhere in the codebase
  for the proper execution of the 'get_card' function.
  """
  remove_img_bg(img_path,"folders/image_enhancment/output.png")
  processed_image=cut_black_regions("/content/output.png")
  processed_image=cv2.cvtColor(processed_image,cv2.COLOR_RGB2BGR)
  processed_image = cv2.resize(processed_image, (700, 480))
  os.remove("folders/image_enhancment/output.png")
  base_name=os.path.basename(img_path)
  (file_Name,ext)=os.path.splitext(base_name)
  cv2.imwrite(f"folders/image_enhancment/output/cards/{file_Name}.png",processed_image)

def resize(img_path:str, width: int =None, height: int =None):
    """
        resize the image.

        Arguments:
            img_path {string} -- image Path
            width {[int]} -- the new width of the resized image
            height {[int]} -- the new height of the resized image

    """
    image=cv2.imread(img_path)
    if width is None and height is None:
        return image

    if width is None:
        r = height / image.shape[0]
        width = int(r * image.shape[1])
    elif height is None:
        r = width / image.shape[1]
        height = int(r * image.shape[0])

    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    base_name=os.path.basename(img_path)
    (file_Name,ext)=os.path.splitext(base_name)
    cv2.imwrite(f"folders/image_enhancment/output/resized_image/{file_Name}.png",resized)

def rotate(img_path:str,scale:float,direction:str,angle_direction:str,angle:int =0):
    """
        Rotate the image in anti-clockwise in a given angle or using specfic direction

        Arguments:
            img_path {string} -- image Path
            scale {[float]} -- the scale of the rotated image
            direction {string} -- the direction you want to rotate the image on it
            angle_direction {string} -- the way you want to rotate the ihe image (direction,angle)
            angle {[int]} -- rotation angle (default: 0)
    """
    image = cv2.imread(img_path)
    D=0
    if direction.lower()=="left":
      D=90
    if direction.lower()=="right":
      D=-90
    if direction.lower()=="flip":
      D=180

    if angle_direction.lower()=="direction":
      rotated = rotate_image(image,D)
      base_name=os.path.basename(img_path)
      (file_Name,ext)=os.path.splitext(base_name)
      cv2.imwrite(f"folders/image_enhancment/output/rotated_image/{file_Name}.png",rotated)


    else:
      rotated = rotate_image(image,angle)
      base_name=os.path.basename(img_path)
      (file_Name,ext)=os.path.splitext(base_name)
      cv2.imwrite(f"folders/image_enhancment/output/rotated_image/{file_Name}.png",rotated)

def avg_blur(image_path:str):
    """
       A Function that apply avg blur filter on image to use it on Many purposes, such as eliminating noises or blur the image.

      Parameters:
      - A image_path {[string]}.
    """
    image = cv2.imread(image_path)
    #! The Kernel Side Length Must be Odd
    blured = cv2.blur(image, (5, 5))
    base_name=os.path.basename(image_path)
    (file_Name,ext)=os.path.splitext(base_name)
    cv2.imwrite(f"folders/image_enhancment/output/DeNoisesImage/avg_blur/{file_Name}.png",blured)

def gauss_blur(image_path:str):
    """
       A Function that apply gauss blur filter on image to use it on Many purposes, such as eliminating noises or blur the image.

      Parameters:
      - A image_path {[string]}.

    """
    image = cv2.imread(image_path)
    # By setting the third parameter to 0, we are instructing OpenCV
    # to automatically compute the weights based on our kernel size
    g_blur=cv2.GaussianBlur(image, (5, 5), 0)

    base_name=os.path.basename(image_path)
    (file_Name,ext)=os.path.splitext(base_name)
    cv2.imwrite(f"folders/image_enhancment/output/DeNoisesImage/gauss_blur/{file_Name}.png",g_blur)

def bilateral(image_path:str):
    """
  A Function that apply bilateral filter on image to use it on Many purposes, such as eliminating noises or sharp the image edges.

  Parameters:
  - A image_path {[string]}.


    """
    image = cv2.imread(image_path)
    filtered = cv2.bilateralFilter(image, 7, 31, 31)

    base_name=os.path.basename(image_path)
    (file_Name,ext)=os.path.splitext(base_name)
    cv2.imwrite(f"folders/image_enhancment/output/DeNoisesImage/bilateral/{file_Name}.png",filtered)

def remove_noise(img_path,method):
  """
    A function named 'remove_noise' designed to apply various noise reduction methods to an image based on the specified 'method' parameter.

    Parameters:
      - img_path {string}: The path to the input image that requires noise reduction.
      - method {string}: The chosen noise reduction method. Supported methods include:
      - 'avg_blur': Applies average blur to the image.
      - 'gauss_blur': Applies Gaussian blur to the image.
      - 'bilateral': Applies bilateral filter to the image.

    Note: It is assumed that the 'avg_blur,' 'gauss_blur,' and 'bilateral' functions are implemented to perform the corresponding noise reduction techniques.
    Additionally, the 'remove_noise' function relies on these helper functions, and they need to be defined for the proper execution of the noise reduction process.
  """
  if method=='avg_blur':
    avg_blur(img_path)
  if method=='gauss_blur':
    gauss_blur(img_path)
  if method=='bilateral':
    bilateral(img_path)

def bright_an_img(img_path:str):
  """
  A Function that enhance image contrast by multiply it with  enhancement factor that can be sepcifecd Manually.

  Parameters:
  - A img_path {[string]}.

  """
  original_image = Image.open(img_path)
  enhancer = ImageEnhance.Brightness(original_image)
  enhanced_image = enhancer.enhance(1.3)  # You can adjust the enhancement factor
  enhanced_array = np.array(enhanced_image)
  enhanced_array=cv2.cvtColor(enhanced_array,cv2.COLOR_BGR2RGB)
  base_name=os.path.basename(img_path)
  (file_Name,ext)=os.path.splitext(base_name)
  cv2.imwrite(f"folders/image_enhancment/output/brightness_images/{file_Name}.png",enhanced_array)

def sharpe_image(img_path):
  img = Image.open(img_path)
  enhancer = ImageEnhance.Sharpness(img)
  img = enhancer.enhance(8.0)
  enhanced_array = np.array(img)
  enhanced_array=cv2.cvtColor(enhanced_array,cv2.COLOR_BGR2RGB)
  base_name=os.path.basename(img_path)
  (file_Name,ext)=os.path.splitext(base_name)
  cv2.imwrite(f"folders/image_enhancment/output/sharpe_images/{file_Name}.png",enhanced_array)

def light_image(img_path:str):
  """
    Enhance the brightness and contrast of an image using autocontrast and save the result.

    Parameters:
    - img_path (str): Path to the input image file.

  """
  original_image = Image.open(img_path)
  enhanced_image=np.array(ImageOps.autocontrast(original_image))
  enhanced_image=cv2.cvtColor(enhanced_image,cv2.COLOR_RGB2BGR)
  base_name=os.path.basename(img_path)
  (file_Name,ext)=os.path.splitext(base_name)
  cv2.imwrite(f"folders/image_enhancment/output/light_images/{file_Name}.png",enhanced_image)
