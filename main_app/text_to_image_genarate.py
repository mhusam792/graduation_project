from imaginairy.api import imagine
from imaginairy.schema import ImaginePrompt
from imaginairy.utils.log_utils import configure_logging
import cv2
from googletrans import Translator
from simple_image_download import simple_image_download as simp
# import matplotlib.pyplot as plt
import glob
import os
import numpy as np

configure_logging()
list(imagine(ImaginePrompt("", size=64, steps=3, negative_prompt="", seed=1)))

def trans(txt:str)->str:
  """
         A Function that Translate string using google Translator API.

         Parameters:
         - A string txt.

         Returns:
         - it Return string text after translate it by google Translator.
  """
  translator = Translator()
  translation_result = translator.translate(txt, dest='en')
  return translation_result.text


parent_folder_path = "folders/genarated_images" ### chang dir

os.makedirs(parent_folder_path, exist_ok=True)

def create_img(method:str,text:str,img_number: int = 6):
  """
         A Function that create an image using artificial intelligence or by searching through the Google search engine.

         Parameters:
         - A string method will be used to create the image AI OR Google search engine.
         - A string descrption of the image will be created.
         - the image number that will be rutruns in Google method only.


  """
  descrption=trans(text)
  if method=="AI":
    prompts = [ImaginePrompt(descrption),]#descrption=وصف الشئ المفقود
    for result in imagine(prompts):
      result.img.save(f"folders/genarated_images/{descrption}.png")
  if method=="google".upper():
    respone=simp.simple_image_download
    keyword=descrption
    respone().download(keyword,img_number+5)
    img_path=glob.glob(os.path.join(f'folders/simple_images/{keyword}',"*.*"))
    images=[]
    not_good=[]
    ######## chang dir
    not_good.append(cv2.cvtColor(cv2.imread("folders/not_good_images/notgood.jpeg"),cv2.COLOR_BGR2RGB))
    not_good.append(cv2.cvtColor(cv2.imread("folders/not_good_images/notgood1.jpeg"),cv2.COLOR_BGR2RGB))
    not_good.append(cv2.cvtColor(cv2.imread("folders/not_good_images/notgood2.jpeg"),cv2.COLOR_BGR2RGB))
    not_good.append(cv2.cvtColor(cv2.imread("folders/not_good_images/notgood3.jpeg"),cv2.COLOR_BGR2RGB))
    for image in img_path:
      img=cv2.imread(image)
      if img is not None:
        rgb_img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        if any(np.array_equal(rgb_img, not_good_img) for not_good_img in not_good):
          os.remove(image)
          continue
        images.append(image)
      else:
        os.remove(image)


create_img("google", "قطه بيضاء اللون", img_number=10)
