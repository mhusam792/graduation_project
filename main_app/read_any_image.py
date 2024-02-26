from paddleocr import PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='ar')
from PIL import Image

def split_txt(txt:str) -> list:
  """
  A Function that split a text into letters to use it in revers_txt function.

  Parameters:
  - text with type string.

  Returns:
  - it return a List that have letters of the input text.

  """
  temp=[]
  for i in txt:
    temp.append(i)
  return temp

def revers_txt(lis:list) -> list:
  """
  A Function that revers a list of texts we use it to reverse arabic words.

  Parameters:
  - A List of texts.

  Returns:
  - it return a new List that have a list with reversd texts.

  """
  newList=[]
  new_txt=""
  for i in lis:
     temp=split_txt(i)
     new_txt=""
     for c in range(len(temp),0,-1):
       new_txt+=temp[c-1]
     newList.append(new_txt)

  return newList

def get_data_(img_path:str)-> list:
  """
  A Function that extract the text from image or specific part from image using paddleOCR.

  Parameters:
  - The image path of the image.

  Returns:
  - it return a list of the words that been extracted from image .
  """
  result = ocr.ocr(img_path, cls=True)
  length=0
  for line in result:
    length=len(line)
    image = Image.open(img_path).convert('RGB')
  txts = []
  for i in range(0,length):
    txts.append(line[i][1][0])
  new_txt=revers_txt(txts)

  return new_txt

from ar_corrector.corrector import Corrector
def data_corection_(data:list)-> list:
  """
  A Function that correct a Arabic words that been written incorrectly (if it needs to correct).

  Parameters:
  - A list of data we aimed to correct.

  Returns:
  - it return a list of the words that been corrected.

  """
  corr = Corrector()
  new_data=[]
  sent=""
  for s in data:
      new_data.append(corr.contextual_correct(s))
  return new_data
