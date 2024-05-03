import cv2
import datetime as dt
from googletrans import Translator
from paddleocr import PaddleOCR
import face_recognition

ocr = PaddleOCR(use_angle_cls=True, lang='ar')

class ID:
    def __init__(self,number:int):

        self.Number=number

    def get_BirthDate(self):
        """
         A Function that extract the brith date from the National ID card number.

         Parameters:
         - the id card number with type intger.

         Returns:
         - it return BirthDate of the id card owner with type datetime.
        """
        centray=""
        Birthyear=""
        self.Number=str(self.Number)
        if self.Number[0]=="2":
            centray="1900"
        elif self.Number[0]=="3":
            centray="2000"
        Birthyear=int(centray)+int(self.Number[1:3])
        Month=int(self.Number[3:5])
        Day=int(self.Number[5:7])
        BirthDate=dt.datetime(Birthyear,Month,Day)
        return BirthDate

    def get_BirthPlace(self):
        """
         A Function that extract the Birth Place from the National ID card number.

         Parameters:
         - the id card number with type intger.

         Returns:
         - it return Birth Place of the id card owner with type string.
        """
        self.Number=str(self.Number)
        BirthPlace_num=self.Number[7:9]
        Governorates=["Cairo","Alexandria","Port Said","Suez","Damietta","Dakahlia","elsharkia","Qalyubia","Kafr El-Sheikh","elgharbia","Menoufia","elbehera","Ismailia","Giza","Bani Sweif","Fayoum","Minya","Asyut","Sahaj","Qena","Aswan","Luxor","The Red Sea","the new Valley","matrooh","North Sinai","South of Sinaa","out of Egypt"]
        ara_Governorates=["القاهرة","الأسكندرية","بور سعيد","السويس","دمياط","الدقهلية","الشرقية","القليوبية","كفر الشيخ","الغربية","المنوفية","البحيرة","الاسماعيلية","الجيزة","بني سويف","الفيوم","المنيا","أسيوط","سوهاج","قنا","اسوان","الأقصر","البحر الأحمر","الوادي الجديد","مطروح","شمال سيناء","جنوب سيناء","خارج مصر"]
        G_Numbers=[1,2,3,4]
        FinalDict={}

        for i in range(11,36):
            if i==30 or i==20:
                continue
            G_Numbers.append(i)

        G_Numbers.append(88)
        x=0
        for i in G_Numbers:
            FinalDict[i]=ara_Governorates[x]
            x=x+1

        BirthPlace_num=int(BirthPlace_num)
        return FinalDict[BirthPlace_num]

    def get_Gender(self):
        """
         A Function that extract the Gender from the National ID card number.

         Parameters:
         - the id card number with type intger.

         Returns:
         - it return Gender of the id card owner with type string (male,female).
        """
        self.Number=str(self.Number)
        Gender=self.Number[len(self.Number)-2]
        Gender=int(Gender)
        if Gender%2==0:
            return "Female"
        else:
            return "Male"

def Age(BirthDate:dt.datetime):
    """
         A Function that Calculate the Age of specific BirthDate.

         Parameters:
         - A BirthDate of the id card owner with type datetime.

         Returns:
         - it return tuble of the with type int the id owner age in Years and how much Months.
    """
    now=dt.datetime.today()
    if now.month>=BirthDate.month:
        M=now.month-BirthDate.month
    else:
       M=(now.month-BirthDate.month)+12
    d=now.day-BirthDate.day
    Days=(now-BirthDate).days
    if d<0:
        F=30
        if BirthDate.month==1 or BirthDate.month==3 or BirthDate.month==5 or BirthDate.month==7 or BirthDate.month==8 or BirthDate.month==10 or BirthDate.month==12:
            F=31
        if BirthDate.month==2 :
            F=28
        d=d+F
    Years=Days//365
    return (Years,M)

def split_txt(txt:str):
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

def revers_txt(lis:list):
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

def get_data_paddelOCR(img_path:str):
  """
  A Function that extract the text from image or specific part from image using paddleOCR.

  Parameters:
  - The image path of the image.

  Returns:
  - it return a Dictionary and the Values of Dictionary the words that been extracted from image and the keys of the Dictionary is specific data
    from the image that we focus on it to be extracted like Frist Name and the rest of the name and so on.
  """
  img=cv2.imread(img_path)

  processed_image = cv2.resize(img, (700, 480))

  processed_image=cv2.cvtColor(processed_image,cv2.COLOR_BGR2RGB)

  m = processed_image[20:80, 200:]
  Nationality=processed_image[225:275, 570:]
  Nationality=cv2.bilateralFilter(Nationality,11,17,17)
  mehna = processed_image[265:320, 450:]
  mehna=cv2.bilateralFilter(mehna,11,17,17)
  id = processed_image[65:115 , 200:470]
  id = cv2.resize(id, (1080, 350))
  name = processed_image[105:155, 250:]
  address = processed_image[175:230, 200:]
  id_data = [id,name,address,m,Nationality,mehna]
  data=["id","name","address","traffic_unit","nationality","mehna"]


  data_out={}


  for x,y in zip(id_data,data):
      text = ocr.ocr(x, cls=True)
      data_out[y]=text


  length=0
  boxes = {}
  txts = {}
  scores = {}
  for key,item in data_out.items():
    if data_out[key][0]==None:
      continue
    length=len(data_out[key][0])
    boxes[key]=[item[0][0][0]]
    txts[key]=[item[0][0][1][0]]
    scores[key]=[item[0][0][1][1]]
    for i in range(1,length):
      boxes[key].append(item[0][i][0])
      txts[key].append(item[0][i][1][0])
      scores[key].append(item[0][i][1][1])

  new_data_out={}
  id=""
  for key,item in txts.items():
    if key=='id':
      id=item[0]
    new_data_out[key]=revers_txt(item)


  name=""
  address=""
  traffic_unit=""
  mehna=""
  Nationality=""
  for key,item in new_data_out.items():
    for i in item[::-1]:
      if key=="nationality":
        Nationality+=i+' '
      if key=="traffic_unit":
        traffic_unit+=i+' '
      if key=="mehna":
        mehna+=i+' '
      if key=='name':
        name+=i+" "
      if key=='address':
        address+=i+" "
  return {'id':id.strip(),"name":name.strip(),"address":address.strip(),
          'traffic_unit':traffic_unit.strip(),'nationality':Nationality.strip()
          ,'job':mehna.strip()}

def trans(txt:str):
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

def get_face(img_path:str):
  """
  A Function that extract the face from Egyptian national ID card.

  Parameters:
  - The image path of the image.

  Returns:
  - corp image with type np.array that have exactly the the face of the person who hold the ID card.
  """
  img1=cv2.imread(img_path)
  rgb_img=cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
  face_loction=face_recognition.face_locations(rgb_img)
  for loc in face_loction:
    y1,x2,y2,x1=loc[0],loc[1],loc[2],loc[3]
    # cv2.rectangle(img1,(x1,y1),(x2,y2),(0,255,0),4)
  return img1[y1-30:y2+18,x1-10:x2+18]

def get_final_data(img_path):
  data=get_data_paddelOCR(img_path)
  id=""
  for key,value in data.items():
    if key=="id":
      for i in value:
        id+=str(i)
  data['name_in_english']=t=trans(data['name'])
  x=ID(int(id))
  x1=x.get_BirthDate()
  x2=x.get_BirthPlace()

  x3=x.get_Gender()
  if x3=="Female":
    x3="انثى"
  elif x3=="Male":
    x3="ذكر"
  age=Age(x1)
  data["birth_date"]=x1.strftime("%Y-%m-%d")
  data["birth_place"]=x2
  data["gender"]=x3
  data["age"]= f"{age[0]} year and {age[1]} month"
  # base_name=os.path.basename(img_path)
  # (file_Name,ext)=os.path.splitext(base_name)
  # cv2.imwrite("/content/face.png",get_face(img_path))
  # new_data["face"]=(get_face(img_path),f"/content/{file_Name}.png")#(مسار الصورة المحفوظةوالصورة نفسها)
  return data