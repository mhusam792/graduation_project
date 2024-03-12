import cv2
import numpy as np
import datetime as dt
from paddleocr import PaddleOCR
import imutils
from imutils import contours
import face_recognition
import os


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

class ID:
    def __init__(self, number):
        self.Number = number

    def get_BirthDate(self):
        centray = ""
        Birthyear = ""
        self.Number = str(self.Number)
        if self.Number[0] == "2":
            centray = "1900"
        elif self.Number[0] == "3":
            centray = "2000"
        Birthyear = int(centray) + int(self.Number[1:3])
        Month = int(self.Number[3:5])
        Day = int(self.Number[5:7])
        BirthDate = dt.datetime(Birthyear, Month, Day)
        return BirthDate

    def get_BirthPlace(self):
        self.Number = str(self.Number)
        BirthPlace_num = self.Number[7:9]

        Governorates = ["Cairo", "Alexandria", "Port Said", "Suez",
                        "Damietta", "Dakahlia", "elsharkia", "Qalyubia",
                        "Kafr El-Sheikh", "elgharbia", "Menoufia", "elbehera",
                        "Ismailia", "Giza", "Bani Sweif", "Fayoum", "Minya", "Asyut",
                        "Sahaj", "Qena", "Aswan", "Luxor", "The Red Sea", "the new Valley",
                        "matrooh", "North Sinai", "South of Sinaa", "out of Egypt"]

        ara_Governorates = ["القاهرة", "الأسكندرية", "بور سعيد", "السويس", "دمياط", "الدقهلية",
                            "الشرقية", "القليوبية", "كفر الشيخ", "الغربية", "المنوفية", "البحيرة",
                            "الاسماعيلية", "الجيزة", "بني سويف", "الفيوم", "المنيا", "أسيوط", "سوهاج",
                            "قنا", "اسوان", "الأقصر", "البحر الأحمر", "الوادي الجديد", "مطروح", "شمال سيناء",
                            "جنوب سيناء", "خارج مصر"]

        G_Numbers = [1, 2, 3, 4]
        FinalDict = {}

        for i in range(11, 36):
            if i == 30 or i == 20:
                continue
            G_Numbers.append(i)

        G_Numbers.append(88)
        x = 0
        for i in G_Numbers:
            FinalDict[i] = ara_Governorates[x]
            x = x + 1

        BirthPlace_num = int(BirthPlace_num)
        return FinalDict[BirthPlace_num]

    def get_Gender(self):
        self.Number = str(self.Number)
        Gender = self.Number[len(self.Number) - 2]
        Gender = int(Gender)
        if Gender % 2 == 0:
            return "Female"
        else:
            return "Male"

class ImageProcessor:
    def __init__(self):
        self.ocr = PaddleOCR(use_angle_cls=True, lang='ar')

    def get_numbers_from_img(self, id):
        id = cv2.GaussianBlur(id, (5,5), 0)
        ref = cv2.threshold(id, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        kernal = np.ones((2, 2), np.uint8)
        ref = cv2.dilate(ref, kernal, iterations=1)
        refCnts = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        refCnts = imutils.grab_contours(refCnts)
        refCnts = contours.sort_contours(refCnts, method="left-to-right")[0]
        digits = {}
        i = 0

        for c in refCnts:
            (x, y, w, h) = cv2.boundingRect(c)
            roi = id[y - 3:y + h + 3, x - 3:x + w + 3]
            if len(roi) > 0:
                roi = cv2.resize(roi, (57, 88))
                roi = cv2.bilateralFilter(roi, 11, 17, 17)
                digits[i] = roi
                i += 1

        return digits

    # chang directory
    def get_number(self, img_path):
        img2 = cv2.imread(img_path)
        img2 = cv2.resize(img2, (700, 480))
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        id = gray2[360:430, 300:]

        templates = []
        for i in range(10):
            template = cv2.imread(f"folders/arabic_tamween_number/number_{i}.png", 0) ###########
            template = cv2.resize(template, (57, 88))
            template = cv2.bilateralFilter(template, 11, 17, 17)
            templates.append(template)

        num_templates = templates

        dig = self.get_numbers_from_img(id)
        numbers = []
        for num in range(0, len(dig)):
            for number in range(0, 10):
                w, h = num_templates[number].shape[::-1]
                res = cv2.matchTemplate(dig[num], num_templates[number], cv2.TM_CCORR_NORMED)
                thresh = 0.999
                loc = np.where(res >= thresh)
                if len(loc[0]) > 0 and len(loc[1]) > 0:
                    numbers.append(number)
                    break
                elif len(loc[0]) == 0 and len(loc[1]) == 0 and number == 9:
                    loc2 = ([], [])
                    thresh2 = thresh - 0.001
                    while len(loc2[0]) == 0 and len(loc2[1]) == 0:
                        for number2 in range(0, 10):
                            res2 = cv2.matchTemplate(dig[num], num_templates[number2], cv2.TM_CCORR_NORMED)
                            loc2 = np.where(res2 >= thresh2)
                            if len(loc2[0]) > 0 and len(loc2[1]) > 0:
                                numbers.append(number2)
                                break
                            elif len(loc2[0]) == 0 and len(loc2[1]) == 0 and number2 == 9:
                                thresh2 -= 0.001
        
        numbers=numbers[0:14]
        return numbers

    def split_txt(self, txt):
        temp = []
        for i in txt:
            temp.append(i)
        return temp

    def revers_txt(self, lis):
        newList = []
        new_txt = ""
        for i in lis:
            temp = self.split_txt(i)
            new_txt = ""
            for c in range(len(temp), 0, -1):
                new_txt += temp[c - 1]
            newList.append(new_txt)

        return newList

    def get_data_paddleOCR(self, img_path):
        img = cv2.imread(img_path)
        processed_image = cv2.resize(img, (700, 480))
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)

        id = processed_image[360:430, 300:]
        id = cv2.resize(id, (1080, 350))
        f_name = processed_image[117:177, 520:]
        l_name = processed_image[165:220, 340:]
        address = processed_image[220:270, 370:]
        regin = processed_image[265:315, 370:]
        id_data = [f_name, l_name, address, regin]
        data = ["frist name", "rest of name", "address", "regin"]

        data_out = {}

        for x, y in zip(id_data, data):
            text = self.ocr.ocr(x, cls=True)
            data_out[y] = text

        length = 0
        boxes = {}
        txts = {}
        scores = {}
        for key, item in data_out.items():
            if data_out[key][0] == None:
                continue
            length = len(data_out[key][0])
            boxes[key] = [item[0][0][0]]
            txts[key] = [item[0][0][1][0]]
            scores[key] = [item[0][0][1][1]]
            for i in range(1, length):
                boxes[key].append(item[0][i][0])
                txts[key].append(item[0][i][1][0])
                scores[key].append(item[0][i][1][1])

        new_data_out = {}

        for key, item in txts.items():
            new_data_out[key] = self.revers_txt(item)

        name = ""
        rest_of_name = ""
        address = ""
        regin = ""
        for key, item in new_data_out.items():
            for i in item[::-1]:
                if key == 'frist name':
                    name += i + " "
                if key == 'rest of name':
                    rest_of_name += i + " "
                if key == 'address':
                    address += i + " "
                if key == 'regin':
                    regin += i + " "
        return {"frist name": name.strip(), "rest_of_name": rest_of_name.strip(),
                 "address": address.strip(), "regin": regin.strip()}

    def get_face(self, img_path):
        img1 = cv2.imread(img_path)
        rgb_img = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        face_location = face_recognition.face_locations(rgb_img)

        for loc in face_location:
            y1, x2, y2, x1 = loc[0], loc[1], loc[2], loc[3]

        return img1[y1 - 30:y2 + 18, x1 - 10:x2 + 18]

    def get_final_data(self, img_path):
        data = self.get_data_paddleOCR(img_path)
        new_data = {}
        new_data["id"] = self.get_number(img_path)
        for key, item in data.items():
            new_data[key] = item
        id = ""
        for key, value in new_data.items():
            if key == "id":
                for i in value:
                    id += str(i)
        
        Month=int(id[3:5])
        Day=int(id[5:7])

        if Month>12 or Month==0:
            id=id.replace(id[3:5], "12")
            new_data['problem']='Date of birth and age will not be correct'
        if Day>31 or Day==0:
            id=id.replace(id[5:7], "29")
            new_data['problem']='Date of birth and age will not be correct'

        new_data["id"] = id.strip()
        x = ID(int(id))
        x1 = x.get_BirthDate()
        x2 = x.get_BirthPlace()

        x3 = x.get_Gender()
        if x3 == "Female":
            x3 = "انثى"
        elif x3 == "Male":
            x3 = "ذكر"
        age = Age(x1)
        new_data["Birth Date"] = x1.strftime("%Y-%m-%d")
        new_data["Birth Place"] = x2
        new_data["Gender"] = x3
        new_data["Age"] = f"{age[0]} year and {age[1]} month"

        # Save the face image to a file
        face_img = self.get_face(img_path)

        base_name=os.path.basename(img_path)
        (file_Name,ext)=os.path.splitext(base_name)

        face_img_path = f"folders/faces_from_ids/{file_Name}.png"
        cv2.imwrite(face_img_path, face_img)

        new_data["face"] = base_name.strip()
        
        return new_data
