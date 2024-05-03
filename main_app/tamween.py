import numpy as np
import cv2
import imutils
from imutils import contours
from paddleocr import PaddleOCR
import datetime as dt

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
        Governorates=["Cairo","Alexandria","Port Said","Suez","Damietta",
                      "Dakahlia","elsharkia","Qalyubia","Kafr El-Sheikh",
                      "elgharbia","Menoufia","elbehera","Ismailia","Giza",
                      "Bani Sweif","Fayoum","Minya","Asyut","Sahaj","Qena",
                      "Aswan","Luxor","The Red Sea","the new Valley","matrooh",
                      "North Sinai","South of Sinaa","out of Egypt"]
        
        ara_Governorates=["القاهرة","الأسكندرية","بور سعيد","السويس","دمياط","الدقهلية",
                          "الشرقية","القليوبية","كفر الشيخ","الغربية","المنوفية","البحيرة",
                          "الاسماعيلية","الجيزة","بني سويف","الفيوم","المنيا","أسيوط","سوهاج",
                          "قنا","اسوان","الأقصر","البحر الأحمر","الوادي الجديد","مطروح","شمال سيناء",
                          "جنوب سيناء","خارج مصر"]
        
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

class IDExtractor:
    def __init__(self, image_path):
        self.image_path = image_path
        self.ocr = PaddleOCR(use_angle_cls=True, lang='ar')

        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Failed to load image. Check the file path and format.")
            self.processed_image = cv2.resize(image, (700, 480))
        except Exception as e:
            print(f"Error: {e}")
            # Handle the error or raise it again based on your application's logic
            raise

        # self.processed_image = cv2.resize(cv2.imread(image_path), (700, 480))

    def preprocess_image(self, region):
        region = cv2.bilateralFilter(region, 11, 17, 17)
        return region

    def get_numbers_from_image(self, region):
        region = cv2.GaussianBlur(region, (3, 3), 0)
        ref = cv2.threshold(region, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        kernel = np.ones((2, 2), np.uint8)
        ref = cv2.dilate(ref, kernel, iterations=1)
        ref_cnts = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ref_cnts = imutils.grab_contours(ref_cnts)
        ref_cnts = contours.sort_contours(ref_cnts, method="left-to-right")[0]
        digits = {}
        i = 0
        for c in ref_cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            roi = region[y - 3 : y + h + 3, x - 3 : x + w + 3]
            if len(roi) > 0:
                roi = cv2.resize(roi, (57, 88))
                roi = cv2.bilateralFilter(roi, 11, 17, 17)
                digits[i] = roi
                i += 1
        return digits

    def get_number(self, line_num):
        img2 = self.processed_image
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        id_region = 0
        if line_num == 1:
            id_region = gray2[349:400, 355:]
        elif line_num == 2:
            id_region = gray2[390:435, 400:]

        num_templates = self.load_number_templates()
        digits = self.get_numbers_from_image(id_region)
        numbers = []
        for num in range(len(digits)):
            for number in range(10):
                w, h = num_templates[number].shape[::-1]
                res = cv2.matchTemplate(
                    digits[num], num_templates[number], cv2.TM_CCORR_NORMED
                )
                thresh = 0.999
                thresh2 = thresh
                loc = np.where(res >= thresh)
                if len(loc[0]) > 0 and len(loc[1]) > 0:
                    numbers.append(number)
                    break
                elif len(loc[0]) == 0 and len(loc[1]) == 0 and number == 9:
                    loc2 = ([], [])
                    thresh2 = thresh - 0.001
                    while len(loc2[0]) == 0 and len(loc2[1]) == 0:
                        for number2 in range(10):
                            res2 = cv2.matchTemplate(
                                digits[num],
                                num_templates[number2],
                                cv2.TM_CCORR_NORMED,
                            )
                            loc2 = np.where(res2 >= thresh2)
                            if len(loc2[0]) > 0 and len(loc2[1]) > 0:
                                numbers.append(number2)
                                break
                            elif (
                                len(loc2[0]) == 0
                                and len(loc2[1]) == 0
                                and number2 == 9
                            ):
                                thresh2 -= 0.001
        if line_num == 1:
            numbers = numbers[0:14]
        elif line_num == 2:
            numbers = numbers[0:12]
        return numbers
    
    # chang dir
    def load_number_templates(self):
        templates = []
        for i in range(10):
            template = cv2.imread(f"folders/arabic_numbers/number_{i}.png", 0) ############## change dir
            template = cv2.resize(template, (57, 88))
            template = cv2.bilateralFilter(template, 11, 17, 17)
            templates.append(template)
        return templates

    def get_data(self):
        id_data = [self.processed_image[290:360, 320:680]]
        data = ["name"]
        data_out = {}

        for x, y in zip(id_data, data):
            text = self.ocr.ocr(x, cls=True)
            data_out[y] = text

        length = 0
        boxes = {}
        txts = {}
        scores = {}
        for key, item in data_out.items():
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
        for key, item in new_data_out.items():
            for i in item[::-1]:
                if key == "name":
                    name += i + " "

        return {"name": name.strip()}

    def revers_txt(self, txt_list):
        new_list = []
        for i in txt_list:
            temp = self.split_txt(i)
            new_txt = ""
            for c in range(len(temp), 0, -1):
                new_txt += temp[c - 1]
            new_list.append(new_txt)
        return new_list

    def split_txt(self, text):
        temp = []
        for i in text:
            temp.append(i)
        return temp

    def get_final_data(self):
        data = self.get_data()
        new_data = {}
        new_data["id"] = self.get_number(1)
        new_data["id_2"] = self.get_number(2)
        for key, item in data.items():
            new_data[key] = item
        id_1 = ""
        id_2 = ""
        for key, value in new_data.items():
            if key == "id":
                for i in value:
                    id_1 += str(i)
            elif key == "id_2":
                for i in value:
                    id_2 += str(i)

        month = int(id_1[3:5])
        day = int(id_1[5:7])
        if month > 12 or month == 0:
            id_1 = id_1.replace(id_1[3:5], "12")
            # new_data["problem"] = "Date of birth and age will not be correct"
        if day > 31 or day == 0:
            id_1 = id_1.replace(id_1[5:7], "29")
            # new_data["problem"] = "Date of birth and age will not be correct"

        new_data["id"] = id_1
        new_data["id_2"] = id_2
        x = ID(int(id_1))
        x1 = x.get_BirthDate()
        x2 = x.get_BirthPlace()
        x3 = x.get_Gender()
        if x3 == "Female":
            x3 = "انثى"
        elif x3 == "Male":
            x3 = "ذكر"
        age = self.Age(x1)
        new_data["birth_date"] = x1.strftime("%Y-%m-%d")
        new_data["birth_place"] = x2
        new_data["gender"] = x3
        new_data["age"] = f"{age[0]} year and {age[1]} month"
        return new_data

    def Age(self, birth_date):
        now = dt.datetime.today()
        if now.month >= birth_date.month:
            M = now.month - birth_date.month
        else:
            M = (now.month - birth_date.month) + 12
        d = now.day - birth_date.day
        days = (now - birth_date).days
        if d < 0:
            F = 30
            if (
                birth_date.month == 1
                or birth_date.month == 3
                or birth_date.month == 5
                or birth_date.month == 7
                or birth_date.month == 8
                or birth_date.month == 10
                or birth_date.month == 12
            ):
                F = 31
            if birth_date.month == 2:
                F = 28
            d = d + F
        years = days // 365
        return (years, M)

    def get_num(self, num):
        if num == '١':
            return "1"
        if num == '٢':
            return "2"
        if num == "٣":
            return "3"
        if num == '٤':
            return "4"
        if num == '٥':
            return "5"
        if num == '٦':
            return "6"
        if num == '٧':
            return "7"
        if num == '٨':
            return "8"
        if num == '٩':
            return "9"

    def convert_to_eng(self, input_dict):
        new_dict = input_dict.copy()
        id_value = new_dict["id"]
        id_2_value = new_dict["id_2"]
        new_id = ""
        new_id_2 = ""
        for i in id_value:
            if i == " ":
                continue
            new_id += str(self.get_num(i))
        for x in id_2_value:
            if x == " ":
                continue
            new_id_2 += str(self.get_num(x))

        for key, value in new_dict.items():
            if key == "id":
                new_dict[key] = new_id
            if key == "id_2":
                new_dict[key] = new_id_2
        return new_dict