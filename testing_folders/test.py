# from fastapi import FastAPI, HTTPException
# from fastapi.responses import JSONResponse
# from pydantic import BaseModel
# from googletrans import Translator
# from langdetect import detect
# import spacy

# app = FastAPI()

# class ObjectFinder:
#     def __init__(self):
#         self.translator = Translator()
#         self.nlp = spacy.load("en_core_web_lg")

#     def translate_to_english(self, find_object, founded_objects):
#         en_founded_objects = []

#         for text in founded_objects:
#             if text is not None:
#                 try:
#                     source_lang = detect(text)
#                     translation_result = self.translator.translate(text, src=source_lang, dest='en')
#                     en_founded_objects.append(translation_result.text)
#                 except Exception as e:
#                     print(f"Error translating '{text}': {str(e)}")
#                     en_founded_objects.append(None)
#             else:
#                 en_founded_objects.append(None)

#         try:
#             source_lang = detect(find_object)
#             en_find_object = self.translator.translate(text=find_object, src=source_lang, dest='en').text
#         except Exception as e:
#             print(f"Error translating : {str(e)}")
#             en_find_object = None

#         return en_find_object, en_founded_objects

#     def compare_objects_similarity(self, find_object, founded_objects):
#         find_object_doc = self.nlp(find_object)
#         similarity_scores = []

#         for obj in founded_objects:
#             if obj is not None:
#                 obj_doc = self.nlp(obj)
#                 similarity_score = find_object_doc.similarity(obj_doc)
#                 similarity_scores.append((obj, similarity_score))
#             else:
#                 similarity_scores.append((None, 0.0))

#         sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
#         top_matches = sorted_scores[:3]

#         return top_matches

# class FindObjectsRequest(BaseModel):
#     find_object: str
#     founded_objects: list

# class FindObjectsResponse(BaseModel):
#     top_matches: list

# object_finder = ObjectFinder()

# @app.post("/find_objects", response_model=FindObjectsResponse, tags=['Comparing Text'])
# async def find_objects(request: FindObjectsRequest):
#     print(request.dict())  # Add this line for debugging
#     try:
#         if not request.find_object:
#             raise HTTPException(status_code=400, detail="Please provide a description")

#         en_find_object, en_founded_objects = object_finder.translate_to_english(
#             find_object=request.find_object, founded_objects=request.founded_objects
#         )
#         top_matches = object_finder.compare_objects_similarity(en_find_object, en_founded_objects)

#         response = {
#             "top_matches": top_matches
#         }

#         return JSONResponse(content=response)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8000)


from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from googletrans import Translator
from langdetect import detect
import spacy

app = FastAPI()

class ObjectFinder:
    def __init__(self):
        self.translator = Translator()
        self.nlp = spacy.load("en_core_web_lg")

    def translate_to_english(self, find_object, founded_objects):
        en_founded_objects = []

        for text in founded_objects:
            if text is not None:
                try:
                    source_lang = detect(text)
                    translation_result = self.translator.translate(text, src=source_lang, dest='en')
                    en_founded_objects.append(translation_result.text)
                except Exception as e:
                    print(f"Error translating '{text}': {str(e)}")
                    en_founded_objects.append(None)
            else:
                en_founded_objects.append(None)

        try:
            source_lang = detect(find_object)
            en_find_object = self.translator.translate(text=find_object, src=source_lang, dest='en').text
        except Exception as e:
            print(f"Error translating : {str(e)}")
            en_find_object = None

        return en_find_object, en_founded_objects

    def compare_objects_similarity(self, find_object, founded_objects):
        find_object_doc = self.nlp(find_object)
        similarity_scores = []

        for obj in founded_objects:
            if obj is not None:
                obj_doc = self.nlp(obj)
                similarity_score = find_object_doc.similarity(obj_doc)
                similarity_scores.append((obj, similarity_score))
            else:
                similarity_scores.append((None, 0.0))

        sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        top_matches = [match[0] for match in sorted_scores[:3]]

        return top_matches

class FindObjectsRequest(BaseModel):
    find_object: str
    founded_objects: list

class FindObjectsResponse(BaseModel):
    top_matches: list

object_finder = ObjectFinder()

@app.post("/find_objects", response_model=FindObjectsResponse, tags=['Comparing Text'])
async def find_objects(request: FindObjectsRequest):
    print(request.dict())  # Add this line for debugging
    try:
        if not request.find_object:
            raise HTTPException(status_code=400, detail="Please provide a description")

        en_find_object, en_founded_objects = object_finder.translate_to_english(
            find_object=request.find_object, founded_objects=request.founded_objects
        )
        top_matches = object_finder.compare_objects_similarity(en_find_object, en_founded_objects)

        response = {
            "top_matches": top_matches
        }

        return JSONResponse(content=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
