from googletrans import Translator
from langdetect import detect
import spacy

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

                # Only consider matches with similarity scores > 50%
                if similarity_score >= 0.25:
                    similarity_scores.append((obj, similarity_score))
            else:
                similarity_scores.append((None, 0.0))

        sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        top_matches = sorted_scores[:3]

        return top_matches
