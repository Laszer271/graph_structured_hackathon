from transformers import AutoTokenizer, AutoModelForTokenClassification,  PreTrainedTokenizer, PreTrainedTokenizerFast
import torch
from gliner import GLiNER
from abc import ABC, abstractmethod

class EntitySearcher(ABC):
    @abstractmethod
    def search_entities(self, input: str):
        pass


text = """
The Chihuahua State Public Security Secretariat (SSPE) arrested 35-year-old Salomón C. T. in Ciudad Juárez, found in possession of a stolen vehicle, a white GMC Yukon, which was reported stolen in the city's streets. The arrest was made by intelligence and police analysis personnel during an investigation in the border city. The arrest is related to a previous detention on February 6, which involved armed men in a private vehicle. The detainee and the vehicle were turned over to the Chihuahua State Attorney General's Office for further investigation into the case. My friend ate those pizzas and dumplings. Michał went shopping and bought bananas.
"""
text_pl = """
Sekretariat Bezpieczeństwa Publicznego Województwa Mazowieckiego (SSPW) aresztował 35-letniego Adama K. M. w Warszawie, znalezionego w posiadaniu skradzionego pojazdu, białego GMC Yukon, który został zgłoszony jako skradziony na ulicach miasta. Aresztowanie zostało dokonane przez personel wywiadu i analizy policyjnej podczas dochodzenia w stolicy. Aresztowanie jest powiązane z wcześniejszym zatrzymaniem z dnia 6 lutego, które dotyczyło uzbrojonych mężczyzn w prywatnym pojeździe. Zatrzymany i pojazd zostali przekazani do Mazowieckiego Prokuratora Generalnego w celu dalszego dochodzenia w sprawie. Zjadł mój kolega te pizze i pierogi. Michał poszedł na zakupy i kupił banany."""


class BertEntitySearcher(EntitySearcher):
    def __init__(self):
        self.tokenizer =AutoTokenizer.from_pretrained("dslim/bert-base-NER")
        self.model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
    
    def _tokenize(self,text_input):
        try:
            return self.tokenizer(text_input, add_special_tokens=False, return_tensors="pt")
        except Exception as e:
            raise f"Error during tokenization of input text {e}"
        
    def search_entities(self, input: str):

        # Tokenize
        tokens_input = self._tokenize(input)

        # Get logits - entities in the text
        with torch.no_grad():
            logits = self.model(**tokens_input).logits
        predicted_token_class_ids = logits.argmax(-1)
        predicted_tokens_classes = [self.model.config.id2label[t.item()] for t in predicted_token_class_ids[0]]

        tokens = self.tokenizer.convert_ids_to_tokens(tokens_input["input_ids"][0].tolist())
        word_ids = tokens_input.word_ids()
        entities = []
        current_word = None
        for token, tag, word_id in zip(tokens, predicted_tokens_classes, word_ids):
            if word_id is not None:
                if word_id != current_word:
                    entities.append({"word": token, "entity": tag})
                    current_word = word_id
                else:
                    entities[-1]["word"] += token.replace("##", "")
            else:
                entities.append({"word": token, "entity": tag})
        
        for entity in entities:
            print(f"Word: {entity['word']} - Entity: {entity['entity']}")

        return entities
    

class GLiNEREntitySearcher(EntitySearcher):
    def __init__(self, language="en") -> None:
        self.model = GLiNER.from_pretrained("EmergentMethods/gliner_medium_news-v2.1")
        self.labels = ["osoba", "miejsce", "data", "wydarzenie", "obiekt", "pojazd", "numer", "organizacja", "jedzenie"] if language=="pl" else ["person", "location", "date", "event", "facility", "vehicle", "number", "organization", "food"]

    def search_entities(self, input: str):
        entities = self.model.predict_entities(input, self.labels)
        return entities
    

if __name__ == "__main__":
    bert_searcher = BertEntitySearcher()
    bert_searcher.search_entities(text)

    print("\n\n ------------- \n\n")

    gliner_searcher = GLiNEREntitySearcher()
    gliner_searcher.search_entities(text)
