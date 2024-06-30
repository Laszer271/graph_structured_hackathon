from chat.prompts import history_prompt
from chat.utils import count_tokens, messages_history
from openai import OpenAI
import json

#tests
import os
from dotenv import load_dotenv
load_dotenv()


class ChatManager:
    def __init__(self,client: OpenAI,model:str) -> None:
        self.client = client
        self.model = model

    def query_model(self, messages_history:list, prompt:str):
        messages_history = [{"role": m["role"], "content": m["content"]} for m in messages_history] if len(messages_history) > 0 else []
        input_messages = [({"role": "user", "content": prompt})]

        stream = self.client.chat.completions.create(
            model = self.model,
            messages = messages_history + input_messages,
            stream = True
        )
        return stream
    
    @staticmethod
    def _history_extractor(response):
        try:
            new_history = json.loads(response)
            return [{"role": "system", "content": "History of current conversation: \n" + new_history["history"]}]
        except json.JSONDecodeError:
            return [{"role": "system", "content": response}]
        except KeyError:
            return [{"role": "system", "content": response}]
    
    def manage_history(self, prev_history:list) -> list:
        #If history is empty, return it
        if (len(prev_history) == 0):
            return prev_history
    
        # Change the history to a string
        prev_history_str = json.dumps(prev_history)

        #Count length of history
        history_length = count_tokens(prev_history_str, self.model)
        print("History length: ", history_length)

        #If history is greater or equal to 500 tokens generate new history by summarizing the previous one
        if history_length >= 500:
            response =  self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role":"system", "content": history_prompt},
                    {"role":"user", "content": "This is previous history: " + prev_history_str + " Return this in json format as given: {{history: summary here}} "}
                ],
                max_tokens=300,
                stop=None,
                temperature=0.7,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                n=1,
                logprobs=None,
                logit_bias=None,
                response_format={ "type": "json_object" }
            )

            print("Total tokens used: ", response.usage.total_tokens)
            print("Number of tokens in history answer: ", response.usage.completion_tokens)

            #Extract 
            return self._history_extractor(response.choices[0].message.content)

        else:
            return prev_history
        
if __name__ == "__main__":
    # ------ TESTS ------
    ai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    manager = ChatManager(client=ai_client, model="gpt-3.5-turbo")
    manager.manage_history(json.dumps(messages_history))