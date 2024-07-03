import os
import base64

import openai

class OpenAIDescriber:
    def __init__(self, openai_api_key=None, model='gpt-4o'):
        if openai_api_key is None:
            openai_api_key = os.getenv('OPENAI_API_KEY')
        self.openai_api_key = openai_api_key
        self.model = model

    def _load_image(self, image_path):
        with open(image_path, 'rb') as f:
            image = base64.b64encode(f.read()).decode('utf-8')
        return image
    
    def _get_default_prompt(self):
        prompt = '''
        You are an OCR assistant that ensures that quality of the OCR and image descriptions is top-notch.

        ```ocr_output
        {ocr_text}
        ```

        Based on the non-ideal output from OCR and the image of the document, please perform a full good OCR of the page given to you as an image.
        The ocr_output can guide you, it will help in reading some hard to read passages and help you define the order of the text.
        ocr_output is however non-ideal and it can miss some text (sometimes even big parts for some tables or images).
        If there is a nested image in the image of the page you should describe it in the following format:
        <IMAGE DESCRIPTION START>
        [Your description containing the key information]
        <IMAGE DESCRIPTION END>

        [IMPORTANT] Remember that the image description should be a summary of the image content, it should let someone who sees the description but not the actual image understand the core information in the image.
        [IMPORTANT] While doing the OCR try to preserve the original formatting and text order as much as possible. Pay attention especially to the tables - try to keep the table structure.
        [IMPORTANT] Do not mistake the image of the whole page with a nested image. OCR must be performed on the page image, while image descriptions must be provided for nested images (e.g. photos, schemas, diagrams, graphs).
        Do not include any additional comments besides OCR and image descriptions, yapping is not allowed.
        '''.strip()
        return prompt

    def describe(self, text_path, image_path):
        client = openai.OpenAI(api_key=self.openai_api_key)

        prompt = self._get_default_prompt()
        with open(text_path, 'r') as f:
            ocr_text = f.read()
        prompt = prompt.format(ocr_text=ocr_text)

        base64_image = self._load_image(image_path)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg:base64,{base64_image}",
                    },
                    },
                ],
                }
            ],
            max_tokens=4096,
        )
        return response.choices[0].message.content, dict(response.usage)
