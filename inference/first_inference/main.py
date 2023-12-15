from fastapi import FastAPI
#from saigamodel import Conversation, Model
from pydantic import BaseModel
from pydantic_struct import Client, DatasetClients, ClientTextRequest, ClientRequest, Request
import numpy as np
app = FastAPI(
    title='Kalhoznay zaglushka'
)


@app.post("/text")
def generate_offer(users_info: DatasetClients) -> Request:
    user_requests = []
    for user in users_info:
        text_requests = []
        print(user)
        user = user[1][0]
        for text_prompt in user.text_info: # прогоняемся по списку промптов, которые нужны для юзера
            product = text_prompt.product
            chanel = text_prompt.chanel
            client_text_request = {
            "text_id": text_prompt.text_id,
            "text": "text xui" , # TODO: добавить код модели
            "note": text_prompt.note,
            "product": {
                "product_id": product.product_id,
                "product_name": product.product_name,
                "product_description": product.product_description
            },
            "chanel": {
                "chanel_id": chanel.chanel_id,
                "chanel_name": chanel.chanel_name,
                "chanel_description": chanel.chanel_description
            },
            "is_good": text_prompt.is_good,
            "temperature": text_prompt.temperature,
            "top_p": text_prompt.top_p
            }
            text_requests.append(client_text_request)
            
        # создаем ClientRequest
        user_request = { 
                    'vid': user.vid,
                    'dataset_id': { 'dataset_id': user.dataset_id.dataset_id,
                                    'dataset_name': user.dataset_id.dataset_name,
                                    'dataset_comment': user.dataset_id.dataset_comment},
                    'text_info':text_requests
                    }
        user_requests.append(user_request)
    return {'UserRequest':user_requests}
















'''
accelerate==0.21.0
bitsandbytes==0.40.2
peft==0.5.0
transformers==4.34.0
pandas~=2.1.4
'''

'''for inp in inputs:
        conversation.add_user_message(inp)
        prompt = conversation.get_prompt(model.tokenizer)
        output = model.generate(model, model.tokenizer, prompt, model.generation_config)

        return {"message": output}'''

