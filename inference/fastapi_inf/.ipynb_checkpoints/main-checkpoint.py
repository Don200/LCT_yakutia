from typing import List

from fastapi import FastAPI
from prompts import PromptGenerator
from pydantic_struct import DatasetClients, ClientTextResponse, Client, ClientRegen
from saigamodel import Conversation, Model

app = FastAPI(title='ML model and personal promt generation')

conversation = Conversation()
model = Model()


@app.post("/text")
def generate_offer(json_dict: DatasetClients) -> List[ClientTextResponse]:
    """
    Основная ручка на бэк. Принимает батч в виде списка данных о клиентах. Подробнее в фале с классами pydantic
    :param json_string: принимает json c бека
    :return: promt персонализированный под каждого клиента банка
    """
    data = json_dict.dict()
    prompt_generator = PromptGenerator()

    config = model.set_config(model.generation_config, data['args']['temp'], data['args']['top_p'])
    result = []
    # итерируемся по массиву клиентов
    for client in data['clients']:

        actual_feat = prompt_generator.take_feat(client)
        given_feat = [[key, value] for key, value in client.items() if key in actual_feat]

        # итерируемся по списку продуктов и списку
        for product in data['products']:
            for channel in data['channels']:
                super_clust = client['super_clust']
                prompt = prompt_generator.get_prompt(super_clust, channel['name'], product['description'],
                                                     actual_feat, given_feat, )
                conversation.add_user_message(prompt)
                print('row:' + prompt)
                prompt = conversation.get_prompt(model.tokenizer)
                print('conv:' + prompt)
                output = model.generate(model.model, model.tokenizer, prompt, config)

                result.append({
                    'text': output,
                    'client_id': client['id'],
                    'product_id': product['id'],
                    'channel_id': channel['id']
                })
    return result


@app.post("/regen")
def regenerate_text(json_dict: ClientRegen) -> str:
    """
    Принимает данные о клиенте, запрос которого нужно перегенерировать
    :param json_dict: json валидированный по формату с `Client`. Подробнее в фале с классами pydantic
    :return: новый текст с учетом комментариев
    """
    data = json_dict.dict()
    prompt_generator = PromptGenerator()
    print(data)
    config = model.set_config(model.generation_config, data['args']['temp'], data['args']['top_p'])

    client = data['client']
    channel = data['channel']
    product = data["product"]
    comment = data['text'] + 'не понравилось следующее:' + data['comment']

    actual_feat = prompt_generator.take_feat(client)
    given_feat = [[key, value] for key, value in client.items() if key in actual_feat]

    super_clust = client['super_clust']
    prompt = prompt_generator.get_prompt(super_clust, channel['description'], product['description'],
                                         actual_feat, given_feat, comment )
    conversation.add_user_message(prompt)
    prompt = conversation.get_prompt(model.tokenizer)
    output = model.generate(model.model, model.tokenizer, prompt, config)

    return output
