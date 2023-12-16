from typing import List

from fastapi import FastAPI
from prompts import PromptGenerator
from pydantic_struct import DatasetClients, ClientTextResponse, Client
from saigamodel import Conversation, Model

app = FastAPI(title='ML model and personal promt generation')

conversation = Conversation()
model = Model()


input_json = {
    "clients": [
        {
        "id": "1785583955",
        "gender": 0,
        "age": 52.68817,
        "reg_region_nm": "Ханты-Мансийский автономный округ - Югра",
        "cnt_tr_all_3m": 201,
        "cnt_tr_top_up_3m": 18,
        "cnt_tr_cash_3m": 6,
        "cnt_tr_buy_3m": 97,
        "cnt_tr_mobile_3m": 7,
        "cnt_tr_oil_3m": 46,
        "cnt_tr_on_card_3m": 11,
        "cnt_tr_service_3m": 2,
        "cnt_zp_12m": 33,
        "sum_zp_12m": 1217025,
        "limit_exchange_count": 0,
        "max_outstanding_amount_6m": -1,
        "avg_outstanding_amount_3m": -1,
        "cnt_dep_act": 0,
        "sum_dep_now": 279.88,
        "avg_dep_avg_balance_1month": 277.87,
        "max_dep_avg_balance_3month": 276.02762,
        "app_vehicle_ind": 0,
        "app_position_type_nm": "-1",
        "visit_purposes": "DCARD",
        "qnt_months_from_last_visit": 10,
        "super_clust": "a. Супер-ЗП (6,15)"
        }
    ],
    "products": [{
        "id": 1,
        "name": "ПК",
        "description": "Классический потребительский кредит"
    }],
    "channels": [{
        "id": 1,
        "name": "TMO",
        "description": "Колл центр (телемеркетинг)"
    }],
    "args": {
        "temp": 0.5,
        "top_p": 0.5
    }
}


@app.post("/text")
def generate_offer(json_dict: DatasetClients) -> List[ClientTextResponse]:
    """
    Основная ручка на бэк. Принимает батч в виде списка данных о клиентах. Подробнее в фале с классами pydantic
    :param json_string: принимает json c бека
    :return: promt персонализированный под каждого клиента банка
    """
    data = json_dict.model_dump()
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
                prompt = prompt_generator.get_prompt(super_clust, channel['description'], product['description'],
                                                     actual_feat, given_feat, )
                conversation.add_user_message(prompt)
                prompt = conversation.get_prompt(model.tokenizer)

                output = model.generate(model.model, model.tokenizer, prompt, config)

                result.append({
                    'text': output,
                    'client_id': client['id'],
                    'product_id': product['id'],
                    'channel_id': channel['id']
                })
    return result


@app.post("text/")
def regenerate_text(json_dict: Client) -> str:
    """
    Принимает данные о клиенте, запрос которого нужно перегенерировать
    :param json_dict: json валидированный по формату с `Client`. Подробнее в фале с классами pydantic
    :return: новый текст с учетом комментариев
    """
    data = json_dict.model_dump()
    prompt_generator = PromptGenerator()
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
