import json
import pickle
from ast import literal_eval
from fastapi import FastAPI
from prompts import PromptGenerator
from pydantic_struct import Client, DatasetClients, ClientTextRequest, ClientRequest, Request
from saigamodel import Conversation, generate, Model

app = FastAPI(title='ML model and personal promt generation')

conversation = Conversation()
model = Model()


input_json = {
    "clients" : [{
    "vid": "1785583955",
    "dataset_id": {
        "dataset_id": 1751554618,
        "dataset_name": "",
        "dataset_comment": ""
    },
    "client_info": {
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
    },
    "text_info": [
        {
            "text_id": 10,
            "text": " text ",
            "note": "ya zaebalsa",
            "product": {
                "product_id": 1,
                "product_name": "ПК",
                "product_description": "Классический потребительский кредит"
            },
            "chanel": {
                "chanel_id": 1,
                "chanel_name": "TMO",
                "chanel_description": "Колл центр (телемеркетинг)"
            },
            "is_good": 1,
            "temperature": 0.3,
            "top_p": 0.4
        }
    ]
}]
}

@app.post("/text")
def generate_offer(json_dict: DatasetClients) -> Request:
    """
    Основная ручка на бэк
    :param json_string: принимает json c бека
    :return: promt персонализированный под каждого клиента банка
    """
    data = json_dict.model_dump()
    prompt_generator = PromptGenerator()
    # итерируемся по массиву клиентов
    for s in data['clients']:

        actual_feat = prompt_generator.take_feat(s)
        given_feat = [[key, value] for key, value in s['client_info'].items() if key in actual_feat]

        # итерируемся по списку текстов
        for text in s["text_info"]:
            product = (text['product']['product_description'])
            channel = prompt_generator.get_channel(text['chanel']['chanel_name'])
            note = text['note']
            super_clust = s['client_info']['super_clust']
            prompt = prompt_generator.get_prompt(super_clust, channel, product, note, actual_feat, given_feat,)

            conversation.add_user_message(prompt)
            prompt = conversation.get_prompt(model.tokenizer)
            output = model.generate(model.model, model.tokenizer, prompt, model.generation_config)

            text['text'] = output
        del s['client_info']
    
    data['UserRequest'] = data.pop('clients')
    return data
