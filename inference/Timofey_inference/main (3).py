import json
import pickle
from ast import literal_eval
from fastapi import FastAPI
from saigamodel import Conversation, generation_config, generate, model, tokenizer

app = FastAPI(title='ML model and personal promt generation')

conversation = Conversation()

# =============Подгрузка инфо. фалов====================#
with open('top7.json', 'r', encoding="utf-8") as f:
    top7 = json.load(f)

with open("describe.pickle", "rb") as f:
    describe = pickle.load(f)

with open("description.pickle", "rb") as f:
    description = pickle.load(f)

with open('channels.pickle', 'rb') as f:
    channels = pickle.load(f)
# ======================================================#

input_json = [{
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
        "sum_zp_12m": 1217025.5,
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


@app.get("/")
def generate_offer(json_dict):
    """
    Основная ручка на бэк
    :param json_string: принимает json c бека
    :return: promt персонализированный под каждого клиента банка
    """
    data = literal_eval(json_dict.decode('utf8'))
    data = json.dumps(data, indent=2, sort_keys=True)

    # итерируемся по массиву клиентов
    for s in input_json:
        def take_feat(s):
            cluster = s['client_info']['super_clust']
            pairs = top7[cluster]
            return [feat[0] for feat in pairs]

        def check(actual_feat, given_feat, describe):
            """
            `check` генерирует описание данные о топ-7 фичах для данного кластера
            :param actual_feat: всегда вызов ф-ции `take_feat`
            :param given_feat: список из пар [feature, value]
            :param describe: базовое описательные статистики для каждого числового признака
            :return:
            """
            text = ''
            ans = dict()
            for i in range(len(given_feat)):
                for j in range(len(given_feat)):
                    if actual_feat[i] in given_feat[j][0]:
                        if given_feat[j][0] == 'visit_purposes':
                            ans[given_feat[j][0]] = '' + given_feat[j][1]
                        elif given_feat[j][0] == 'reg_region_nm':
                            ans[given_feat[j][0]] = '' + given_feat[j][1]
                        else:
                            if given_feat[j][1] <= describe[given_feat[j][0]]['mean']:
                                ans[given_feat[j][0]] = "меньше, чем этот же показатель в среднем у клиентов банка"
                            else:
                                ans[given_feat[j][0]] = "больше, чем этот же показатель в среднем у клиентов банка"

            for feat in actual_feat:
                text += ''.join(description[feat] + ' ' + ans[feat]) + '\n'
            return text

        actual_feat = take_feat(s)
        given_feat = [[key, value] for key, value in s['client_info'].items() if key in actual_feat]

        # итерируемся по списку текстов
        for text in s["text_info"]:
            product = (text['product']['product_description'])
            channel = channels[text['chanel']['chanel_name']]
            note = text['note']
            if s['client_info']['super_clust'] != 'e. Супер-аффлуент (-1)':
                promt = (f'{channel} для клиента Газпромбанка. '
                         f'Порекомендуй ему {product}.'
                         f'Вот важная информации о клиенте {check(actual_feat, given_feat, describe)}.'
                         f'Так же учти эту очень важную информацию: {note}')
            else:
                promt = (f'{channel} для очень ценного клиента Газпромбанка, '
                         f'у него большой счет в банке или депозит. Порекомендуй ему {product}.'
                         f'В его сообщении должен чувствоваться индивидуальный '
                         f'подход к клиенту. Вот важная информации о клиенте {check(actual_feat, given_feat, describe)}'
                         f'Так же учти эту очень важную информацию: {note}')

            conversation.add_user_message(promt)
            prompt = conversation.get_prompt(tokenizer)
            output = generate(model, tokenizer, prompt, generation_config)

            text['text'] = output
    return
