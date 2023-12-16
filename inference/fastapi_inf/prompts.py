import pickle
import json
class PromptGenerator:
    
    def __init__(self):
        # =============Подгрузка инфо. фалов====================#
        with open('./pickles/top7.json', 'r', encoding="utf-8") as f:
            self.top7 = json.load(f)

        with open("./pickles/describe.pickle", "rb") as f:
            self.describe = pickle.load(f)

        with open("./pickles/description.pickle", "rb") as f:
            self.description = pickle.load(f)

        with open('./pickles/channels.pickle', 'rb') as f:
            self.channels = pickle.load(f)
        # ======================================================#
            

    def take_feat(self, s: dict):
        """

        :param s:
        :return: 7 пар фича - значение
        """
        cluster = s['super_clust']
        pairs = self.top7[cluster]
        return [feat[0] for feat in pairs]
    
    def check(self, actual_feat, given_feat):
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
                            if given_feat[j][1] <= self.describe[given_feat[j][0]]['mean']:
                                ans[given_feat[j][0]] = "меньше, чем этот же показатель в среднем у клиентов банка"
                            else:
                                ans[given_feat[j][0]] = "больше, чем этот же показатель в среднем у клиентов банка"

            for feat in actual_feat:
                text += ''.join(self.description[feat] + ' ' + ans[feat]) + '\n'
            return text
    
    def get_channel(self, channel: str):
        return self.channels[channel]
    
    def get_prompt(self, super_clust, channel, product, actual_feat, given_feat, comment='') -> str:
        if comment == '':
            if super_clust != 'e. Супер-аффлуент (-1)':
                promt = (f'{channel} для клиента банка. '
                            f'Порекомендуй ему {product}.'
                            f'Вот важная информации о клиенте {self.check(actual_feat, given_feat)} учти её'
                            f', но не упоминай эту информацию в сообщении клиенту.')
            else:
                promt = (f'{channel} для очень ценного клиента банка, '
                            f'у него большой счет в банке или депозит. Порекомендуй ему {product}.'
                            f'В его сообщении должен чувствоваться индивидуальный '
                            f'подход к клиенту. Вот важная информации о клиенте {self.check(actual_feat, given_feat)} учти её'
                            f', но не упоминай эту информацию в сообщении клиенту.')
        else:
            if super_clust != 'e. Супер-аффлуент (-1)':
                promt = (f'{channel} для клиента банка. '
                            f'Порекомендуй ему {product}.'
                            f'Вот важная информации о клиенте {self.check(actual_feat, given_feat)} учти её'
                            f', но не упоминай эту информацию в сообщении клиенту.'
                            f'В предидущем тексте, который выглядел так: {comment}.')
            else:
                promt = (f'{channel} для очень ценного клиента банка, '
                            f'у него большой счет в банке или депозит. Порекомендуй ему {product}.'
                            f'В его сообщении должен чувствоваться индивидуальный '
                            f'подход к клиенту. Вот важная информации о клиенте {self.check(actual_feat, given_feat)} учти её'
                            f', но не упоминай эту информацию в сообщении клиенту.'
                            f'В предидущем тексте, который выглядел так: {comment}.')

        return promt
