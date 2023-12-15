from pydantic import BaseModel
class DatasetId(BaseModel):
    dataset_id: int
    dataset_name: str
    dataset_comment: str
class ClientInfo(BaseModel):
    gender: int
    age: float
    reg_region_nm: str
    cnt_tr_all_3m: int
    cnt_tr_top_up_3m: int
    cnt_tr_cash_3m: int
    cnt_tr_buy_3m: int
    cnt_tr_mobile_3m: int
    cnt_tr_oil_3m: int
    cnt_tr_on_card_3m: int
    cnt_tr_service_3m: int
    cnt_zp_12m: int
    sum_zp_12m: int
    limit_exchange_count: int
    max_outstanding_amount_6m: float
    avg_outstanding_amount_3m: float
    cnt_dep_act: int
    sum_dep_now: float
    avg_dep_avg_balance_1month: float
    max_dep_avg_balance_3month: float
    app_vehicle_ind: int
    app_position_type_nm: str
    visit_purposes: str
    qnt_months_from_last_visit: int
    super_clust: str
class Product(BaseModel):
    product_id: int
    product_name: str
    product_description: str

class Chanel(BaseModel):
    chanel_id: int
    chanel_name: str
    chanel_description: str

class TextInfo(BaseModel):
    text_id: int
    note: str
    product: Product
    chanel: Chanel
    is_good: int
    temperature: float
    top_p: float

class Client(BaseModel):
    vid: str
    dataset_id: DatasetId
    client_info: ClientInfo
    text_info: list[TextInfo]

# дата, которая приходит
class DatasetClients(BaseModel):
    clients: list[Client]
#_________________________________________________
    
 #единица ответа для юзера
class ClientTextRequest(BaseModel):
    text_id: int
    text: str
    note: str
    product: Product
    chanel: Chanel
    is_good: int
    temperature: float
    top_p: float

class ClientRequest(BaseModel):
    vid: str
    dataset_id: DatasetId
    text_info: list[ClientTextRequest]

#собираем датасет
class Request(BaseModel):
    UserRequest: list[ClientRequest]