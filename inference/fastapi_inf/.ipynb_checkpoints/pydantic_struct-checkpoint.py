from pydantic import BaseModel
from typing import List, Union


class Client(BaseModel):
    id: Union[int, None]
    gender: Union[int, None]
    age: Union[float, None]
    reg_region_nm: Union[str, None]
    cnt_tr_all_3m: Union[int, None]
    cnt_tr_top_up_3m: Union[int, None]
    cnt_tr_cash_3m: Union[int, None]
    cnt_tr_buy_3m: Union[int, None]
    cnt_tr_mobile_3m: Union[int, None]
    cnt_tr_oil_3m: Union[int, None]
    cnt_tr_on_card_3m: Union[int, None]
    cnt_tr_service_3m: Union[int, None]
    cnt_zp_12m: Union[int, None]
    sum_zp_12m: Union[int, None]
    limit_exchange_count: Union[int, None]
    max_outstanding_amount_6m: Union[float, None]
    avg_outstanding_amount_3m: Union[float, None]
    cnt_dep_act: Union[int, None]
    sum_dep_now: Union[float, None]
    avg_dep_avg_balance_1month: Union[float, None]
    max_dep_avg_balance_3month: Union[float, None]
    app_vehicle_ind: Union[int, None]
    app_position_type_nm: Union[str, None]
    visit_purposes: Union[str, None]
    qnt_months_from_last_visit: Union[int, None]
    super_clust: Union[str, None]
    dataset_id: Union[int, None]


class Product(BaseModel):
    id: int
    name: Union[str, None]
    description: Union[str, None]


class Channel(BaseModel):
    id: int
    name: Union[str, None]
    description: Union[str, None]


class Args(BaseModel):
    temp: float
    top_p: float


# данные, которые приходят
class DatasetClients(BaseModel):
    clients: List[Client]
    products: List[Product]
    channels: List[Channel]
    args: Args


class ClientRegen(BaseModel):
    client: Client
    product: Product
    channel: Channel
    text: str
    comment: Union[str, None]
    args: Args


# единица ответа для юзера
class ClientTextResponse(BaseModel):
    text: str
    client_id: int
    product_id: int
    channel_id: int