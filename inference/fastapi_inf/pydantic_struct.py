from pydantic import BaseModel
from typing import List


class Client(BaseModel):
    id: int
    gender: int | None
    age: float | None
    reg_region_nm: str | None
    cnt_tr_all_3m: int | None
    cnt_tr_top_up_3m: int | None
    cnt_tr_cash_3m: int | None
    cnt_tr_buy_3m: int | None
    cnt_tr_mobile_3m: int | None
    cnt_tr_oil_3m: int | None
    cnt_tr_on_card_3m: int | None
    cnt_tr_service_3m: int | None
    cnt_zp_12m: int | None
    sum_zp_12m: int | None
    limit_exchange_count: int | None
    max_outstanding_amount_6m: float | None
    avg_outstanding_amount_3m: float | None
    cnt_dep_act: int | None
    sum_dep_now: float | None
    avg_dep_avg_balance_1month: float | None
    max_dep_avg_balance_3month: float | None
    app_vehicle_ind: int | None
    app_position_type_nm: str | None
    visit_purposes: str | None
    qnt_months_from_last_visit: int | None
    super_clust: str | None
    dataset_id: int


class Product(BaseModel):
    id: int
    name: str | None
    description: str | None


class Channel(BaseModel):
    id: int
    name: str | None
    description: str | None


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
    comment: str | None
    args: Args


# _________________________________________________

# единица ответа для юзера
class ClientTextResponse(BaseModel):
    text: str
    client_id: int
    product_id: int
    channel_id: int

