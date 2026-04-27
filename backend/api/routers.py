from api.tg import tg_router
from api.vk import vk_router
from api.yt import yt_router
from api.zn import zn_router

all_routers = [
    tg_router,
    vk_router,
    zn_router,
    yt_router
]