import uvicorn as uvicorn
from fastapi import FastAPI
import shutil
from fastapi.middleware.cors import CORSMiddleware

from api.routers import all_routers

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


for router in all_routers:
    app.include_router(router)


if __name__ == '__main__':
    uvicorn.run(app)
