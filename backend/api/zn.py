import os
import shutil

import cv2
from fastapi import APIRouter, UploadFile, File
from starlette.responses import JSONResponse
from starlette.status import HTTP_400_BAD_REQUEST

from utils.global_utils import clear_extract_dir
from utils.zn_utils import ZnParser

zn_router = APIRouter(
    prefix="/zn",
    tags=["Zn"],
)


@zn_router.post("/file")
async def zn_file(file: UploadFile = File(...)):
    if file.filename.split('.')[1] == 'zip':
        with open(f"uploaded_files/{file.filename}", "wb") as f:
            shutil.copyfileobj(file.file, f)
        shutil.unpack_archive(f"uploaded_files/{file.filename}", "extract")
        response = {}
        for file in os.listdir("extract"):
            image = cv2.resize(cv2.imread(f"extract/{file}", 0), (0, 0), fx=0.8, fy=0.8)
            vkk = ZnParser(image)
            answer = vkk.run_pic()
            response[file] = answer
        clear_extract_dir()
        return JSONResponse(response)
    else:
        clear_extract_dir()
        return JSONResponse({'status': 'bad_file'}, status_code=HTTP_400_BAD_REQUEST)
