import os
import shutil

import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File
from starlette.responses import JSONResponse
from starlette.status import HTTP_400_BAD_REQUEST

from utils.global_utils import clear_extract_dir
from utils.youtube.youtube_sub_parser import YoutubeSubscriberParser
from utils.youtube.youtube_views_parser import YouTubeViewsParser
from config import reader

yt_router = APIRouter(
    prefix="/yt",
    tags=["Yt"],
)


@yt_router.post("/file")
async def yt_file(file: UploadFile = File(...)):
    if file.filename.split('.')[1] == 'zip':
        with open(f"uploaded_files/{file.filename}", "wb") as f:
            shutil.copyfileobj(file.file, f)
        shutil.unpack_archive(f"uploaded_files/{file.filename}", "extract")
        response = {}
        for file in os.listdir("extract"):
            image = cv2.resize(cv2.imread(f"extract/{file}", 0), (0, 0), fx=0.8, fy=0.8)
            ytt1 = YoutubeSubscriberParser(image)
            answer = ytt1.run_pic()
            ytt2 = YouTubeViewsParser(image)
            model = ytt2.model
            answer2 = np.array([" ".join(reader.readtext(image, detail=0, paragraph=True))])
            answer2 = model.predict(answer2)
            an = {"Подписчики": answer, "Просмотры": answer2}
            response[file] = an
        clear_extract_dir()
        return JSONResponse(response)
    else:
        clear_extract_dir()
        return JSONResponse({'status': 'bad_file'}, status_code=HTTP_400_BAD_REQUEST)
