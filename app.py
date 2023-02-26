from flask import Flask, render_template, request
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
from matplotlib.pyplot import imread
from matplotlib.pyplot import imshow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input,decode_predictions
import uvicorn
from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import shutil
from fastapi.responses import JSONResponse
from mangum import Mangum


app = FastAPI()
handler = Mangum(app)

loaded_model_imageNet=load_model("vishal_model_resnet50.h5")


app.mount("/static", StaticFiles(directory="static"), name="static")


# loaded_model_imageNet=load_model("vishal_model_resnet50.h5")

# UPLOAD_FOLDER = "C:/Users/hp/Desktop/Flask_Project_Melanoma/static/images"
templates = Jinja2Templates(directory="templates")
UPLOAD_FOLDER = "static/images/"


@app.get('/', response_class=HTMLResponse)
async def melanoma_index(request: Request):
    print(request.method)
    return templates.TemplateResponse("index.html", {"request": request})


@app.get('/home',  response_class=HTMLResponse)
async def melanoma_home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})



@app.get('/about', response_class=HTMLResponse)
async def melanoma_about(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})


@app.get('/solution', response_class=HTMLResponse)
async def hello_word(request: Request):
    return templates.TemplateResponse("solution.html",{"request": request,"btnshow":True, "imageExist":True})


@app.post('/solution')
def upload_predict(request: Request, patientImage:UploadFile = File(...)):
    print(patientImage.filename)
    UploadedFile = UPLOAD_FOLDER + patientImage.filename
    if request.method == "POST":
        image_file = patientImage.filename

        with open(UploadedFile, "wb") as buffer:
         shutil.copyfileobj(patientImage.file, buffer)


        if image_file:
            image_location = os.path.join(UPLOAD_FOLDER, image_file)
            # image_file.save(image_location)
            # ------------------New Code-----------------------------
            img_path=image_location
            img=cv2.imread(img_path)
            img=cv2.resize(img,(100,100))
            x=np.expand_dims(img,axis=0)
            x=preprocess_input(x)
            result=loaded_model_imageNet.predict(x)
            p=list((result*100).astype('int'))
            pp=list(p[0])
            ss = max(pp)
            index=pp.index(max(pp))
            name_class=['Benign','Malignant']
            Final_Result = name_class[index]
            # ------------------New Code End-------------------------
    return templates.TemplateResponse("solution.html",{"request": request, "result" : Final_Result, "imageName" : patientImage.filename, "imageExist" : False, "btnshow" : False})






if __name__ == '__main__':
    uvicorn.run(app,host="0.0.0.0",port=9000)

