from fastapi import FastAPI
from pydantic import BaseModel
import json

import onnxruntime
import numpy
import pandas
import time

import base64
from PIL import Image
from io import BytesIO

import database as db
from Brain.inference import onnxPredictData as brain
from Covid19.inference import onnxPredictData as covid
from Diabetes.inference import onnxPredictData as diabetes #['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']
from HeartFailure.inference import onnxPredictData as heart
from LungCancer.inference import onnxPredictData as lung
from Tuberculosis.inference import onnxPredictData as tuber

app = FastAPI()

class Diabetes(BaseModel):
    gender : int
    age : int
    hypertension : int
    heart_disease : int
    smoking_history : int
    bmi : float
    HbA1c_level : float
    blood_glucose_level : int

@app.post("/diabetes")
async def diabetesPred(data : Diabetes):
    # data = db.getData("https://run.mocky.io/v3/b3ce155c-ba3d-4f28-a960-4a276d184efa")
    data = data.model_dump()
    
    l = []
    classes = ['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']
    for i in classes:
        l.append(data[i])
    
    if diabetes([l])[0]:
        return {"diabetes" : "Yes"}
    else:
        return {"diabetes" : "No"}

class HeartFailure(BaseModel):
    age : int
    gender : int
    chestpaintype : int
    cholesterol : int
    fastingbs : int
    maxhr : int
    exerciseangina : int
    oldpeak : int
    st_slope : int

@app.post("/heartfailure")
async def heartfailurePred(data : HeartFailure):
    # data = db.getData("https://run.mocky.io/v3/e94526be-b588-4b77-955a-783a6103478e")
    data = data.model_dump()
    
    l = []
    classes = ['age', 'gender', 'chestpaintype', 'cholesterol', 'fastingbs', 'maxhr', 'exerciseangina', 'oldpeak', 'st_slope']
    for i in classes:
        l.append(data[i])
    
    if heart([l])[0]:
        return {"heart" : "Yes"}
    else:
        return {"heart" : "No"}

class Lung(BaseModel):
    gender : int
    age : int
    smoking : int
    yellow_fingers : int
    anxiety : int
    chronic_disease : int
    fatigue : int
    allergy : int
    wheezing : int
    alcohol_consuming : int
    coughing : int
    shortness_of_breath : int
    swallowing_difficulty : int
    chest_pain : int

@app.post("/lung")
async def lungPred(data : Lung):
    # data = db.getData("https://run.mocky.io/v3/e39f87ae-f779-4651-8275-f3e2905b1719")
    data = data.model_dump()
    
    l = []
    classes = ["gender", "age", "smoking", "yellow_fingers", "anxiety", "chronic_disease", "fatigue", "allergy", "wheezing", "alcohol_consuming", "coughing", "shortness_of_breath", "swallowing_difficulty", "chest_pain"]
    for i in classes:
        l.append(data[i])
    
    if lung([l])[0]:
        return {"lung" : "Yes"}
    else:
        return {"lung" : "No"}

class ImageInput(BaseModel):
    img : str

@app.post("/covid")
async def covidPred(data : ImageInput):
    data = data.model_dump()
    data = Image.open(BytesIO(base64.b64decode(data["img"]))).convert("RGB")
    return {"covid" : covid(data)}

@app.post("/tuberculosis")
async def tuberPred(data : ImageInput):
    data = data.model_dump()
    data = Image.open(BytesIO(base64.b64decode(data["img"]))).convert("RGB")
    return {"tuberculosis" : tuber(data)}

@app.post("/brain")
async def brainPred(data : ImageInput):
    data = data.model_dump()
    data = Image.open(BytesIO(base64.b64decode(data["img"]))).convert("RGB")
    return {"brain" : brain(data)}

class Basic(BaseModel):
    userid : str
    name : str
    gender : int
    age : int
    height : int
    weight : int

@app.post("/basic")
async def basic(data : Basic):
    data = data.model_dump()
    # print(data)
    db.insertUser(data)
    # time.sleep(5)
    # print(db.getUserData(data["userid"])[0])
    # tosend = db.getUserData(data["userid"])[0]
    # tosend.pop("_id")
    # print(tosend)
    return {"response" : 200}


@app.post("/getuserdata/{userid}")
async def sendUserData(userid : str):
    tosend = db.getUserData(userid)[0]
    tosend.pop("_id")
    return {"data" : tosend}