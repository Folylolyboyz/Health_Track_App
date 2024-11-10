from fastapi import FastAPI, Response
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import httpx
from contextlib import asynccontextmanager
import os

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

# app = FastAPI()

@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(cyclic_func())
    yield
    task.cancel()

async def cyclic_func():
    while True:
        try:
            async with httpx.AsyncClient() as client:
                await client.get(os.environ["siteurl"])
                await asyncio.sleep(300)  # Every 5 minutes
        except Exception as e:
            print(f"Error in cyclic_func: {e}")
            await asyncio.sleep(60)  # Wait for 1 minute

app = FastAPI(lifespan=lifespan)

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    # allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

@app.get("/")
def homepage():
    return {"response" : "Hello World"}

class Diabetes(BaseModel):
    userid : str
    heart_disease : str #int
    HbA1c_level : str #float
    blood_glucose_level : str #int

@app.post("/model/diabetes")
def diabetesPred(data : Diabetes):
    # data = db.getData("https://run.mocky.io/v3/b3ce155c-ba3d-4f28-a960-4a276d184efa")
    data = data.model_dump()
    userid = data.pop("userid")
    userdata = db.getUserData(userid)
    alldata = {**data, **userdata}
    
    
    sys, dia = map(int, userdata["bloodpressure"].split("/"))
    if sys >= 140 or dia >=90:
        alldata["hypertension"] = 1
    else:
        alldata["hypertension"] = 0
    
    
    floatclasses = ["heart_disease", "HbA1c_level", "blood_glucose_level"]
    for i in floatclasses:
        try:
            data[i] = float(data[i])
        except:
            return {"error" : 400}
    
    
    l = []
    classes = ['gender', 'age', 'hypertension', 'heart_disease', 'smoking', 'bmi', 'HbA1c_level', 'blood_glucose_level']
    for i in classes:
        l.append(alldata[i])
    
    ans = diabetes([l])[0]
    del data, userid, userdata, alldata, floatclasses, l, classes, i
    if ans:
        return {"diabetes" : "Yes"}
    else:
        return {"diabetes" : "No"}

class HeartFailure(BaseModel):
    userid : str
    chestpaintype : str #int
    cholesterol : str #int
    fastingbs : str #int
    maxhr : str #int
    exerciseangina : str #int
    oldpeak : str #int
    st_slope : str #int

@app.post("/model/heartfailure")
def heartfailurePred(data : HeartFailure):
    # data = db.getData("https://run.mocky.io/v3/e94526be-b588-4b77-955a-783a6103478e")
    data = data.model_dump()
    userid = data.pop("userid")
    userdata = db.getUserData(userid)
    alldata = {**data, **userdata}
    
    
    intclasses = ["chestpaintype", "cholesterol", "fastingbs", "maxhr", "exerciseangina", "oldpeak", "st_slope"]
    for i in intclasses:
        try:
            data[i] = int(data[i])
        except:
            return {"error" : 400}
    
    
    l = []
    classes = ['age', 'gender', 'chestpaintype', 'cholesterol', 'fastingbs', 'maxhr', 'exerciseangina', 'oldpeak', 'st_slope']
    for i in classes:
        l.append(alldata[i])
    
    ans = heart([l])[0]
    del data, userid, userdata, alldata, intclasses, l, classes, i
    if ans:
        return {"heart" : "Yes"}
    else:
        return {"heart" : "No"}

class Lung(BaseModel):
    userid : str
    yellow_fingers : str #int
    anxiety : str #int
    chronic_disease : str #int
    fatigue : str #int
    wheezing : str #int
    coughing : str #int
    shortness_of_breath : str #int
    swallowing_difficulty : str #int
    chest_pain : str #int

@app.post("/model/lung")
def lungPred(data : Lung):
    # data = db.getData("https://run.mocky.io/v3/e39f87ae-f779-4651-8275-f3e2905b1719")
    data = data.model_dump()
    userid = data.pop("userid")
    userdata = db.getUserData(userid)
    alldata = {**data, **userdata}
    
    
    intclasses = ["yellow_fingers", "anxiety", "chronic_disease", "fatigue", "wheezing", "coughing", "shortness_of_breath", "swallowing_difficulty", "chest_pain"]
    for i in intclasses:
        try:
            data[i] = int(data[i])
        except:
            return {"error" : 400}
    
    
    l = []
    classes = ["gender", "age", "smoking", "yellow_fingers", "anxiety", "chronic_disease", "fatigue", "allergy", "wheezing", "alcohol", "coughing", "shortness_of_breath", "swallowing_difficulty", "chest_pain"]
    for i in classes:
        l.append(alldata[i])
    
    
    ans = lung([l])[0]
    del data, userid, userdata, alldata, intclasses, l, classes, i
    if ans:
        return {"lung" : ans}
    else:
        return {"lung" : "No"}

class ImageInput(BaseModel):
    img : str

@app.post("/model/lungimage")
def covidPred(data : ImageInput):
    data = data.model_dump()
    data = Image.open(BytesIO(base64.b64decode(data["img"]))).convert("RGB")
    ans1 = covid(data)
    ans2 = tuber(data)
    del data
    return {"covid" : ans1, "tuber" : ans2}

@app.post("/model/covid")
def covidPred(data : ImageInput):
    data = data.model_dump()
    data = Image.open(BytesIO(base64.b64decode(data["img"]))).convert("RGB")
    ans = covid(data)
    del data
    return {"covid" : ans}

@app.post("/model/tuberculosis")
def tuberPred(data : ImageInput):
    data = data.model_dump()
    data = Image.open(BytesIO(base64.b64decode(data["img"]))).convert("RGB")
    ans = tuber(data)
    del data
    return {"tuberculosis" : ans}

@app.post("/model/brain")
def brainPred(data : ImageInput):
    data = data.model_dump()
    data = Image.open(BytesIO(base64.b64decode(data["img"]))).convert("RGB")
    ans = brain(data)
    del data
    return {"brain" : ans}

class Basic(BaseModel):
    userid : str
    name : str
    gender : str #int
    age : str #int
    height : str #int
    weight : str #int

@app.post("/basic")
def basic(data : Basic):
    data = data.model_dump()
    # print(data)
    intclasses = ["gender", "age", "height", "weight"]
    for i in intclasses:
        try:
            data[i] = int(data[i])
        except:
            return Response(status_code=204)
    db.insertUser(data)
    del data
    # time.sleep(5)
    # print(db.getUserData(data["userid"])[0])
    # tosend = db.getUserData(data["userid"])[0]
    # tosend.pop("_id")
    # print(tosend)
    return {"response" : 200}

class Health(BaseModel):
    userid : str
    bloodpressure : str
    allergy : str #int
    smoking : str #int
    alcohol : str #int

@app.post("/health")
def health(data : Health):
    data = data.model_dump()
    
    intclasses = ["allergy", "smoking", "alcohol"]
    for i in intclasses:
        try:
            data[i] = int(data[i])
        except:
            return Response(status_code=204)
    db.insertUser(data)
    del data
    return {"response" : 200}

@app.post("/getuserdata/{userid}")
def sendUserData(userid : str):
    tosend = db.getUserData(userid)
    if tosend:
        tosend.pop("_id")
        # return {"data" : tosend}
        return tosend
    else:
        Response(status_code=204)

@app.get("/getuserdata/{userid}")
def sendUserData(userid : str):
    tosend = db.getUserData(userid)
    if tosend:
        tosend.pop("_id")
        # return {"data" : tosend}
        return tosend
    else:
        Response(status_code=204)