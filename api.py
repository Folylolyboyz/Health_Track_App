from fastapi import FastAPI
from pydantic import BaseModel

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
    userid : str
    heart_disease : str #int
    HbA1c_level : str #float
    blood_glucose_level : str #int

@app.post("/model/diabetes")
async def diabetesPred(data : Diabetes):
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
    
    if diabetes([l])[0]:
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
async def heartfailurePred(data : HeartFailure):
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
    
    if heart([l])[0]:
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
async def lungPred(data : Lung):
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
    
    if lung([l])[0]:
        return {"lung" : "Yes"}
    else:
        return {"lung" : "No"}

class ImageInput(BaseModel):
    img : str

@app.post("/model/covid")
async def covidPred(data : ImageInput):
    data = data.model_dump()
    data = Image.open(BytesIO(base64.b64decode(data["img"]))).convert("RGB")
    return {"covid" : covid(data)}

@app.post("/model/tuberculosis")
async def tuberPred(data : ImageInput):
    data = data.model_dump()
    data = Image.open(BytesIO(base64.b64decode(data["img"]))).convert("RGB")
    return {"tuberculosis" : tuber(data)}

@app.post("/model/brain")
async def brainPred(data : ImageInput):
    data = data.model_dump()
    data = Image.open(BytesIO(base64.b64decode(data["img"]))).convert("RGB")
    return {"brain" : brain(data)}

class Basic(BaseModel):
    userid : str
    name : str
    gender : str #int
    age : str #int
    height : str #int
    weight : str #int

@app.post("/basic")
async def basic(data : Basic):
    data = data.model_dump()
    # print(data)
    intclasses = ["gender", "age", "height", "weight"]
    for i in intclasses:
        try:
            data[i] = int(data[i])
        except:
            return {"error" : 400}
    db.insertUser(data)
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
async def health(data : Health):
    data = data.model_dump()
    
    intclasses = ["allergy", "smoking", "alcohol"]
    for i in intclasses:
        try:
            data[i] = int(data[i])
        except:
            return {"error" : 400}
    db.insertUser(data)
    
    return {"response" : 200}

@app.post("/getuserdata/{userid}")
async def sendUserData(userid : str):
    tosend = db.getUserData(userid)
    tosend.pop("_id")
    return {"data" : tosend}