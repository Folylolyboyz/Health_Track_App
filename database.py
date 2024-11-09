from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import pymongo.synchronous.collection as mongoDatatypes
import requests

uri = "mongodb+srv://HealthTrackApp:HealthTrackApp@cluster0.ndcvj.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))

# Send a ping to confirm a successful connection
# try:
#     client.admin.command('ping')
#     print("Pinged your deployment. You successfully connected to MongoDB!")
# except Exception as e:
#     print(e)

db = client["HealthTrackAppDatabase"]
cl = db["HealthTrackAppCollection"]

columns = ["_id", 
           "name", 
           "gender", 
           "height", 
           "weight", 
           "bmi", 
           "smoking", 
           "alcohol", 
           "allergy"]

def insertUser(cl: mongoDatatypes.Collection, data:dict):
    if userExists(cl, data["_id"]):
        print("User exists. Try updating.")
        return
    data["bmi"] = data["weight"] // ((data["height"]/100)**2)
    cl.insert_one(data)

def updateUser(cl: mongoDatatypes.Collection, data:dict):
    if not userExists(cl, data["_id"]):
        print("User does not exists. Try inserting.")
        return
    cl.update_one({"_id" : data["_id"]}, {"$set" : data})


def getData(api: str) -> dict:
    data = dict(requests.get(api).json())
    return data

def userExists(cl: mongoDatatypes.Collection, id: int) -> bool:
    data = list(cl.find({"_id" : id}))
    if data:
        return True
    return False

data = getData("https://run.mocky.io/v3/5e12de4f-77dd-4b04-abcc-3dae527e08f7")
insertUser(cl, data)