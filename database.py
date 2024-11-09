import requests
import pymongo.synchronous.collection as mongoDatatypes

from databaseConnection import client

cl = client()

# columns = ["_id", 
#            "name", 
#            "age", 
#            "gender", 
#            "height", 
#            "weight", 
#            "bmi", 
#            "smoking", 
#            "alcohol", 
#            "allergy"]

def insertUser(data:dict):
    if userExists(data["userid"]):
        print("User exists. Try updating.")
        updateUser(data)
        return
    data["bmi"] = data["weight"] // ((data["height"]/100)**2)
    cl.insert_one(data)
    print("User inserted")

def updateUser(data:dict):
    if not userExists(data["userid"]):
        print("User does not exists. Try inserting.")
        return
    cl.update_one({"userid" : data["userid"]}, {"$set" : data})
    print("User updated")

def getData(api: str) -> dict:
    data = dict(requests.get(api).json())
    return data

def userExists(userid: int) -> bool:
    data = list(cl.find({"userid" : userid}))
    if data:
        return True
    return False

def getUserData(userid: int):
    data = list(cl.find({"userid" : userid}))
    return data[0]

# data = getData("https://run.mocky.io/v3/c3796c22-2916-4bc6-a57a-a62caf18fd2f")
# updateUser(cl, data)