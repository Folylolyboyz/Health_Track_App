from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import pymongo.synchronous.collection as mongoDatatypes
import os

def client() -> mongoDatatypes.Collection:
    uri = os.environ["uri"]

    # Create a new client and connect to the server
    client = MongoClient(uri, server_api=ServerApi('1'))

    # Send a ping to confirm a successful connection
    # try:
    #     client.admin.command('ping')
    #     print("Pinged your deployment. You successfully connected to MongoDB!")
    # except Exception as e:
    #     print(e)

    db = client[os.environ["database"]]
    cl = db[os.environ["collection"]]
    return cl