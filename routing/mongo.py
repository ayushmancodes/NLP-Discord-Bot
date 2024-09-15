from pymongo import MongoClient
from dotenv import load_dotenv
import os
load_dotenv()

# Create a global MongoClient instance
mongo_client = None
ID=os.getenv("MONGO_URL")
def get_client():
    global mongo_client
    if mongo_client is None:
        mongo_client = MongoClient(ID)
    return mongo_client


if __name__=="__main__":
    mongo_client = MongoClient(ID)
    db = mongo_client['discord_db']
    collection = db['toxicity_levels']
    demo_doc={"user":"#1234","toxicity_level":2}
    insert_doc=collection.insert_one(demo_doc)
    mongo_client.close()