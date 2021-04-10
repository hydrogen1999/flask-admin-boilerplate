from pymongo import MongoClient
import configuration

CONNECTION_STRING = "mongodb+srv://hydrogen1999:25121999thinh@dss.kvv08.mongodb.net/dss?retryWrites=true&w=majority"

#connect to mongodb
mongoconnection = MongoClient(CONNECTION_STRING)

db = mongoconnection.dss