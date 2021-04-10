from app import app
import urllib

# secret key for user session
app.secret_key = "ITSASECRET"

#setting up mail
app.config['MAIL_SERVER']='smtp.gmail.com' #mail server
app.config['MAIL_PORT'] = 587 #mail port
app.config['MAIL_USERNAME'] = 'nguyencongthinh1999@gmail.com' #email
app.config['MAIL_PASSWORD'] ='hydrogen1999vuxuchu' #password
app.config['MAIL_USE_TLS'] = True #security type
app.config['MAIL_USE_SSL'] = False #security type

#database connection parameters
# connection_params = {
#     'user': 'hydrogen1999',
#     'password': '25121999thinh',
#     'host': 'dss.kvv08.mongodb.net',
#     'port': '27017',
#     'namespace': 'dss',
# }
