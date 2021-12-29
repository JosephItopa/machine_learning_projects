"""
Ensure your create an app on heroku and then follow the command below:

0) pip freeze > requirements.txt
1) login heroku login -i
2) initiate git: git init
3) heroku git:remote -a <app name>
4) git add .
5) git commit -am "flask-app"
6) git push heroku master 
7) Test your api on postman
"""
from app.flaskapp import app

if __name__ == "__main__":
    app.run()
