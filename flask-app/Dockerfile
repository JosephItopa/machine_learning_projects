# specify base image
FROM python:3

# Specify working directory
WORKDIR ./

# copy files from local system to container
ADD ./app app

# extract & save all needed libraries
RUN pip freeze > requirements.txt

# install libraries
RUN pip install -r requirements.txt

# Define which app should run when container starts
ENTRYPOINT ["PYTHON"]

# start flask-app from command prompt
CMD ["python", "app/flask-app.py"]