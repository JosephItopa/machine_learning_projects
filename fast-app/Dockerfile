# Load a python image
FROM python:3

# save dependencies
RUN pip freeze > requirements.txt

# copy solution from local to docker
COPY . /src

# 
COPY ./requirements.txt /src/requirements.txt

# Specify working directory
WORKDIR /src

EXPOSE 8000:8000
# install dependencies on container
RUN pip install -r requirements.txt

# ue entry point above to run the app
CMD ["python", "fast-app.py"]