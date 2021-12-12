# Load a python image
FROM python:3

# save dependencies
RUN pip freeze > requirements.txt

# copy solution from local to docker
COPY . /src

# Specify working directory
WORKDIR /src

# copy dependencies
COPY ./requirements.txt /src/requirements.txt

# install dependencies on container
RUN pip install -r requirements.txt

EXPOSE 5000:5000

# ue entry point above to run the app
CMD ["python", "fast-app.py"]