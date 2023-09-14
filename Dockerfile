#Run an official pythonruntime as the base image
FROM python:3.11

#Set the working directory in the container
WORKDIR /app

#Copy the requirements file into the container
COPY requirements.txt .

#Install the python dependencies
RUN pip install --no-cache-dir -r requirements.txt

#copy the model file into the container
COPY boost.pkl .

#copy the python script into the container
COPY main.py .

#Run the python script when the container launches
ENTRYPOINT ["python", "main.py"]