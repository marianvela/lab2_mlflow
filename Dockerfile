# python version
FROM python:3.9

# install requirements
COPY requirements.txt .
RUN apt-get update && \
    apt-get install -y git && \
    pip install --no-cache-dir -r requirements.txt

# working directory
WORKDIR /app

# copy project into container
COPY . .

# expose port for API processing
EXPOSE 8000

# default commands
CMD ["python", "main.py"]
