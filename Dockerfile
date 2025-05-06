FROM python:3.11.11-slim
RUN apt-get update && apt-get install -y procps git
LABEL authors="wonters"
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .

ENTRYPOINT ["", "-b"]