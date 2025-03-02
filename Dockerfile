FROM python:3.13-alpine

RUN apt-get update && apt-get install -y git

COPY requirements.txt /

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

COPY ./src /app/src

ENV PYTHONUNBUFFERED=1

WORKDIR /app

CMD [ "sh" ]