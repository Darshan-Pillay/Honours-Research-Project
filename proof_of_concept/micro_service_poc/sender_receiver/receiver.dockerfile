# syntax=docker/dockerfile:1
FROM ubuntu:latest

RUN apt-get update
RUN apt-get install -y python3 pip
RUN apt-get install -y curl

RUN pip install Flask --break-system-packages

RUN mkdir poc
WORKDIR /poc
COPY receiver.py /poc

ENTRYPOINT ["flask", "--app", "receiver", "run", "--host=0.0.0.0"]