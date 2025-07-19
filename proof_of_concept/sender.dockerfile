# syntax=docker/dockerfile:1
FROM ubuntu:latest

RUN apt-get update
RUN apt-get install -y python3 pip
RUN apt-get install -y curl

RUN pip install Flask --break-system-packages
RUN  pip install requests --break-system-packages

RUN mkdir poc
WORKDIR /poc
COPY sender.py /poc

ENTRYPOINT ["flask", "--app", "sender", "run", "--host=0.0.0.0"]