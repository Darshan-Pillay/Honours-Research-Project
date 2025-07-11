# syntax=docker/dockerfile:1
FROM ubuntu:latest

RUN apt-get update
RUN apt-get install -y python3 pip

RUN pip install Flask --break-system-packages

RUN mkdir test
WORKDIR /test
COPY hello_world.py /test