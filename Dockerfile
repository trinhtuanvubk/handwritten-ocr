# FROM paddlepaddle/paddle:2.4.1-gpu-cuda11.7-cudnn8.4-trt8.4
FROM nvidia/cuda:11.3.1-runtime-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive


RUN apt-get update -y
RUN apt-get install python3.8 -y
RUN apt-get install python3-pip -y
RUN python3.8 -m pip install --upgrade pip
RUN apt-get install 'ffmpeg'\
    'libsm6'\
    'libxext6'  -y
RUN apt-get install python3.8-dev -y

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
WORKDIR /app-src
COPY requirements.txt /app-src/

RUN python3.8 -m pip install -r requirements.txt