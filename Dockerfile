FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

RUN apt-get update
RUN apt-get -y install ffmpeg tmux pkg-config
RUN apt-get install -y \
    libavformat-dev libavcodec-dev libavdevice-dev \
    libavutil-dev libswscale-dev libavresample-dev libavfilter-dev libgtk2.0-dev
RUN apt-get -y install build-essential python3 python3-dev python3-pip python3-venv
RUN python3 -m pip install pip --upgrade

COPY requirements.txt .
RUN pip install -r requirements.txt
