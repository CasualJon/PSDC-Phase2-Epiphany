FROM python:3.10.12-bullseye

## DO NOT EDIT these 3 lines.
RUN mkdir /challenge
COPY ./ /challenge
WORKDIR /challenge

## Install your dependencies here using apt install, etc.
RUN apt-get update && apt-get install --no-install-recommends -y \
    build-essential \
    libopenblas-dev \
    libgomp1 \
    && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/*

## Upgrade pip
RUN pip install --upgrade pip

## Include the following line if you have a requirements.txt file.
RUN pip install --no-cache-dir -r requirements.txt
