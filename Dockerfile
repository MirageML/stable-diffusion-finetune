FROM nvidia/cuda:11.6.0-base-ubuntu18.04

## Some utilities
RUN apt-get update -y &&     apt-get install -y build-essential libfuse-dev libcurl4-openssl-dev libxml2-dev pkg-config libssl-dev mime-support automake libtool wget tar git unzip
RUN apt-get install lsb-release -y  && apt-get install zip -y && apt-get install vim -y
RUN apt-get install ffmpeg libsm6 libxext6  -y

## Install AWS CLI
RUN apt-get update && apt-get install -y python3.8 python3.8-dev
RUN alias python3=python3.8
RUN apt-get install -y         python3-pip         python3-setuptools         groff         less     && python3.8 -m pip install --upgrade pip     && apt-get clean
RUN python3.8 -m pip install --no-cache-dir install --upgrade awscli supabase

## Install S3 Fuse
RUN rm -rf /usr/src/s3fs-fuse
RUN git clone https://github.com/s3fs-fuse/s3fs-fuse/ /usr/src/s3fs-fuse
WORKDIR /usr/src/s3fs-fuse
RUN ./autogen.sh && ./configure && make && make install

## Create folder
WORKDIR /home
RUN mkdir s3bucket
RUN mkdir user-datasets

WORKDIR /home
# RUN wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh
# RUN chmod +x Anaconda3-2022.05-Linux-x86_64.sh
# RUN ./Anaconda3-2022.05-Linux-x86_64.sh
RUN git clone https://KishoreMayank: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@github.com/MirageML/stable-diffusion-finetune.git

WORKDIR /home/stable-diffusion-finetune
RUN git config --global http.postBuffer 1048576000
RUN python3.8 -m pip install -r requirements.txt
# RUN conda env create -f environment.yaml
# RUN conda activate ldm
RUN python3.8 scripts/preload_models.py
RUN git lfs install
RUN git clone https://amankishore:Testpass%40123@huggingface.co/CompVis/stable-diffusion-v-1-4-original

# RUN git clone https://KishoreMayank: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@github.com/MirageML/python-launcher.git

ENTRYPOINT []
