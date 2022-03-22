FROM pytorchlightning/pytorch_lightning:base-cuda-py3.9-torch1.10

# set workdir
WORKDIR /

# install zip and download_movielens
RUN apt-get update -y
RUN apt-get install -y zip unzip 


# download requirements
COPY requirements.txt /requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install pathlib==1.0.1


# pytorch version align for gpu
RUN pip install torch==1.7.1+cu110 \
torchvision==0.8.2+cu110 torchtext==0.8.1 \
-f https://download.pytorch.org/whl/torch_stable.html 