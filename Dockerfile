FROM pytorchlightning/pytorch_lightning:base-cuda-py3.9-torch1.10

# set workdir
WORKDIR /

# install zip and download_movielens
COPY download_movielens.sh /download_movielens.sh
RUN apt-get update -y
RUN apt-get install -y zip unzip 
RUN sh download_movielens.sh 

# download requirements
COPY requirements.txt /requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install pathlib==1.0.1


