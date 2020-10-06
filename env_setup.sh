#! /bin/bash

# to remote ssh tunnel into the docker, use below command:
# ssh -L 8000:localhost:8888 riversome@192.168.1.196

# to fire up docker instance from within the instance:
# sudo docker run --gpus all -it -p 8888:8888 tensorflow/tensorflow:latest-gpu-jupyter /bin/bash

# install shell apps
apt-get install tmux && apt-get install nano
# when inside docker, bash this script
pip install pandas scikit-learn seaborn
pip install spacy && python -m spacy download en_core_web_lg
pip install jupyterlab

# set up github
git config --global user.name "ElvinOuyang"
git config --global user.email "elvin.ouyang@gmail.com"
git config credential.helper store

# set up notebook password
jupyter notebook password

# execution to fire up the env
tmux new -d -s jupyter_session 'jupyter lab --notebook-dir=/tf --ip 0.0.0.0 --no-browser --allow-root'
