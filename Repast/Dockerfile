FROM ubuntu:22.04

ENV PYTHON_VERSION=3.10.12

#Set of all dependencies needed for pyenv to work on Ubuntu
RUN apt-get update && apt-get -y install locales 

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y expect

RUN apt-get install -y --no-install-recommends make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget ca-certificates curl llvm libncurses5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev mecab-ipadic-utf8 git

# Set-up necessary Env vars for PyEnv
#ENV PYENV_ROOT=/root/.pyenv
#ENV PATH=$PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH

# Install pyenv
#RUN set -ex \
#    && curl https://pyenv.run | bash \
#    && pyenv update \
#    && pyenv install $PYTHON_VERSION \
#    && pyenv global $PYTHON_VERSION \
#    && pyenv rehash


#RUN yes | apt-get install software-properties-common
#RUN apt-get update
#RUN yes | add-apt-repository ppa:deadsnakes/ppa
#RUN yes | apt install python3.11

RUN apt-get install -y python3
RUN apt-get install -y python-is-python3
RUN apt-get install -y python3-pip

RUN yes | apt install mpich
#RUN yes | apt-get install build-essential linux-generic libmpich-dev libopenmpi-dev

#RUN yes | pip3 install --user mpi4py

#RUN yes | pip install --upgrade pip

RUN env CC=mpicxx pip install repast4py

RUN yes | apt install vim
RUN yes | pip3 install scipy==1.12.0
RUN yes | pip3 install numba==0.60.0
RUN yes | pip3 install numpy==1.23.5
RUN yes | pip3 install osmnx==1.9.1
RUN yes | pip3 install matplotlib==3.8.2
RUN yes | pip3 install dm-haiku==0.0.11
RUN yes | pip3 install tqdm==4.66.1
RUN yes | pip3 install SALib==1.5.0
RUN yes | pip3 install seaborn==0.13.2
RUN yes | pip3 install pandas==2.2.2
RUN yes | pip3 install pyshp==2.3.1

# Optional : Checks Pyenv version on container start-up
# ENTRYPOINT [ "pyenv","version" ]

CMD /bin/bash
