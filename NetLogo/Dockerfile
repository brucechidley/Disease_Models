FROM ubuntu:22.04

ENV PYTHON_VERSION=3.11.9

ARG JAVA_VERSION='jdk-21.0.2'
ARG INSTALL_DIR='/usr/lib/jdk'
ARG JAVA_FILENAME=${JAVA_VERSION}_linux-x64_bin.tar.gz
ENV JAVA_VERSION=$JAVA_VERSION
ENV JAVA_HOME="$INSTALL_DIR/$JAVA_VERSION"
ENV PATH="$PATH:$INSTALL_DIR/$JAVA_VERSION/bin"

#Set of all dependencies needed for pyenv to work on Ubuntu
RUN apt-get update && apt-get -y install locales 

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y expect

RUN apt-get install -y --no-install-recommends make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget ca-certificates curl llvm libncurses5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev mecab-ipadic-utf8 git

RUN apt install wget
RUN wget https://download.oracle.com/java/21/archive/jdk-21.0.2_linux-x64_bin.tar.gz

#COPY "$JAVA_FILENAME" /
RUN set -x && \
    mkdir -vp "$INSTALL_DIR" && \
    tar xzvf "/$JAVA_FILENAME" -C "$INSTALL_DIR" && \
    chmod -vR a+x "$INSTALL_DIR/$JAVA_VERSION/bin" && \
    apt-get update -y && \
    rm -rfv "/$JAVA_FILENAME" /var/lib/apt/lists/*

    
RUN apt-get update && apt-get -y install locales 

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y expect

RUN apt-get install -y --no-install-recommends make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget ca-certificates curl llvm libncurses5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev mecab-ipadic-utf8 git


RUN apt-get install -y python3
RUN apt-get install -y python3-pip

#RUN yes | apt install vim

RUN yes | pip3 install numpy==1.26.4
RUN yes | pip3 install osmnx==1.9.1
RUN yes | pip3 install matplotlib==3.8.2
RUN yes | pip3 install SALib==1.5.0
RUN yes | pip3 install seaborn==0.13.2
RUN yes | pip3 install pandas==2.2.2
RUN yes | pip3 install pyshp==2.3.1
RUN yes | pip3 install scipy==1.12.0
RUN yes | pip3 install pynetlogo==0.5.2

# Optional : Checks Pyenv version on container start-up
# ENTRYPOINT [ "pyenv","version" ]

CMD /bin/bash
