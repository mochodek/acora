FROM ubuntu:20.10

LABEL maintainer="Miroslaw Ochodek <miroslaw.ochodek@gmail.com>"
LABEL description="This is a Docker image for ACORA."

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get -yqq update \
    && apt-get -yqq update \
    && apt-get -yqq upgrade \
    && apt-get install -yqq \
        git \
        wget \
        libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /root/

RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda3-4.5.4-Linux-x86_64.sh -O ~/miniconda.sh \
    && /bin/bash ~/miniconda.sh -b -p /opt/conda \
    && rm ~/miniconda.sh

ENV PATH /opt/conda/bin:$PATH

COPY conda-env.yml .
RUN conda env create -v -f ./conda-env.yml \
    && rm ./conda-env.yml

SHELL ["/bin/bash", "--login", "-c"]

RUN echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc \
    && echo "conda activate" >> ~/.bashrc \
    && ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh 

RUN  conda activate acora \
    && python -m ipykernel install --user --name acora --display-name "Python ACORA" 

VOLUME [ "/root/acora", "/root/workspace" ]
COPY . "/root/acora"
ENV ACORA_HOME "/root/acora"

WORKDIR /root/acora
RUN conda activate acora \
    && pip install -e .

EXPOSE 8888 

WORKDIR /root/workspace


