FROM ubuntu:18.04 

RUN apt-get update 

RUN apt-get install -y wget 

RUN wget https://github.com/Samsung/ONE/releases/download/1.21.0/one-compiler_1.21.0_amd64.deb 

RUN apt install -y ./one-compiler_1.21.0_amd64.deb 

WORKDIR /home/ 

RUN mkdir test 

WORKDIR /home/test 

CMD ["onecc", "-C", "onecc.template.cfg"]
