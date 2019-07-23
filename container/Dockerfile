FROM nvidia/cuda:9.2-cudnn7-devel-ubuntu18.04

RUN apt-get install python3-dev python3-pip git

RUN cd /root && git clone https://github.com/tloki/CVPR19-Face-Anti-spoofing

WORKDIR /root/CVPR19-Face-Anti-spoofing

CMD['/usr/bin/bash']