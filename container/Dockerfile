FROM nvidia/cuda:9.2-cudnn7-devel-ubuntu18.04

WORKDIR /root

RUN apt-get update && apt-get install python3-dev python3-pip git libsm6 libxext6 libxrender-dev wget -y

RUN cd /root && git clone https://github.com/tloki/CVPR19-Face-Anti-spoofing

WORKDIR /root/CVPR19-Face-Anti-spoofing

RUN pip3 install -r /root/CVPR19-Face-Anti-spoofing/requirements.txt

CMD ['/bin/bash']