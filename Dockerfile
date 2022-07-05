FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-runtime

RUN apt-get -y update
RUN apt-get -y install git
RUN apt-get -y install vim
RUN apt-get -y install make

# install python dependencies
RUN pip install --upgrade pip
RUN pip install torch torchvision
RUN pip install tensorflow



# create a user etc
ENV EDITOR vim
ENV PYTHONPATH=/workspace/sempar/nl2bash
# RUN useradd --create-home kyduff
# RUN usermod -aG sudo kyduff
# USER kyduff