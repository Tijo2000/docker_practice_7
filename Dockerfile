FROM ubuntu:latest

RUN  apt-get update -y
RUN  apt-get install python3 -y \
     && apt-get install python3-pip -y

RUN pip install matplotlib
RUN pip install protobuf==3.20.0
RUN apt-get install libglib2.0-0 -y
RUN pip install opencv-python==4.5.4.60
RUN pip install opencv-contrib-python==4.5.4.60
RUN apt-get install libgl1 -y
RUN pip install tensorflow==2.8.0


WORKDIR /usr/app/src

RUN mkdir -p /object_detection/protos
RUN mkdir -p /object_detection/utils
RUN mkdir -p /saved_model/variables

RUN mkdir test

COPY inference.py ./
COPY label_map.pbtxt ./
COPY /saved_model/saved_model.pb ./saved_model/
COPY /saved_model/variables/* ./saved_model/variables/
COPY /object_detection/protos/* ./object_detection/protos/
COPY /object_detection/utils/* ./object_detection/utils/
COPY /test/* ./test/


#CMD [ "python3", "./inference.py" ]