FROM alpine/git as git-env
WORKDIR /app
RUN git clone https://github.com/waleedka/coco && git clone https://github.com/niconiconi1234/Mask_RCNN
COPY . /app/Mask_RCNN

FROM nvcr.io/nvidia/tensorflow:22.12-tf1-py3 as runner
WORKDIR /app
COPY --from=git-env /app /app
WORKDIR /download
RUN apt-get update && \
    apt-get install ffmpeg libsm6 libxext6 -y
WORKDIR /app/Mask_RCNN
# install dependencies and download pretrained model
RUN pip3 install -r requirements_docker.txt && \
    python3 setup.py install && \
    wget https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5
WORKDIR /app/coco/PythonAPI
RUN python3 setup.py install
WORKDIR /app/Mask_RCNN/samples/maskrcnn-demo-http-server
CMD ["python3", "maskrcnn_demo_http_server.py"]