FROM alpine/git as git-env
WORKDIR /app
RUN git clone https://github.com/waleedka/coco && git clone https://github.com/niconiconi1234/Mask_RCNN
COPY . /app/Mask_RCNN

FROM tensorflow/tensorflow:1.15.0-gpu-py3 as runner
WORKDIR /app
COPY --from=git-env /app /app
WORKDIR /download
# Nvidia has updated the key, so we need to get the new one. For more info, see: https://forums.developer.nvidia.com/t/notice-cuda-linux-repository-key-rotation/212772
RUN apt-key del 7fa2af80 && \
    rm -rf /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get update && \
    apt-get install wget -y && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb && \
    dpkg -i cuda-keyring_1.0-1_all.deb && \
    apt-get update && \
    apt-get install ffmpeg libsm6 libxext6 -y
WORKDIR /app/Mask_RCNN
# install dependencies and download pretrained model
RUN pip3 install -r requirements.txt && \
    python3 setup.py install && \
    wget https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5
WORKDIR /app/coco/PythonAPI
RUN python3 setup.py install
WORKDIR /app/Mask_RCNN/samples/maskrcnn-demo-http-server
CMD ["python3", "maskrcnn_demo_http_server.py"]