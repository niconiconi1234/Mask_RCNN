import grpc
import maskrcnn_pb2
import maskrcnn_pb2_grpc
import logging
import requests
import numpy as np
import io


def run():
    image_url = 'https://img1.baidu.com/it/u=2144028537,2703596309&fm=253&fmt=auto&app=138&f=JPEG?w=500&h=889'
    http_rsp = requests.get(image_url)

    assert http_rsp.status_code == 200

    image_binary = http_rsp.content

    with grpc.insecure_channel('localhost:50051', options=[
        ('grpc.max_receive_message_length', 100*1024*1024),
    ]) as channel:
        stub = maskrcnn_pb2_grpc.MaskRCNNStub(channel)
        response = stub.maskrcnn(maskrcnn_pb2.MaskRCNNRequest(b_image=image_binary))

        # # class_names = np.frombuffer(response.b_class_names, dtype=str)
        # b_rois = np.frombuffer(response.b_rois, dtype=np.int32)
        # b_class_ids = np.frombuffer(response.b_class_ids, dtype=np.int32)
        # b_scores = np.frombuffer(response.b_scores, dtype=np.float32)
        # b_masks = np.frombuffer(response.b_masks, dtype=bool)

        def bytes2nparray(bytes):
            np_bytes = io.BytesIO(bytes)
            return np.load(np_bytes, allow_pickle=True)

        class_names = bytes2nparray(response.b_class_names)
        rois = bytes2nparray(response.b_rois)
        class_ids = bytes2nparray(response.b_class_ids)
        scores = bytes2nparray(response.b_scores)
        masks = bytes2nparray(response.b_masks)

        a = 3


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    run()
