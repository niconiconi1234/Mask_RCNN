import logging
import maskrcnn_pb2
import maskrcnn_pb2_grpc
import grpc
from concurrent import futures
from PIL import Image
import numpy as np
import io
import os
import sys
from mrcnn import visualize
from PIL import Image

ROOT_DIR = os.path.abspath('../../')
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'samples/coco/'))  # To find local version

# autopep8: off
import coco
from mrcnn import visualize
import mrcnn.model as modellib
from mrcnn import utils
# autopep8: on

# load model trained from coco dataset
COCO_MODEL_PATH = os.path.join(ROOT_DIR, 'mask_rcnn_coco.h5')
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


class InferenceConfig(coco.CocoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


# inference config
config = InferenceConfig()
config.display()

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

# create model object in inference mode.
MODEL_DIR = os.path.join(ROOT_DIR, 'logs')
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)
model.keras_model._make_predict_function()


class MyMaskRCNNServicer(maskrcnn_pb2_grpc.MaskRCNNServicer):
    def maskrcnn(self, request: maskrcnn_pb2.MaskRCNNRequest, context):
        b_image = request.b_image
        image_stream = io.BytesIO(b_image)
        image_np = np.array(Image.open(image_stream))

        result = model.detect([image_np], verbose=1)[0]  # maskrcnn detect
        rois = result['rois']
        masks = result['masks']
        class_ids = result['class_ids']
        scores = result['scores']

        masked_image = visualize.display_instances(image_np, rois, masks, class_ids,
                                                   class_names, scores=scores, show_bbox=True, show_mask=True)
        pil_masked_image = Image.fromarray(masked_image)
        file_path = (os.path.join(ROOT_DIR, 'samples/maskrcnn-demo-server/masked_image.jpg'))
        pil_masked_image.save(file_path)

        def nparray2bytes(nparray):
            np_bytes = io.BytesIO()
            np.save(np_bytes, nparray, allow_pickle=True)
            return np_bytes.getvalue()

        return maskrcnn_pb2.MaskRCNNResponse(b_rois=nparray2bytes(rois),
                                             b_masks=nparray2bytes(masks),
                                             b_class_ids=nparray2bytes(class_ids),
                                             b_class_names=nparray2bytes(np.array(class_names)),
                                             b_scores=nparray2bytes(scores))


def serve():
    port = 50051
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    maskrcnn_pb2_grpc.add_MaskRCNNServicer_to_server(MyMaskRCNNServicer(), server)
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    logging.info(f'maskrcnn server started at port {port}')
    server.wait_for_termination()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    serve()
