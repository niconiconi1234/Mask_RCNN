import logging
from PIL import Image
import numpy as np
import io
import os
import sys
from PIL import Image
from flask import Flask, request
from waitress import serve as waitress_serve
import base64

logging.basicConfig(level=logging.INFO)


ROOT_DIR = os.path.abspath('../../')
sys.path.append(ROOT_DIR)
# To find local version
sys.path.append(os.path.join(ROOT_DIR, 'samples/coco/'))

# autopep8: off
import coco
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
logging.info('COCO_MODEL_PATH: '+COCO_MODEL_PATH)
model.load_weights(COCO_MODEL_PATH, by_name=True)
logging.info('Loading COCO MODEL Finished!')
model.keras_model._make_predict_function()


# flask server
app = Flask(__name__)


@app.route('/detect', methods=['POST'])
def detect():

    def numpy_to_base64(img):
        img = Image.fromarray(img.astype('uint8'))
        rawBytes = io.BytesIO()
        img.save(rawBytes, "JPEG")
        rawBytes.seek(0)
        return "data:image/jpeg;base64," + base64.b64encode(rawBytes.read()).decode('utf-8')

    def base64_to_numpy(b64_img):
        b64_img = ',' in b64_img and b64_img.split(',')[1] or b64_img
        img = Image.open(io.BytesIO(base64.b64decode(b64_img)))
        return np.array(img)

    b64_img = request.json['image']
    np_img = base64_to_numpy(b64_img)

    r = model.detect([np_img], verbose=1)[0]  # maskrcnn detect
    rois = r['rois']
    masks = r['masks']
    class_ids = r['class_ids']
    res_class_names = [class_names[i] for i in class_ids]
    scores = r['scores']

    # True to 255, False to 0
    masks = np.where(masks == True, 255, 0).astype(np.uint8)
    masks_imgs = []

    # number of instance
    noi = rois.shape[0]
    for i in range(noi):
        msk = masks[:, :, i]
        masks_imgs.append(numpy_to_base64(msk))

    return {
        'rois': rois.tolist(),
        'masks': masks_imgs,
        'class_names': res_class_names,
        'scores': scores.tolist()
    }


def serve():
    port = 50052
    waitress_serve(app, host='0.0.0.0', port=port)


if __name__ == '__main__':
    logging.info('in main')
    serve()
