import cv2
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from object_detection.utils import label_map_util
#%matplotlib inline

LABEL_MAP_PATH = "./label_map.pbtxt"
saved_model_dir = "./saved_model"



category_index = label_map_util.create_category_index_from_labelmap(LABEL_MAP_PATH, use_display_name=True)

# PATH_TO_SAVED_MODEL = os.path.join(saved_model_dir,"saved_model")
detect_fn = tf.saved_model.load(saved_model_dir)

test_dir = "./test"
files = [x for x in os.listdir(test_dir) if x.endswith(".jpg")]

def detect(image):
    image_np = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gray = image_np.copy()
    image_np = cv2.cvtColor(image_np,cv2.COLOR_GRAY2RGB)
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]
    detections = detect_fn(input_tensor)

    height, width, _ = image.shape
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                   for key, value in detections.items()}

    idx = np.where(detections["detection_scores"] > 0.5)
    coords = detections["detection_boxes"][idx]
    #ymin, xmin, ymax, xmax
    coords[:,0] = [max([0,x]) for x in coords[:,0]]
    coords[:,1] = [max([0,x]) for x in coords[:,1]]
    coords[:,2] = [min([1,x]) for x in coords[:,2]]
    coords[:,3] = [min([1,x]) for x in coords[:,3]]

    coords[:,0] *= height
    coords[:,1] *= width
    coords[:,2] *= height
    coords[:,3] *= width
    return coords

count = 1
for file in files[:5]: #visualize the first 5 images
    image = cv2.imread(os.path.join(test_dir,file))

    coords = detect(image)


    plt.figure(figsize=(8,8))
    plt.title(file)
    plt.imshow(image)
    plt.savefig("./image_" + str(count) + ".png")

    count = count + 1
    #drawing a rectange bounding box
    for coord in coords:
        plt.gca().add_patch(Rectangle((coord[2],coord[0]),coord[3]-coord[1],coord[2]-coord[0],linewidth=1,edgecolor='r',facecolor='none'))