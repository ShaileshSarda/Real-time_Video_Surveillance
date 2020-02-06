# Imports Libraries and Packages
import tensorflow as tf
from utils import backbone
from object_count_algorithm import object_counter

# Provide the input video path
input_video = "./input_video/Pedestrian_overpass.mp4"

detection_graph, category_index = backbone.set_model('ssd_mobilenet_v1_coco_2018_01_28', 'mscoco_label_map.pbtxt')

is_color_recognition_enabled = 0 # Set it to 1 for enabling the color prediction for the detected objects
ROI = 900 # Set roi line position to count the detected objects
deviation = 1 # To represent the object counting area

object_counter.object_counting(input_video, detection_graph, category_index, is_color_recognition_enabled, ROI, deviation) # counting all the objects
