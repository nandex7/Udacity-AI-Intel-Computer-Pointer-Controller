'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

import cv2
import numpy as np
import logging as log
from openvino.inference_engine import IENetwork, IECore
import warnings
import math
warnings.filterwarnings("ignore")

class Gaze_estimation_model:
    '''
    Class for the Gaze Estimation Model.
    '''

    def __init__(self, model_name, device='CPU', extensions=None):
        '''
        this method is to set instance variables.
        '''
        self.model_weights = model_name + '.bin'
        self.model_structure = model_name + '.xml'
        self.device = device
        self.extension = extensions
        #self.threshold = threshold
        self.extension = extensions
        self.image_changed = None
        self.first_coords = None
        self.coords = None
        self.pre_image = None
        self.x_output=0
        self.y_output=0
        try:
            self.model = IENetwork(self.model_structure, self.model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")

        self.input_name=next(iter(self.model.inputs))
        self.input_shape=self.model.inputs[self.input_name].shape
        self.output_name=next(iter(self.model.outputs))
        self.output_shape=self.model.outputs[self.output_name].shape

    def load_model(self):
        
        self.model = IENetwork(self.model_structure, self.model_weights)
        self.core = IECore()
        self.net = self.core.load_network(network=self.model, device_name=self.device, num_requests=1)
        #self.net = self.core.load_network(network = self.model, device_name = args.device, num_requests = 1)

    def predict(self, left_eye_img, right_eye_img, output_head_pose_estimation):
        '''
        This method is meant for running predictions on the input image.
        '''
        self.left_eye_processed, self.right_eye_processed = self.preprocess_input(left_eye_img, right_eye_img)
        self.results = self.net.infer(
            inputs={'left_eye_image': self.left_eye_processed, 'right_eye_image': self.right_eye_processed,
                    'head_pose_angles': output_head_pose_estimation})
        self.cursor_coords, self.output_gaze = self.preprocess_output(self.results, output_head_pose_estimation)

        return self.cursor_coords, self.output_gaze

    
    def preprocess_input(self, left_eye_img, right_eye_img):
        
        resize =60

        left_eye_processed = cv2.resize(left_eye_img, (resize, resize))
        left_eye_processed = left_eye_processed.transpose((2, 0, 1))
        left_eye_processed = left_eye_processed.reshape(1, *left_eye_processed.shape)

        right_eye_processed = cv2.resize(right_eye_img, (resize, resize))
        right_eye_processed = right_eye_processed.transpose((2, 0, 1))
        right_eye_processed = right_eye_processed.reshape(1, *right_eye_processed.shape)
        return left_eye_processed, right_eye_processed

    def preprocess_output(self, outputs, output_head_pose_estimation):
        '''
       
        '''

        roll = output_head_pose_estimation[2]
        outputs = outputs[self.output_name][0]
        #converting to radians.
        cos_t = math.cos(roll * math.pi / 180)
        sin_t = math.sin(roll * math.pi / 180)

        x_output = outputs[0] * cos_t + outputs[1] * sin_t
        y_output = -outputs[0] * sin_t + outputs[1] * cos_t
        return (x_output, y_output), outputs
