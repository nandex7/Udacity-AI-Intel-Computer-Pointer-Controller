import cv2
import numpy as np
import logging as log
from openvino.inference_engine import IENetwork, IECore


class Facial_landmarks_detection_model:
    def __init__(self, model_name, device, extensions=None):
        
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
        try:
            self.model=IENetwork(self.model_structure, self.model_weights)
            self.core = IECore()
            #core = IECore()
            #model=core.read_network(model=model_structure, weights=model_weights)
            
            self.net = self.core.load_network(network=self.model, device_name=self.device, num_requests=1)

        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")

        self.input_name=next(iter(self.model.inputs))
        self.input_shape=self.model.inputs[self.input_name].shape
        self.output_name=next(iter(self.model.outputs))
        self.output_shape=self.model.outputs[self.output_name].shape

    def load_model(self):
        
        #self.core = IECore()
        #self.net = self.core.load_network(network = self.model, device_name = args.device, num_requests = 1)
        self.model = IENetwork(self.model_structure, self.model_weights)
        self.core = IECore()
        
    def predict(self, image):
     
        self.pre_image = self.preprocess_input(image)
        self.results = self.net.infer(inputs={self.input_name: self.pre_image})
        self.pre_output = self.preprocess_output(self.results, image)

        left_x_min=self.pre_output[0]-10
        left_y_min=self.pre_output[1]-10
        left_x_max=self.pre_output[0]+10
        left_y_max=self.pre_output[1]+10
        
        right_x_min=self.pre_output[2]-10
        right_y_min=self.pre_output[3]-10
        right_x_max=self.pre_output[2]+10
        right_y_max=self.pre_output[3]+10
        
        left_eye_img =  image[left_y_min:left_y_max, left_x_min:left_x_max]
        right_eye_img = image[right_y_min:right_y_max, right_x_min:right_x_max]
        eye_coords = [[left_x_min,left_y_min,left_x_max,left_y_max], [right_x_min,right_y_min,right_x_max,right_y_max]]
        
        return left_eye_img, right_eye_img, eye_coords

    def check_model(self):
        pass

    def preprocess_input(self, image):
      
        n, channel, height, width = self.input_shape
        
        image_converted=cv2.resize(image, (width, height), interpolation = cv2.INTER_AREA)
        
        # Change image from Hight Width Channel to Channel Hight Width
        image_converted = image_converted.transpose((2, 0, 1))
        image_converted = image_converted.reshape((n, channel, height,width))

        return image_converted


    def preprocess_output(self, outputs, image):
        
       
        outputs = outputs[self.output_name][0]
        left_eye_x = int(outputs[0] * image.shape[1])
        left_eye_y = int(outputs[1] * image.shape[0])
        right_eye_x = int(outputs[2] * image.shape[1])
        right_eye_y = int(outputs[3] * image.shape[0])

        return  (left_eye_x, left_eye_y,
                right_eye_x, right_eye_y)
