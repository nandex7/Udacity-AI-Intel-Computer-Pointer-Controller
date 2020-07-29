'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

import cv2
import numpy as np
import logging as log
from openvino.inference_engine import IENetwork, IECore


class Head_pose_estimation_model:
    '''
    Class for the Head Pose Estimation Model.
    '''

    def __init__(self, model_name, device, extensions=None):
        
        self.model_weights = model_name + '.bin'
        self.model_structure = model_name + '.xml'
        self.device = device
        self.extension = extensions

        try:
            self.model = IENetwork(self.model_structure, self.model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")

        self.input_name = next(iter(self.model.inputs))
        self.input_shape = self.model.inputs[self.input_name].shape
        self.output_name = next(iter(self.model.outputs))
        self.output_shape = self.model.outputs[self.output_name].shape

    def load_model(self):
       
        self.model = IENetwork(self.model_structure, self.model_weights)
        self.core = IECore()
            
        self.net = self.core.load_network(network=self.model, device_name=self.device, num_requests=1)

    def predict(self, image):
       
        self.pre_image = self.preprocess_input(image)
        self.results = self.net.infer(inputs={self.input_name: self.pre_image})
        self.output_result = self.preprocess_output(self.results)
       
        return self.output_result

        self.image_processed = self.preprocess_input(image) 
        input_name = self.input_name
        image_processed_dict=  {input_name: self.image_processed}
        print('Pass this step Head Pose stimation.')
        infer = self.net.infer(image_processed_dict)
        self.output_infer = self.preprocess_output(self.infer)
        
        return self.output_infer
        
    def check_model(self):
        pass

    def preprocess_input(self, image):
     
        pre_frame = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        pre_frame = pre_frame.transpose((2, 0, 1))
        pre_frame = pre_frame.reshape(1, *pre_frame.shape)
        return pre_frame

    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        output = []
        output.append(outputs['angle_y_fc'].tolist()[0][0])
        output.append(outputs['angle_p_fc'].tolist()[0][0])
        output.append(outputs['angle_r_fc'].tolist()[0][0])
        return output
