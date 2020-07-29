'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import cv2
import numpy as np
from openvino.inference_engine import IENetwork,IECore
import logging as logger
class Face_detection_model:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device, threshold,extensions=None):
    
        self.model_weights = model_name+'.bin'
        self.model_structure = model_name+'.xml'
        self.device = device
        self.threshold = threshold
        self.extension = extensions
        self.image_changed = None
        self.first_coords = None
        self.coords = None
        self.pre_image = None
        self.net = None
        self.net = None
        self.core = None
        try:
            self.model=IENetwork(self.model_structure, self.model_weights)
            #core = IECore()
            #model=core.read_network(model=model_structure, weights=model_weights)
             
        except Exception as e:
            raise ValueError("Could not Initialize the network. Have you enterred the correct model path?")

        self.input_name=next(iter(self.model.inputs))
        self.input_shape=self.model.inputs[self.input_name].shape
        self.output_name=next(iter(self.model.outputs))
        self.output_shape=self.model.outputs[self.output_name].shape

    def load_model(self):
       
        self.model = IENetwork(self.model_structure, self.model_weights)
        self.core = IECore()
        self.net = self.core.load_network(network=self.model, device_name=self.device, num_requests=1)
        #self.net = self.core.load_network(network = self.model, device_name = args.device, num_requests = 1)

    def predict(self, image):
       
         
        self.image_processed = self.preprocess_input(image) 
        input_name = self.input_name
        image_processed_dict=  {input_name: self.image_processed}
        print('Pass this step')
        infer = self.net.infer(image_processed_dict)
        output = infer[self.output_name]
       

        self.coords, self.image = self.draw_outputs(output, image)

        self.first_coords = self.coords[0]

        print("First coords:",self.first_coords)

        image_change = image[self.first_coords[1]:self.first_coords[3],
                             self.first_coords[0]:self.first_coords[2]]

        return self.first_coords, image_change

        
    def draw_outputs(self, result, image):
        count = 0  
        width_height = list(image.shape)
        width =width_height[1]
        height =width_height[0]
        obj_points=[]
        for obj_input in result[0][0]:  
            confidence = obj_input[2]  
            if confidence >= self.threshold: 
                x_min = int(obj_input[3] * width)
                y_min = int(obj_input[4] * height)
                x_max = int(obj_input[5] * width)
                y_max = int(obj_input[6] * height)  
                obj_points.append((x_min, y_min, x_max, y_max))
                image = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 255), 1) 
                Person_confidence = '%s: %.1f%%' % ("Person", round(confidence * 100, 1))
                #Person_coord = '%s: %.1f%%' % ("Coord:", obj_input) 
                y_pixel= 120
                cv2.putText(image, Person_confidence, (15, y_pixel), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
               # cv2.putText(image, Person_coord, (15, y_pixel), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            count += 1    
        
        return obj_points ,image

    def check_model(self):
        raise NotImplementedError

    def preprocess_input(self, image):
        
        n, channel, height, width = self.input_shape
        
        image_converted=cv2.resize(image, (width, height), interpolation = cv2.INTER_AREA)
        
        # Change image from Hight Width Channel to Channel Hight Width
        image_converted = image_converted.transpose((2, 0, 1))
        image_converted = image_converted.reshape((n, channel, height,width))

        return image_converted


    def preprocess_output(self, outputs):
        self.outputs = cv2.resize(outputs, (self.output_shape[3], self.output_shape[2]))
        return self.outputs
