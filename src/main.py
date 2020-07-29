import cv2
import os
import numpy as np
import logging as log
import time
from face_detection import Face_detection_model
from facial_landmarks_detection import Facial_landmarks_detection_model
from gaze_estimation import Gaze_estimation_model
from head_pose_estimation import Head_pose_estimation_model
from mouse_controller import MouseController
from argparse import ArgumentParser
from input_feeder import InputFeeder
import math


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-fm", "--face_detection_model", required=True, type=str,
                        help="Path xml file - face detection model ")
    parser.add_argument("-lm", "--facial_landmarks_model", required=True, type=str,
                        help="Path xml file - facial landmarks detection model")
    parser.add_argument("-hm", "--head_pose_model", required=True, type=str,
                        help="Path xml file head pose estimation model xml")
    parser.add_argument("-gm", "--gaze_estimation_model", required=True, type=str,
                        help="Path xml file  gaze estimation model")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path video file or CAM")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    # So we are using four models to run so the device receive device for each model this is more for testing MYRIAD and 
    # NCS2 OR MYRIAD doesn't support FP32 . So FACE DETECTION just support FP32                          
    parser.add_argument("-d", "--device", type=str, default="CPU,CPU,CPU,CPU",
                        help="Specify the target device to infer in the order: FACE DETECTION, LANDMARK, HEAD POSE AND GAZE"
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                            " (CPU CPU CPU CPU  by default)"
                            "MYRIAD SAMPLE: CPU,MYRIAD,MYRIAD,MYRIAD")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                             "(0.5 by default)")
    parser.add_argument("-vf", "--visualization_flag", required=False, nargs='+',
                        default=[],
                        help="Example:   "
                             "for see the visualization of different model outputs of each frame,"
                             "fm for Face Detection Model, lm for Facial Landmark Detection Model"
                             "hm for Head Pose Estimation Model, gm for Gaze Estimation Model.")
    parser.add_argument("-o", '--output_path', default='/benchmark/', type=str)                         
    return parser

def process_visual_flags(
                        frame_image,visual_flags,frame,image_change,first_coords
                        ,eye_coord,output_head_pose_estimation) :

    color = (0, 255, 0) # GREEN 
    if 'fm' in visual_flags:
        cv2.rectangle(frame_image, (first_coords[0], first_coords[1]),
                                  (first_coords[2], first_coords[3]), color, 5)
    
    if 'lm' in visual_flags:
        cv2.rectangle(frame_image, (eye_coord[0][0], eye_coord[0][1]), (eye_coord[0][2],eye_coord[0][3]),
                              color,2)
        cv2.rectangle(frame_image, (eye_coord[1][0], eye_coord[1][1]), (eye_coord[1][2], eye_coord[1][3]),
                              color,2)
    if 'hm' in visual_flags:
        cv2.putText(frame_image,"Headpose ouput yaw:{:.2f} | pitch:{:.2f} | roll:{:.2f}".format(output_head_pose_estimation[0],
                                                                             output_head_pose_estimation[1],
                                                                             output_head_pose_estimation[2]),
                            (20, 20), cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)
    if 'gm' in visual_flags:
        position_half_face = (frame_image.shape[1] / 2, frame_image.shape[0] / 2, 0)
        cv2.putText(frame_image,"Gaze output: x= {:.2f} , y= {:.2f} , z= {:.2f}".format(
                output_head_pose_estimation[0], output_head_pose_estimation[1], output_head_pose_estimation[2]),(20, 80),
            cv2.FONT_HERSHEY_COMPLEX,1, color, 2)
        draw_axes(frame_image, position_half_face, output_head_pose_estimation[0], output_head_pose_estimation[1], output_head_pose_estimation[2], 50, 950)
        
    return frame_image

# https://knowledge.udacity.com/questions/171017
#Reference code Udacity knowledge.
def draw_axes(frame, center_of_face, yaw, pitch, roll, scale, focal_length):
    yaw *= np.pi / 180.0
    pitch *= np.pi / 180.0
    roll *= np.pi / 180.0
    cx = int(center_of_face[0])
    cy = int(center_of_face[1])
    r_x = np.array([[1, 0, 0],
                    [0, math.cos(pitch), -math.sin(pitch)],
                    [0, math.sin(pitch), math.cos(pitch)]])
    r_y = np.array([[math.cos(yaw), 0, -math.sin(yaw)],
                    [0, 1, 0],
                    [math.sin(yaw), 0, math.cos(yaw)]])
    r_z = np.array([[math.cos(roll), -math.sin(roll), 0],
                    [math.sin(roll), math.cos(roll), 0],
                    [0, 0, 1]])

    r = r_z @ r_y @ r_x
    camera_matrix = build_camera_matrix(center_of_face, focal_length)
    xaxis = np.array(([1 * scale, 0, 0]), dtype='float32').reshape(3, 1)
    yaxis = np.array(([0, -1 * scale, 0]), dtype='float32').reshape(3, 1)
    zaxis = np.array(([0, 0, -1 * scale]), dtype='float32').reshape(3, 1)
    zaxis1 = np.array(([0, 0, 1 * scale]), dtype='float32').reshape(3, 1)
    o = np.array(([0, 0, 0]), dtype='float32').reshape(3, 1)
    o[2] = camera_matrix[0][0]
    xaxis = np.dot(r, xaxis) + o
    yaxis = np.dot(r, yaxis) + o
    zaxis = np.dot(r, zaxis) + o
    zaxis1 = np.dot(r, zaxis1) + o
    xp2 = (xaxis[0] / xaxis[2] * camera_matrix[0][0]) + cx
    yp2 = (xaxis[1] / xaxis[2] * camera_matrix[1][1]) + cy
    p2 = (int(xp2), int(yp2))
    cv2.line(frame, (cx, cy), p2, (0, 0, 255), 2)
    xp2 = (yaxis[0] / yaxis[2] * camera_matrix[0][0]) + cx
    yp2 = (yaxis[1] / yaxis[2] * camera_matrix[1][1]) + cy
    p2 = (int(xp2), int(yp2))
    cv2.line(frame, (cx, cy), p2, (0, 255, 0), 2)
    xp1 = (zaxis1[0] / zaxis1[2] * camera_matrix[0][0]) + cx
    yp1 = (zaxis1[1] / zaxis1[2] * camera_matrix[1][1]) + cy
    p1 = (int(xp1), int(yp1))
    xp2 = (zaxis[0] / zaxis[2] * camera_matrix[0][0]) + cx
    yp2 = (zaxis[1] / zaxis[2] * camera_matrix[1][1]) + cy
    p2 = (int(xp2), int(yp2))
    cv2.line(frame, p1, p2, (255, 0, 0), 2)
    cv2.circle(frame, p2, 3, (255, 0, 0), 2)
    return frame


def build_camera_matrix(center_of_face, focal_length):
    cx = int(center_of_face[0])
    cy = int(center_of_face[1])
    camera_matrix = np.zeros((3, 3), dtype='float32')
    camera_matrix[0][0] = focal_length
    camera_matrix[0][2] = cx
    camera_matrix[1][1] = focal_length
    camera_matrix[1][2] = cy
    camera_matrix[2][2] = 1
    return camera_matrix


def main():
    # command line arguments
    args = build_argparser().parse_args()
    input_filename = args.input
    log_object = log.getLogger()
    visual_flags = args.visualization_flag
    print("Visual flags:",visual_flags)

    device_models=args.device
    print("deviceModels:",device_models)
    device_list = device_models.split(",")
    print("deviceList:",device_list)
    print("deviceFirst:",device_list[1])
    str_cam ="cam"
    output_path = args.output_path

    if input_filename.lower() == str_cam:
        input_feeder = InputFeeder(str_cam)
    else:
        if not os.path.isfile(input_filename):
            log_object.error("Error: Can not find the video o image file.")
            exit(1)
        input_feeder = InputFeeder("video", input_filename)


    
    obj_face_detection = Face_detection_model(model_name=args.face_detection_model,device=device_list[0], threshold=args.prob_threshold,
                                                          extensions=args.cpu_extension)
    
    obj_facial_landmarks  = Facial_landmarks_detection_model(model_name=args.facial_landmarks_model,device=device_list[1],
                                                             extensions=args.cpu_extension)

    obj_gaze_estimation = Gaze_estimation_model(model_name=args.gaze_estimation_model, device=device_list[2],
                                                 extensions=args.cpu_extension)

    obj_head_pose_estimation = Head_pose_estimation_model(model_name=args.head_pose_model, device=device_list[3]
                                                        , extensions=args.cpu_extension)
    
    mouse_controller_object = MouseController('medium', 'fast')

    start_time = time.time()
    obj_face_detection.load_model()
    model_load_time_face = time.time() - start_time
    
    start_landmark_time = time.time()
    obj_facial_landmarks.load_model()
    model_load_time_landmarks = time.time() - start_landmark_time

    start_headpose_time = time.time()
    obj_head_pose_estimation.load_model()
    model_load_time_headpose = time.time() - start_headpose_time    

 
    start_gaze_time = time.time()
    obj_gaze_estimation.load_model()
    model_load_time_gaze = time.time() - start_gaze_time

    models_load_time = time.time() - start_time
    
    log_object.info("Info:Models loading time(face, landmark, gaze, head_pose): {:.3f} ms".format(models_load_time * 1000))
    
    input_feeder.load_data()
    
    counter = 0
    start_inference_time = time.time()
    
    log_object.info("Info:Start inferencing ")
    print(input_feeder.next_batch())
    for ret,frame in input_feeder.next_batch():
        #print(flag)
        #print(frame)
        if not ret:
            break
        pressed_key = cv2.waitKey(60)
        counter = counter + 1
        print("counter:",counter)


        
        first_coords, image_change = obj_face_detection.predict(frame)
        inference_face_time = round(time.time() - start_inference_time, 1)
        
        print("Inference face time:",inference_face_time)

        left_eye_img, right_eye_img, eye_coord = obj_facial_landmarks.predict(image_change)
        inference_landmark_time = round(time.time() - start_inference_time, 1)

        print("Inference landmark time:",inference_landmark_time)
        
        if first_coords == 0:
            continue
            
        output_head_pose_estimation = obj_head_pose_estimation.predict(image_change)
        inference_head_time = round(time.time() - start_inference_time, 1)

        print("Inference inference_head_time:",inference_head_time)
        
        mouse_coordinate, gaze_vector = obj_gaze_estimation.predict(left_eye_img, right_eye_img,
                                                                             output_head_pose_estimation)

        inference_gaze_time = round(time.time() - start_inference_time, 1)

        print("Inference inference_gaze_time:",inference_gaze_time)

        frame_image = frame.copy()                
        if len(visual_flags) != 0:

            preview=process_visual_flags(frame_image,visual_flags,frame,image_change,first_coords
                                                ,eye_coord,output_head_pose_estimation) 
        else:
            preview = frame_image

        

        fps_face = int(counter) / inference_gaze_time
        color =(0,255,0)

        cv2.putText(frame_image,"Inference: = {:.2f}".format(inference_gaze_time),(20, 180),cv2.FONT_HERSHEY_COMPLEX,1, color, 2)  
        mouse_controller_object.move(mouse_coordinate[0], mouse_coordinate[1])        

        cv2.putText(frame_image,"FPS: = {:.2f}".format(fps_face),(20, 220),cv2.FONT_HERSHEY_COMPLEX,1, color, 2)  
        
        image_new = cv2.resize(preview, (700, 700))
        cv2.imshow('Visualization', image_new)
        mouse_controller_object.move(mouse_coordinate[0], mouse_coordinate[1])        
                 
    
        if pressed_key == 27:
            log_object.error("exit key is pressed..")
            break

    #Time calculations for every model.
    #inference_facefinal_time =    inference_face_time
    #inference_landmarkfinal_time    = inference_landmark_time - inference_face_time
    #inference_headfinal_time    = inference_head_time - inference_landmark_time
    #inference_gazefinal_time    = inference_gaze_time - inference_head_time

    inference_total_time = round(time.time() - start_inference_time, 1)
    print("Inference inference_total_time:",inference_total_time)
    
    #fps_face = int(counter) / inference_face_time
    #fps_landmark = int(counter) / inference_landmark_time
    #fps_head  = int(counter) / inference_head_time
    #fps_gaze  = int(counter) / inference_gaze_time
    fps_total = int(counter) / inference_total_time
    print("fps_total:",fps_total)
    with open(output_path+'statstotal.txt', 'w') as f:    
        f.write(str(inference_total_time) + '\n')
        f.write(str(fps_total) + '\n')
        f.write(str(models_load_time) + '\n')


    with open(output_path+'statsmodels.txt', 'w') as f:
       # f.write(str(inference_facefinal_time)+ ','+str(inference_landmarkfinal_time)+','+str(inference_headfinal_time)+','+str(inference_gazefinal_time)+ '\n')
       # f.write(str(fps_face)+ ','+str(fps_landmark)+','+str(fps_head)+','+str(fps_gaze)+ '\n')
       # f.write(str(model_load_time_face)+ ','+str(model_load_time_landmarks)+','+str(model_load_time_headpose)+','+str(model_load_time_gaze)+ '\n')
        f.write(str(model_load_time_face) + '\n')
        f.write(str(model_load_time_landmarks) + '\n')
        f.write(str(model_load_time_headpose) + '\n')
        f.write(str(model_load_time_gaze) + '\n')
        
    log_object.info("Info:Finishing Video")    
    input_feeder.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
