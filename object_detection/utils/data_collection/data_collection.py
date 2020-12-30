#!/usr/bin/env python3

import numpy as np
from cv2 import cv2
from datetime import datetime
import os

from agent import PurePursuitPolicy
from utils import launch_env, seed
from utils import launch_env, seed, makedirs, display_seg_mask, display_img_seg_mask

DATASET_DIR="../../dataset"
sim_path = '/home/mokleit/dt-exercises/object_detection/sim/'

ext = '.png'

ranges = {
    'cone': {'lower': (167,100,94), 'upper': (255,119,110)},
    'bus': {'lower': (189,119,0), 'upper': (255,255,30)},
    'duckie': {'lower': (86,92,200), 'upper': (150,140,255)},
    'truck': {'lower': (107,90,100), 'upper': (146,125,140)},
    'background': {'lower': (224,0,246), 'upper': (255,10,255)}
}

kernels = {
    'background': {'kernel': (15,15), 'lower': 100},
    'bus': {'kernel': (7,7), 'lower': 120},
    'cone': {'kernel': (5,5), 'lower': 115},
    'duckie': {'kernel': (3,3), 'lower': 115},
    'truck': {'kernel': (7,7), 'lower':105}
}

npz_index = 0
def save_npz(img, boxes, classes):
    global npz_index
    filename = sim_path + 'npz/' + str(npz_index) + '.npz'
    np.savez(filename, *(img, boxes, classes))
    print('Saved ', npz_index)
    npz_index += 1

def remove_snow(img, save):    
    #Remove snow
    # kernel = np.ones((6,6),np.uint8)
    # filtered = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    #Save raw and filtered images
    if save:
        #Define image names
        # filtered_image_name = sim_path + 'raw/filtered_image_' + str(mask_index) + ext  
        raw_image_name = sim_path + 'raw/raw_image_' + str(npz_index) + ext
        cv2.imwrite(raw_image_name, img)
        # cv2.imwrite(filtered_image_name, filtered)
    
    return img

def compute_bounding_box_coordinates(image, name):
    bboxes = []
    kernel = np.ones(kernels[name]['kernel'], np.uint8)
    filtered = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    # Convert to gray scale
    gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, kernels[name]['lower'], 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for countour in contours:
        x,y,w,h = cv2.boundingRect(countour)
        cv2.rectangle(filtered,(x,y),(x+w,y+h),(0,255,0),2) 
        box_coordinates = [x, y, x + w, y + h]
        bboxes.append(box_coordinates)
    
    cv2.imwrite(sim_path + 'mask/'+name+ str(npz_index)+'.png', filtered)

    return bboxes  

def mask_object(filtered, name):
    duckie_mask = cv2.inRange(filtered, ranges[name]['lower'], ranges[name]['upper'])
    temp = filtered.copy()
    temp[duckie_mask == 0] = (0,0,0)
    return temp

def clean_segmented_image(seg_img):
    # TODO
    # Tip: use either of the two display functions found in util.py to ensure that your cleaning produces clean masks
    # (ie masks akin to the ones from PennFudanPed) before extracting the bounding boxes
    bboxes = []
    classes = []
    snowless_img = remove_snow(seg_img, True)
    
    #Compute mask and bounding boxes for each class
    background_mask = mask_object(snowless_img, 'background')
    background_bboxes = compute_bounding_box_coordinates(background_mask, 'background')
    bboxes.extend(background_bboxes)
    backgrounds = np.zeros(len(background_bboxes), dtype=int)
    classes.extend(backgrounds)

    duckie_mask = mask_object(snowless_img, 'duckie')
    duckie_bboxes = compute_bounding_box_coordinates(duckie_mask, 'duckie')
    bboxes.extend(duckie_bboxes)
    duckies = np.ones(len(duckie_bboxes), dtype=int)
    classes.extend(duckies)

    cone_mask = mask_object(snowless_img, 'cone')
    cone_bboxes = compute_bounding_box_coordinates(cone_mask, 'cone')
    bboxes.extend(cone_bboxes)
    cones = np.full(len(cone_bboxes), 2, dtype=int)
    classes.extend(cones)

    truck_mask = mask_object(snowless_img, 'truck')
    truck_bboxes = compute_bounding_box_coordinates(truck_mask, 'truck')
    bboxes.extend(truck_bboxes)
    trucks = np.full(len(truck_bboxes), 3, dtype=int)
    classes.extend(trucks)   

    bus_mask = mask_object(snowless_img, 'bus')
    bus_bboxes = compute_bounding_box_coordinates(bus_mask, 'bus')
    bboxes.extend(bus_bboxes)
    buses = np.full(len(bus_bboxes), 4, dtype=int)
    classes.extend(buses)   

    return bboxes, classes

seed(123)
environment = launch_env()

policy = PurePursuitPolicy(environment)

MAX_STEPS = 500

while True:
    obs = environment.reset()
    environment.render(segment=True)
    rewards = []

    nb_of_steps = 0

    while True:
        action = policy.predict(np.array(obs))

        obs, rew, done, misc = environment.step(action) # Gives non-segmented obs as numpy array
        segmented_obs = environment.render_obs(True)  # Gives segmented obs as numpy array

        rewards.append(rew)
        environment.render(segment=int(nb_of_steps / 50) % 2 == 0)

        # TODO boxes, classes = clean_segmented_image(segmented_obs)
        # TODO save_npz(obs, boxes, classes)
        resized_img = cv2.resize(segmented_obs, (224,224))
        boxes, classes = clean_segmented_image(resized_img)
        save_npz(resized_img, boxes, classes)

        nb_of_steps += 1

        if done or nb_of_steps > MAX_STEPS:
            break