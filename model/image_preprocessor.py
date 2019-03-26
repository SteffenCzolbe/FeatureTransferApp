"""
aligns custom images.
code adapted from: 
https://github.com/carykh/alignedCelebFaces/blob/master/code/faceAligner.py
"""


import face_recognition
import numpy as np
from skimage import transform
import os.path
import time
from PIL import Image
import shutil

DESIRED_X = 32
DESIRED_Y = 31
DESIRED_SIZE = 22

FINAL_IMAGE_WIDTH = 64
FINAL_IMAGE_HEIGHT = 64


def getAvg(face, landmark):
    cum = np.zeros((2))
    for point in face[landmark]:
        cum[0] += point[0]
        cum[1] += point[1]
    return cum/len(face[landmark])

def getNorm(a):
    return (a-np.mean(a))/np.std(a)
                
def align_image(fname, dest_fname):
    image_face_info = face_recognition.load_image_file(fname)
    face_landmarks = face_recognition.face_landmarks(image_face_info)

    image_numpy = np.array(Image.open(fname).convert('RGB'))
    colorAmount = 0
    if len(image_numpy.shape) == 3:
        nR = getNorm(image_numpy[:,:,0])
        nG = getNorm(image_numpy[:,:,1])
        nB = getNorm(image_numpy[:,:,2])
        colorAmount = np.mean(np.square(nR-nG))+np.mean(np.square(nR-nB))+np.mean(np.square(nG-nB))
    if not (len(face_landmarks) == 1 and colorAmount >= 0.04): # We need there to only be one face in the image, AND we need it to be a colored image.
        print('WARNING: none / too many faces recognized in file "' + fname + '"')
    else:
        leftEyePosition = getAvg(face_landmarks[0],'left_eye')
        rightEyePosition = getAvg(face_landmarks[0],'right_eye')
        nosePosition = getAvg(face_landmarks[0],'nose_tip')
        mouthPosition = getAvg(face_landmarks[0],'bottom_lip')

        centralPosition = (leftEyePosition+rightEyePosition)/2

        faceWidth = np.linalg.norm(leftEyePosition-rightEyePosition)
        faceHeight = np.linalg.norm(centralPosition-mouthPosition)
        # if faceWidth >= faceHeight*0.7 and faceWidth <= faceHeight*1.5:

        faceSize = (faceWidth+faceHeight)/2

        toScaleFactor = faceSize/DESIRED_SIZE
        toXShift = (centralPosition[0])
        toYShift = (centralPosition[1])
        toRotateFactor = np.arctan2(rightEyePosition[1]-leftEyePosition[1],rightEyePosition[0]-leftEyePosition[0])

        rotateT = transform.SimilarityTransform(scale=toScaleFactor,rotation=toRotateFactor,translation=(toXShift,toYShift))
        moveT = transform.SimilarityTransform(scale=1,rotation=0,translation=(-DESIRED_X,-DESIRED_Y))

        outputArr = transform.warp(image=image_numpy,inverse_map=(moveT+rotateT))[0:FINAL_IMAGE_HEIGHT,0:FINAL_IMAGE_WIDTH]

        img = Image.fromarray(np.uint8(outputArr*255))
        img.save(dest_fname)

def preprocess_images(): 
    additonal_images = os.listdir('./additional_images/')
    additonal_images.remove('.gitkeep')
    if len(additonal_images) > 0:
        if os.path.isdir('./additional_images_preprocessed/'):
            shutil.rmtree('./additional_images_preprocessed/') 
        
        os.mkdir('./additional_images_preprocessed/')

        for fname in additonal_images:
            name, ext = fname.split('.')
            align_image('./additional_images/' + fname, './additional_images_preprocessed/' + name + '.png')