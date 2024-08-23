"""
only work in window ,,,,because virtual box issue.

for MTCNN, detection["keypoints"]: has:
left_eye
right_eye
nose
mouth_left
so, we can not determine the mouth open by MTCNN only.
use dlib 





"""
import os
import json
import argparse
import time
from mtcnn import MTCNN
import pandas as pd
import matplotlib.pyplot as plt
from deepface import DeepFace
import cv2 
import moviepy.editor as mp
#import mediapipe

from scipy.spatial import distance as dist
from imutils import face_utils 
import numpy as np 
import imutils 
import dlib 

parser = argparse.ArgumentParser()
parser.add_argument('filename', type=str)

args = parser.parse_args()
if not os.path.exists(args.filename):
    os.mkdir(args.filename)

facesdir = os.path.join(args.filename, "faces")
if not os.path.exists(facesdir):
    os.mkdir(facesdir)

#facesdbdir = os.path.join(facesdir, "facedb")
#if not os.path.exists(facesdbdir):
#    os.mkdir(facesdbdir)


input = args.filename + '.mp4'  
# Insert Local Video File Path
clip = mp.VideoFileClip(input)
"""
use MTCNN
"""

"""
https://github.com/mauckc/mouth-open/blob/master/detect_open_mouth.py#L17
"""

def mouth_aspect_ratio(mouth):
	# compute the euclidean distances between the two sets of
	# vertical mouth landmarks (x, y)-coordinates
	A = dist.euclidean(mouth[2], mouth[10])  # 51, 59
	B = dist.euclidean(mouth[4], mouth[8])  # 53, 57

	# compute the euclidean distance between the horizontal
	# mouth landmark (x, y)-coordinates
	C = dist.euclidean(mouth[0], mouth[6])  # 49, 55

	# compute the mouth aspect ratio
	mar = (A + B) / (2.0 * C)

	# return the mouth aspect ratio
	return mar


#detector0 = dlib.get_frontal_face_detector()
predictor0 = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# grab the indexes of the facial landmarks for the mouth
(mStart, mEnd) = (49, 68)

detections = []
faceList = []

""" 
facelist : [
   {  img: ,  faceid:, gender:  }
] 
"""

#rects = detector0(img, 0)
detector = MTCNN()


def calculateMouth(img, box):
    x, y, w, h = box
    rect = dlib.rectangle(x, y, x+w, y+h) 
    shape = predictor0(img, rect)
    shape = face_utils.shape_to_np(shape) 
    # extract the mouth coordinates, then use the
    # coordinates to compute the mouth aspect ratio
    mouth = shape[mStart:mEnd] 
    mouthMAR = mouth_aspect_ratio(mouth) 
    print(" mouth = " + str(mouthMAR))
    return mouthMAR


 
 

"""
check if a face already exits, if not, put into list
return face  object
"""
def registerFace(img1): 
    for face in faceList:
        output = DeepFace.verify(img1, face["img"], enforce_detection=False)
        if output['verified']:
            return face
   
    aface = {"faceid": len(faceList), "img": img1, "gender": "", "emotion": ""}
    faceList.append(aface)
    return aface

""" 
def populateSpeakerToModel():
    for i in range(finalcount):
        agroup = regroupList[i]
        faceid = agroup.get("faceid", -1)
        if faceid == -1:
            continue
        face = faceList[faceid]
        if face["gender"] == '':
            obj = DeepFace.analyze(img_path=face["img"], actions=[
                'gender'], enforce_detection=False)  # , 'emotion'])
            if len(obj) > 0:
                face["gender"] = "Female" if obj[0]["gender"]["Woman"] > obj[0]["gender"]["Man"] else "Male"
        agroup["speaker"] = { "id": faceid, "gender": face["gender"]}
"""

"""
find active speaker in clip[startTime, endTime]
return: face obj config
"""
def detectSpeaker(startTime, endTime):
    fid2mouth = {}
    duration = endTime - startTime
    toffset = 0.2
    if duration > 7:
        toffset = 3
    elif duration >4:
        toffset = 1
    framedir = os.path.join(facesdir,   "tmp.png")
    count = 5
    while count > 0 and startTime + toffset < endTime:
        clip.save_frame(framedir, t=startTime + toffset)
        fimg = cv2.cvtColor(cv2.imread(framedir), cv2.COLOR_BGR2RGB)
        detections = detector.detect_faces(fimg)  
        for detection in detections:
            confidence = detection["confidence"]
            #for a in detection["keypoints"]:
            #  print(a)
            if confidence > 0.85:
                #print(detection["box"])
                mouth = calculateMouth(fimg, detection["box"])
                x, y, w, h = detection["box"] 
                detected_face = fimg[int(y):int(y+h), int(x):int(x+w)]
                faceobj = registerFace(detected_face)
                faceid = faceobj["faceid"]
                config = fid2mouth.get(str(faceid), None)
                if config is None:
                    config = {"id": faceid, "min": mouth, "max": mouth, "x": x,
                              "y": y, "w": w, "h": h, "time": startTime + toffset}
                    fid2mouth[str(faceid)] = config
                else:
                    if config["min"] > mouth:
                        config["min"] = mouth
                    if config["max"] < mouth:
                        config["max"] = mouth
        
        toffset += 0.5
        count -= 1
        #find best match
    found = None
    for key, value in fid2mouth.items():
        if found is None:
            found = value
        else:
            if found["max"] - found["min"] + found["max"] < value["max"] - value["min"] + value["max"]:
                found = value
    if found is None:
        return None
    face = faceList[found["id"]]

    clip.save_frame(framedir, t=found["time"])
    fimg = cv2.cvtColor(cv2.imread(framedir), cv2.COLOR_BGR2RGB)  
    x,y,w,h = found["x"], found["y"], found["w"], found["h"]
    detected_face = fimg[int(y):int(y+h), int(x):int(x+w)] 
    obj = DeepFace.analyze(img_path=fimg, actions=['emotion'], enforce_detection=False)
    if len(obj) > 0: 
        face["emotion"] = obj[0]["dominant_emotion"]
    return face
    #return -1 if  found is None else found["id"]
    


project_json_file = os.path.join(args.filename, "project.json")
#load from disk,  just for testing,  file may be modified by UI
project = json.load(open(project_json_file, ))
speakers = project.get("speakers") if project.get(
    "speakers") else []
default_speaker = project.get("speaker")
regroupList = project.get("captions") 
finalcount = len(regroupList)

activeSpeakers = []
for i in range(finalcount):
    agroup = regroupList[i]
    print(agroup)
    face = detectSpeaker(agroup["start"], agroup['end'])
    if face is None:
        continue
    #face = faceList[faceid]
    faceid = face["faceid"]
    if face["gender"] == '':
        obj = DeepFace.analyze(img_path=face["img"], actions=[
            'gender'], enforce_detection=False)  # , 'emotion'])
        if len(obj) > 0:
            face["gender"] = "Female" if obj[0]["gender"]["Woman"] > obj[0]["gender"]["Man"] else "Male"
    agroup["speakerId"] = faceid  
    agroup["emotion"] = face["emotion"] 
    ## pick / filter all speakers used
    found = False
    for sp in activeSpeakers:
        if sp["id"] == faceid:
            found = True
            break
    if not found:
        activeSpeakers.append({"id": faceid, "gender": face["gender"], "fromFaceUI":True})
        imgfile = os.path.join(facesdir, str(faceid) + ".png")
        cv2.imwrite(imgfile, face["img"])

project["speakers"] = activeSpeakers

 

project_json = json.dumps(project, indent=4)
# Writing to whisper_json_object.json
project_json_file = os.path.join(args.filename, "project.json")
with open(project_json_file, "w") as outfile:
    outfile.write(project_json)
 
 


""" 
use mediapipe face_mesh
"""
""" 
mp_face_mesh = mediapipe.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
landmarks = results.multi_face_landmarks[0]

face_oval = mp_face_mesh.FACEMESH_FACE_OVAL

df = pd.DataFrame(list(face_oval), columns= ["p1", "p2"])
routes_idx = []

p1 = df.iloc[0]["p1"]
p2 = df.iloc[0]["p2"]

for i in range(0, df.shape[0]):

    #print(p1, p2)

    obj = df[df["p1"] == p2]
    p1 = obj["p1"].values[0]
    p2 = obj["p2"].values[0]

    route_idx = []
    route_idx.append(p1)
    route_idx.append(p2)
    routes_idx.append(route_idx)

# -------------------------------

for route_idx in routes_idx:
    print(
        f"Draw a line between {route_idx[0]}th landmark point to {route_idx[1]}th landmark point")


"""

