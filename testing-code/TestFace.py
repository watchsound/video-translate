"""
only work in window ,,,,because virtual box issue.
"""
 
#from mtcnn import MTCNN
import pandas as pd
import matplotlib.pyplot as plt
#from deepface import DeepFace
import cv2

#import mediapipe
 

"""
use cv2, refer: https://www.datacamp.com/tutorial/face-detection-python-opencv
"""
img = cv2.imread("faces.png") 
#fig = plt.figure(figsize=(8, 8))
#plt.axis('off')
#plt.imshow(img[:, :, ::-1])
#plt.show()  
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
#https://github.com/opencv/opencv/tree/master/data/haarcascades
face_classifier = cv2.CascadeClassifier(
   # cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
   cv2.data.haarcascades + "haarcascade_upperbody.xml"
) 
face = face_classifier.detectMultiScale(
    gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(10, 10)
)
print(" faces found = " + str(len(face)))

for (x, y, w, h) in face:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
 
plt.figure(figsize=(20, 10))
plt.imshow(img_rgb)
plt.axis('off')
plt.show()



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


"""
use MTCNN
"""

""" 
img = cv2.cvtColor(cv2.imread(
    "faces.png"), cv2.COLOR_BGR2RGB)


detector = MTCNN() 
detections = detector.detect_faces(img) 
embeddings = []
for detection in detections:
   confidence = detection["confidence"]
   if confidence > 0.90:
      x, y, w, h = detection["box"]
     # print(detection["box"])
      detected_face = img[int(y):int(y+h), int(x):int(x+w)] 
      embedding = DeepFace.represent(
          detected_face, model_name='Facenet', enforce_detection=False)
      embeddings.append(embedding)
"""
