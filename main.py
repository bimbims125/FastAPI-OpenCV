import uvicorn
import os
import cv2
import base64

from fastapi import FastAPI, File, UploadFile

app = FastAPI(title='Face Detection API')

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

cascade_path = os.path.join(BASE_DIR, 'haardcascade/haarcascade_frontalface_default.xml')
app_path = os.path.join(BASE_DIR)

@app.post("/upload-face")
async def upload_face(file: UploadFile = File(...)):
  photo = await file.read()
  b64_string = base64.b64encode(photo)
  b64_decode = base64.b64decode(b64_string)
  with open(f"{app_path}/media/temp.jpg", "wb") as f:
    f.write(b64_decode)

  face_cascade = cv2.CascadeClassifier(cascade_path)
  image = cv2.imread(f"{app_path}/media/temp.jpg")
  grayscaled = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  detected_faces = face_cascade.detectMultiScale(
    grayscaled,
    1.3,
    minNeighbors=5,
    minSize=(30,30)
  )

  for (x, y, w, h) in detected_faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (255,0,0),2)
  # cv2.imwrite(f"{app_path}/media/temps.jpg",
  return {'total_faces': len(detected_faces), 'message':f"{len(detected_faces)} detected!"} if len(detected_faces) > 1 else {'total_faces': 0, 'message': 'No face detected'}





if  __name__ == '__main__':
  uvicorn.run('main:app', reload=True)
