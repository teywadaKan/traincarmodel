import cv2
import os
import base64
import requests
import pickle

url = "http://localhost:8080/api/genhog"

def img2vec(img):
    v, buffer = cv2.imencode(".jpg", img)
    img_str = base64.b64encode(buffer)
    data = "image data,"+str.split(str(img_str),"'")[1]
    data = data.split(",")[1]
    response = requests.get(url, json={"img":data})
    return response.json()["Hog"]

list_data = []
path = "Cars Dataset-20230818T161925Z-001\\Cars Dataset\\train"
# path = "Cars Dataset-20230818T161925Z-001\\Cars Dataset\\test"
i=0
for subfolder in os.listdir(path):
    # print(subfolder)
    for f in os.listdir(os.path.join(path,subfolder)):
        img = cv2.imread(os.path.join(path,subfolder)+"\\"+f)
        img_hog = img2vec(img)
        img_hog.append(i)
        list_data.append(img_hog)
        print(i,os.path.join(path,subfolder)+"\\"+f)
        # print(os.path.join(path,subfolder)+"\\"+f)
    i=i+1

write_path = "carTranModel.pkl"
pickle.dump(list_data,open(write_path,"wb"))
# write_path = "carTestModel.pkl"
# pickle.dump(list_data,open(write_path,"wb"))
print("data preparation is done")