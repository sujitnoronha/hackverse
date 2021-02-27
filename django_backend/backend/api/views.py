from django.shortcuts import render
from rest_framework.response import Response
from rest_framework.decorators import api_view
from rest_framework.views import APIView
from rest_framework.decorators import authentication_classes, permission_classes
from rest_framework.permissions import AllowAny, IsAuthenticatedOrReadOnly
from rest_framework import status


from api.models import analytics,person,plocation
from api.serializers import anSerializer


import cv2
import base64
import jsonify
import PIL
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from imageio import imread
from skimage.transform import resize
from keras.models import load_model
import time
import keras.backend.tensorflow_backend as K
K.clear_session()

cascade_path = os.path.join(os.getcwd(),'api\model\cv2\haarcascade_frontalface_alt2.xml')

image_dir_basepath = './api/data/images/'
names = ['Dylan','Akshaye','LarryPage', 'MarkZuckerberg', 'BillGates','TomCruise','SujitNoronha','Pushpak','Unknown']
image_size = 160

model_path = os.path.join(os.getcwd(),'api\model\keras\model\sfacenet_keras.h5')

model = load_model(model_path)

def prewhiten(x):
    if x.ndim == 4:
        axis = (1, 2, 3)
        size = x[0].size
    elif x.ndim == 3:
        axis = (0, 1, 2)
        size = x.size
    else:
        raise ValueError('Dimension should be 3 or 4')

    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    std_adj = np.maximum(std, 1.0/np.sqrt(size))
    y = (x - mean) / std_adj
    return y

def l2_normalize(x, axis=-1, epsilon=1e-10):
    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
    return output


def load_and_align_images(filepaths, margin):
    cascade = cv2.CascadeClassifier(cascade_path)
    
    aligned_images = []
    for filepath in filepaths:
        img = imread(filepath)

        faces = cascade.detectMultiScale(img,
                                         scaleFactor=1.1,
                                         minNeighbors=4)
        for i in range(len(faces)):
            (x, y, w, h) = faces[i]
            cropped = img[y-margin//2:y+h+margin//2,
                        x-margin//2:x+w+margin//2, :]
            aligned = resize(cropped, (image_size, image_size), mode='reflect')
            aligned_images.append(aligned)    
    return np.array(aligned_images)

def calc_embs(filepaths, margin=10, batch_size=1):
    aligned_images = prewhiten(load_and_align_images(filepaths, margin))
    pd = []
    for start in range(0, len(aligned_images), batch_size):
        pd.append(model.predict_on_batch(aligned_images[start:start+batch_size]))
    embs = l2_normalize(np.concatenate(pd))

    return embs

def train(dir_basepath, names, max_num_img=50):
    labels = []
    embs = []
    for name in names:
        dirpath = os.path.abspath(dir_basepath + name)
        filepaths = [os.path.join(dirpath, f) for f in os.listdir(dirpath)][:max_num_img]
        embs_ = calc_embs(filepaths)    
        labels.extend([name] * len(embs_))
        embs.append(embs_)
        
    embs = np.concatenate(embs)
    le = LabelEncoder().fit(labels)
    y = le.transform(labels)
    clf = SVC(kernel='linear', probability=True).fit(embs, y)
    return le, clf

def infer(le, clf, filepaths):
    embs = calc_embs(filepaths)
    pred = le.inverse_transform(clf.predict(embs))
    return pred
print("training begins")

le, clf = train(image_dir_basepath, names)
print('training ends')






# Create your views here.
@api_view(['GET'])
def apiOverview(request,*args,**kwargs):
    api_urls = {
        'sceneapi': 'scene/',
    }
    return Response(api_urls)

@permission_classes((AllowAny, ))
class AnInfo(APIView):
    def get(self,request,*args, **kwargs):
        info = analytics.objects.all().order_by('-id')
        print(info)
        serializer = anSerializer(info, many=True)
        return Response(serializer.data)

    def post(self,request,*args,**kwargs):
        serializer = anSerializer(data = request.data)
        if serializer.is_valid():
            serializer.save()
        return Response(serializer.data,status= status.HTTP_201_CREATED)




def dashboard(request):
    stats = analytics.objects.all().order_by('-id')
    last_action = stats[0]
    pcount = [int(i.peoplecount) for i in stats][:6]
    context = {
        "stat": stats,
        "last": last_action,
        "pcount": list(reversed(pcount)),
    }
    return render(request, 'api/dashboard.html',context)


def facedetect(request, id):
    scene = analytics.objects.get(id = id)
    print(scene.sceneimage.url, scene.sceneimage)
    test_filepaths = [scene.sceneimage]
    preds = infer(le, clf, test_filepaths)
    li = []
    for i in preds:
        if i != "Unknown":
            li.append(i)
            try:
                p = person.objects.get(name= i)
            except:
                p = person.objects.create(name=i)
            loc = plocation.objects.create(name = p,location=scene.location)
    
    context = {
        "scene": scene,
        "pred":li,
        "count": len(preds)
    }
    return render(request, 'api/facedetect.html',context)


def lastimage(request):
    images = analytics.objects.all().order_by('-id')
    context = {
        "stat": images,
    }
    return render(request, 'api/cctvshots.html',context)        


def pdetails(request, id):
    per = person.objects.get(id = id)
    return render(request,'api/pdetails.html',{"person": per})

