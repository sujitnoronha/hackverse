<p align="center">
  <h1 align="center">Hackverse-2.0</h1>
  <h1 align="center">RIM.ai - An Ai based surveillance system</h1>
</p>

<p align="center">
  <img src="https://github.com/sujitnoronha/researchinmotion/blob/master/images/demo_vid.png?raw=true" max-width="800px" alt="normal"><br>
  <img src="https://github.com/sujitnoronha/researchinmotion/blob/master/images/demo_vid2.png?raw=true" max-width="800px" alt="shooting">
</p>

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Contributors</a>
    </li>
  </ol>
</details>

## About The Project


Imagine having to never monitor surveillance videos and find people manually in those videos. Our system bridges the gap between traditional CCTV systems and state-of-the-art Computer Vision techniques to help eliminate the redundant and inefficient task performed by opersonnel. 
</br>



### Built With

* [TensorFlow](https://www.tensorflow.org/)
* [OpenVino](https://docs.openvinotoolkit.org/latest/index.html)
* [Yolo](https://pjreddie.com/darknet/yolo/)
* [Python](https://www.python.org/)
* [Django](https://www.djangoproject.com/)
* [OpenCv](https://pypi.org/project/opencv-python/)
* [Keras-FaceNet](https://github.com/nyoki-mtl/keras-facenet)


## Getting Started

### Prerequisites
You should have the specific OpenVino version installed (openvino 2020.4.287).
* [OpenVino](https://docs.openvinotoolkit.org/2020.4/index.html)
* [Python 3.7](https://www.python.org/downloads/release/python-370/)


### Add necessary folders and files to the project
* [DriveLink](https://drive.google.com/drive/folders/1up-4amoZv49FnDrswRbb8pQlAEKAlxJ7)
add these folders to the main directory of the project 
* model/
* vid/
* data/
* create a model folder in the django_backend/backend/api/ folder add the facemodel contents in the folder 

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/sujitnoronha/researchinmotion.git
   ```
2. Install python packages
   ```sh
   pip install -r requirements.txt
   ```
3. Run your project.
   ```sh
   python vino.inf.py
   ```



### Running the Server
 1. get to the backend directory
 ```sh
 cd django_backend/backend 
 ```
2. Install python packages
   ```sh
   pip install -r requirements.txt
   ```


## Contributors
Developed by <a href="https://github.com/sujitnoronha">Sujit Noronha </a> , <a href="https://github.com/dylandsouza00"> Dylan Dsouza</a> , <a href="https://github.com/FrankyPinto">Franky Pinto</a>.

## License
[MIT](https://choosealicense.com/licenses/mit/)