# 📷Visual Object Recognition Program

- Project Duration : 2023.01 ~ 2023.02
- Project Description : Embedded system designed for recognizing and defining objects captured by a camera using deep learning-trained image data. (data : Imagenet_2017).

---
Constructed both the embedded system and the computational pipeline, developing a Python algorithm for image classification in a TensorFlow session, and interpreting results in LabVIEW program designed for image reception.

---
## LabVIEW
- Out put : Visual Object Recognition
- When the camera captures an object, the program will detect the object and classify it based on the training image data.
- ![image](https://github.com/yelangsung/Visual-Object-Recognition-Program/assets/113841190/76a82dc4-9b0c-469e-98b1-75eb42a5ab03)
- ** Reference : https://www.youtube.com/watch?v=szxegYAev7M&ab_channel=NationalInstrumentsKoreaAE
- -> Referring to VI Code, Writte and Complete Python Algorithm

### Step1. Connect the Camera
- ![image](https://github.com/yelangsung/Visual-Object-Recognition-Program/assets/113841190/fe7017a5-3448-472a-87bb-cae2f2a0796b)

### Step2. while obtaining images(Default)
- ![image](https://github.com/yelangsung/Visual-Object-Recognition-Program/assets/113841190/ecb4f1ff-6997-4873-bafd-0a3c16ea9aad)

### Step3. sub Vi

- converting the data type, which is used to store images and is different from the data type handled in Python, into a data type that can be used for both.
- ![image](https://github.com/yelangsung/Visual-Object-Recognition-Program/assets/113841190/9202b370-9efa-4ca2-b6a5-da78b03fd32e)

### Step4. Object Recognition and Classification using Python Algorithm
- ![image](https://github.com/yelangsung/Visual-Object-Recognition-Program/assets/113841190/769075a1-8839-4e03-a887-6a3858d18aef)
