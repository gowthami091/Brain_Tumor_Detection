import numpy as np
from keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input

model=load_model("BrainTumor10EpochsCategorical.h5")

test_file_path = "C:/Users/gowth/OneDrive/Desktop/Project_Work/pred/pred0.jpg"


img = image.load_img(test_file_path,target_size = (64, 64))

test_img = image.img_to_array(img)

test_img

test_img.shape

test_img = np.expand_dims(test_img, axis = 0)

test_img

test_img = preprocess_input(test_img)

test_img

predict_test = model.predict(test_img)

print(" ")
print("=====  Result is given below  =====")

print(np.argmax(predict_test))
if np.argmax(predict_test) == 0:
  print("This Patient has no Brain Tumour")
  
else:
  print("This Patient has Brain Tumour")

print(" ")


