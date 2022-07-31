import cv2
import os
from tensorflow.keras.utils import normalize
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation , Dropout, Flatten, Dense
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix , classification_report
from mlxtend.plotting import plot_confusion_matrix

image_directory="C:/Users/gowth/OneDrive/Desktop/Project_Work/dataset"

no_tumor_images=os.listdir(image_directory + "/no")
yes_tumor_images=os.listdir(image_directory + "/yes")
dataset=[]
label=[]

Input_size=64

#print(no_tumor_images)

for i,image_name in enumerate(no_tumor_images):
    if(image_name.split(".")[1] == "jpg"):
        image=cv2.imread(image_directory+"/no/"+image_name)
        image=Image.fromarray(image,"RGB")
        image=image.resize((Input_size,Input_size))
        dataset.append(np.array(image))
        label.append(0)
    


for i,image_name in enumerate(yes_tumor_images):
    if(image_name.split(".")[1] == "jpg"):
        image=cv2.imread(image_directory+"/yes/"+image_name)
        image=Image.fromarray(image,"RGB")
        image=image.resize((Input_size,Input_size))
        dataset.append(np.array(image))
        label.append(1)

#print(len(dataset))
#print(len(label))

dataset=np.array(dataset)
label=np.array(label)

x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size=0.2, random_state=0)

#Reshape=(n, image_width, image_height, n_channel)

print(x_train.shape)
print(y_train.shape)

print(x_test.shape)
print(y_test.shape)

x_train=normalize(x_train,axis=1)
x_test=normalize(x_test,axis=1)

y_train=to_categorical(y_train, num_classes=2)
y_test=to_categorical(y_test, num_classes=2)


#Model Building

model=Sequential()

model.add(Conv2D(32,(3,3),input_shape=(Input_size,Input_size,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,(3,3),kernel_initializer="he_uniform"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32,(3,3),kernel_initializer="he_uniform"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation("softmax"))
print(" ")
model.summary()

#Binary CrossEntropy=1,sigmoid
#Cross Entryopy=2,softmax

model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])

result = model.fit(x_train,y_train,batch_size=16, verbose=1, epochs=30,validation_data=(x_test,y_test),shuffle=False)

model.save("BrainTumor10EpochsCategorical.h5")

batch_size = 32

Y_pred = model.predict_generator(x_test , steps=4500 / batch_size+1)
y_pred = np.argmax(Y_pred, axis=1)

print("===============================")
print(" ")
print('Confusion Matrix')
print(" ")
cm = confusion_matrix(y_test.argmax(axis=1), y_pred)
print(cm)

fig,ax = plot_confusion_matrix(conf_mat = cm ,
                                show_absolute = True,
                                show_normed = True,
                                colorbar = True,
                                cmap = 'Dark2')
plt.show()

print(" ")
print("================================")

print(" ")

print("===============================")
print(" ")
print('Classification Report')
print(" ")

report = classification_report(y_test.argmax(axis=1), y_pred, target_names=['yes','no'])
print(report) 
print(" ")
print("================================")


# Printing the graphs 

plt.plot(result.history['loss'], label = 'Train Loss')
plt.plot(result.history['val_loss'], label = 'Val Loss')
plt.legend()
plt.show()

plt.plot(result.history['accuracy'], label = 'Train Acc')
plt.plot(result.history['val_accuracy'], label = 'Val Acc')
plt.legend()
plt.show()


