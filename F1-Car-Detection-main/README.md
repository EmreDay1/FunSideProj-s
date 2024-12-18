# F1-Car-Detection

This is an AI model which is built on an image dataset of F1 cars mosltly from the 2020 season, but also  including older images for teams such as Mercedes, Ferrari and Red Bull

## Explanation of the models code 


## Classes of the model

#### 0: ALpha Tauri

#### 1: Ferrari

#### 2: Mclaren

#### 3: Mercedes AMG Petronas

#### 4: Racing Point

#### 5: Red Bull Racing

#### 6: Renault(modern, 2016-2021)

#### 7: Williams F1

## Example usage: RB Racing

```python

import cv2
import numpy as np
import tensorflow as tf


model = tf.keras.models.load_model('f1_car_detection_model.h5')


def preprocess_frame(frame, target_size=(224, 224)):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, target_size)
    frame = frame / 255.0
    return np.expand_dims(frame, axis=0)


video_path = '/path/to/video/file' 
video_capture = cv2.VideoCapture(video_path)


fps = video_capture.get(cv2.CAP_PROP_FPS)
width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))


output_video_path = 'output_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

while True:

    ret, frame = video_capture.read()

    
    if not ret:
        break

    input_data = preprocess_frame(frame)


    predictions = model.predict(input_data)
    predicted_class = np.argmax(predictions)


    if predicted_class == 5: 
        label = 'Red Bull Racing F1 car'


        h, w, _ = frame.shape
        x, y, w, h = int(0.1 * w), int(0.1 * h), int(0.8 * w), int(0.8 * h)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        cv2.putText(frame, f'Detected: {label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    

    output_video.write(frame)


video_capture.release()
output_video.release()

print(f"Processed video saved to {output_video_path}")

```

This is a code to detect Red Bull Racing Cars using the model. In this code when the car is detected a label shows up on the page and a bounding box is drawn around the car.

The code was tested on Checo Perez's New York drive with the RB racing car. 

Note: The model is much more sucsessfull at detecting cars up close due to the nature of the datasets.

### Result



https://github.com/EmreDay1/F1-Car-Detection/assets/120194760/0043dd34-35b6-4457-9918-3c3054eb85e8


## Example usage: Mercedes AMG Petronas 

```python

import cv2
import numpy as np
import tensorflow as tf


model = tf.keras.models.load_model('f1_car_detection_model.h5')


def preprocess_frame(frame, target_size=(224, 224)):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, target_size)
    frame = frame / 255.0
    return np.expand_dims(frame, axis=0)


video_path = '/path/to/video/file'
video_capture = cv2.VideoCapture(video_path)


fps = video_capture.get(cv2.CAP_PROP_FPS)
width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))


output_video_path = 'output_video_mercedes.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

while True:

    ret, frame = video_capture.read()

    
    if not ret:
        break

    input_data = preprocess_frame(frame)


    predictions = model.predict(input_data)
    predicted_class = np.argmax(predictions)


    if predicted_class == 3:  
        label = 'Mercedes F1 car'


        h, w, _ = frame.shape
        x, y, w, h = int(0.1 * w), int(0.1 * h), int(0.8 * w), int(0.8 * h)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        cv2.putText(frame, f'Detected: {label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    

    output_video.write(frame)


video_capture.release()
output_video.release()

print(f"Processed video saved to {output_video_path}")
```

Here the same thing is done for the Mercedes F1 cars

### Result

https://github.com/EmreDay1/F1-Car-Detection/assets/120194760/9fea8a0c-14cb-44db-922e-a0af8d83434e



