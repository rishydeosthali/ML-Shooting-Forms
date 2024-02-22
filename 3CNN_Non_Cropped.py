from google.colab import drive
drive.mount('/content/drive',force_remount=True)

!pip install mediapipe opencv-python
from google.colab.patches import cv2_imshow
from sklearn import preprocessing
import cv2
import mediapipe as mp
import numpy as np
import glob
import numpy as np




mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
training_data = []
numb = 0

def take_coordinates(landmarks,array_of_landmarks):
    frame_data = []
    for i in range(32):
        #array_of_landmarks.append(landmarks[i].x)
        #array_of_landmarks.append(landmarks[i].y)
        frame_data.append(landmarks[i].x)
        frame_data.append(landmarks[i].y)
    array_of_landmarks.append(frame_data)

def calculate_angle(a,b,c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    ##radians = np.arctan2(c[0] - b[0], c[1] - b[1]) - np.arctan2(a[0] - b[0], a[1] - b[1])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0:
        angle = 360-angle
    return angle


good_shot_directory = '/content/drive/MyDrive/Good_Shots/'
bad_shot_directory = '/content/drive/MyDrive/Bad_Shots/'

video_extensions = ['*.mp4', '*.avi', '*.mov']

# Search for video files in the directory using glob
good_shots = []
bad_shots = []
for extension in video_extensions:
    good_shots.extend(glob.glob(good_shot_directory + extension))
    bad_shots.extend(glob.glob(bad_shot_directory + extension))




final_data = good_shots + bad_shots
labels = []
for i in range(len(final_data)):
    if(i<=len(good_shots)-1):
        labels.append(1) ## 1 in labels is good shot and 0 in labels is a bad shot
    else:
        labels.append(0)



!pip install remotezip tqdm opencv-python einops
# Install TensorFlow 2.10
!pip install tensorflow
import tqdm
import random
import pathlib
import itertools
import collections

import cv2
import einops
import numpy as np
import remotezip as rz
import seaborn as sns
import matplotlib.pyplot as plt

import tensorflow as tf
import keras
from keras import layers

def format_frames(frame,output_size):
  frame = tf.image.convert_image_dtype(frame, tf.float32)
  frame = tf.image.resize_with_pad(frame,*output_size)
  return frame

def frames_from_video_file(video_path, n_frames, output_size, frame_step):
  result = []
  src = cv2.VideoCapture(str(video_path))
  video_length = src.get(cv2.CAP_PROP_FRAME_COUNT)
  #print(video_length)
  need_length = 1 + (n_frames-1)*frame_step
  if(need_length > video_length):
    start = 0
  else:
    max_start = video_length - need_length
    start = random.randint(0, max_start + 1)
  src.set(cv2.CAP_PROP_POS_FRAMES, start)
  ret, frame = src.read()
  result.append(format_frames(frame, output_size))
  #print(result)
  for _ in range(n_frames - 1):
    for _ in range(frame_step):
      ret, frame = src.read()
    if ret:
      frame = format_frames(frame, output_size)
      result.append(frame)
    else:
      result.append(np.zeros_like(result[0]))
  src.release()
  result = np.array(result)[..., [2, 1, 0]]

  return result



HEIGHT = 100
WIDTH = 100
class Conv2Plus1D(keras.layers.Layer):
  def __init__(self, filters, kernel_size, padding):
    """
      A sequence of convolutional layers that first apply the convolution operation over the
      spatial dimensions, and then the temporal dimension.
    """
    super().__init__()
    self.seq = keras.Sequential([
        # Spatial decomposition
        layers.Conv3D(filters=filters,
                      kernel_size=(1, kernel_size[1], kernel_size[2]),
                      padding=padding),
        # Temporal decomposition
        layers.Conv3D(filters=filters,
                      kernel_size=(kernel_size[0], 1, 1),
                      padding=padding)
        ])

  def call(self, x):
    return self.seq(x)

class ResidualMain(keras.layers.Layer):
  """
    Residual block of the model with convolution, layer normalization, and the
    activation function, ReLU.
  """
  def __init__(self, filters, kernel_size):
    super().__init__()
    self.seq = keras.Sequential([
        Conv2Plus1D(filters=filters,
                    kernel_size=kernel_size,
                    padding='same'),
        layers.LayerNormalization(),
        layers.ReLU(),
        Conv2Plus1D(filters=filters,
                    kernel_size=kernel_size,
                    padding='same'),
        layers.LayerNormalization()
    ])

  def call(self, x):
    return self.seq(x)

class Project(keras.layers.Layer):
  """
    Project certain dimensions of the tensor as the data is passed through different
    sized filters and downsampled.
  """
  def __init__(self, units):
    super().__init__()
    self.seq = keras.Sequential([
        layers.Dense(units),
        layers.LayerNormalization()
    ])

  def call(self, x):
    return self.seq(x)

class ResizeVideo(keras.layers.Layer):
  def __init__(self, height, width):
    super().__init__()
    self.height = height
    self.width = width
    self.resizing_layer = layers.Resizing(self.height, self.width)

  def call(self, video):
    """
      Use the einops library to resize the tensor.

      Args:
        video: Tensor representation of the video, in the form of a set of frames.

      Return:
        A downsampled size of the video according to the new height and width it should be resized to.
    """
    # b stands for batch size, t stands for time, h stands for height,
    # w stands for width, and c stands for the number of channels.
    old_shape = einops.parse_shape(video, 'b t h w c')
    images = einops.rearrange(video, 'b t h w c -> (b t) h w c')
    images = self.resizing_layer(images)
    videos = einops.rearrange(
        images, '(b t) h w c -> b t h w c',
        t = old_shape['t'])
    return videos

def add_residual_block(input, filters, kernel_size):
  """
    Add residual blocks to the model. If the last dimensions of the input data
    and filter size does not match, project it such that last dimension matches.
  """
  out = ResidualMain(filters,
                     kernel_size)(input)

  res = input
  # Using the Keras functional APIs, project the last dimension of the tensor to
  # match the new filter size
  if out.shape[-1] != input.shape[-1]:
    res = Project(out.shape[-1])(res)

  return layers.add([res, out])
  
  
  from sklearn.model_selection import train_test_split
  video_data = []
  labels = np.array(labels)
  
  #tf.TensorSpec(shape = (None, None, None, 3), dtype = tf.float32)
  for x in final_data:
    #y = convert_frame(x, 40, (244,100), 2)
    y = frames_from_video_file(x, 40, (100,100),1)
  
    video_data.append(y)
  X = tf.stack(video_data, axis=0)
  num = X.shape[0]
  all_indices = np.array(list(range(num)))
  train_ind, test_ind = train_test_split(all_indices, test_size=0.3)
  train_ind = list(train_ind)
  X_train = tf.gather(X, train_ind, axis = 0)
  X_test = tf.gather(X, test_ind, axis = 0)
  y_train = labels[train_ind]
  y_test = labels[test_ind]
  
  
  
  
  
  
  
  #X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)
  input_shape = (None, 40, HEIGHT, WIDTH, 3)
  input = layers.Input(shape=(input_shape[1:]))
  x = input
  
  x = Conv2Plus1D(filters=16, kernel_size=(3, 7, 7), padding='same')(x)
  x = layers.BatchNormalization()(x)
  x = layers.ReLU()(x)
  x = ResizeVideo(HEIGHT // 2, WIDTH // 2)(x)
  
  # Block 1
  x = add_residual_block(x, 16, (3, 3, 3))
  x = ResizeVideo(HEIGHT // 4, WIDTH // 4)(x)
  
  # Block 2
  x = add_residual_block(x, 32, (3, 3, 3))
  x = ResizeVideo(HEIGHT // 8, WIDTH // 8)(x)
  
  # Block 3
  x = add_residual_block(x, 64, (3, 3, 3))
  x = ResizeVideo(HEIGHT // 16, WIDTH // 16)(x)
  
  # Block 4
  x = add_residual_block(x, 128, (3, 3, 3))
  
  x = layers.GlobalAveragePooling3D()(x)
  x = layers.Flatten()(x)
  x = layers.Dense(10)(x)
  
  model = keras.Model(input, x)
  
  model.build((None, 40, HEIGHT, WIDTH, 3))
keras.utils.plot_model(model, expand_nested=True, dpi=60, show_shapes=True)


model.compile(loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer = keras.optimizers.Adam(learning_rate = 0.0001),
              metrics = ['accuracy'])
history = model.fit(X_train,y_train,
                    epochs = 12,
                     validation_data = (X_test,y_test))



from google.colab.patches import cv2_imshow
from sklearn import preprocessing
import cv2
import mediapipe as mp
import numpy as np
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.preprocessing.sequence import pad_sequences  # Add this import


def create_NN_data():
  mp_drawing = mp.solutions.drawing_utils
  mp_pose = mp.solutions.pose
  training_data = []
  numb = 0

  def take_coordinates(landmarks,array_of_landmarks):
      frame_data = []
      for i in range(32):
          #array_of_landmarks.append(landmarks[i].x)
          #array_of_landmarks.append(landmarks[i].y)
          frame_data.append(landmarks[i].x)
          frame_data.append(landmarks[i].y)
      array_of_landmarks.append(frame_data)

  def calculate_angle(a,b,c):
      a = np.array(a)
      b = np.array(b)
      c = np.array(c)
      radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
      ##radians = np.arctan2(c[0] - b[0], c[1] - b[1]) - np.arctan2(a[0] - b[0], a[1] - b[1])
      angle = np.abs(radians*180.0/np.pi)
      if angle > 180.0:
          angle = 360-angle
      return angle


  good_shot_directory = '/content/drive/MyDrive/Good_Shots/'
  bad_shot_directory = '/content/drive/MyDrive/Bad_Shots/'

  video_extensions = ['*.mp4', '*.avi', '*.mov']

  # Search for video files in the directory using glob
  good_shots = []
  bad_shots = []
  for extension in video_extensions:
      good_shots.extend(glob.glob(good_shot_directory + extension))
      bad_shots.extend(glob.glob(bad_shot_directory + extension))




  final_data = good_shots + bad_shots
  labels = []
  for i in range(len(final_data)):
      if(i<=len(good_shots)-1):
          labels.append(1) ## 1 in labels is good shot and 0 in labels is a bad shot
      else:
          labels.append(0)


  for x in range(len(final_data)): #len(final_data)
      array_of_landmarks = []
      cap = cv2.VideoCapture(final_data[x]) ## final_data[1]
      ## number within determines webcam
      with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
          while cap.isOpened():

            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            annotated_frame = frame.copy()
            if results.pose_landmarks is not None:
                landmarks = results.pose_landmarks.landmark
            take_coordinates(landmarks,array_of_landmarks)





            mp_drawing.draw_landmarks(annotated_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

          cap.release()
          cv2.destroyAllWindows()
          training_data.append(array_of_landmarks)
  flattened_videos = [sum(frames,[]) for frames in training_data]
  max_sequence_length = max(len(arr) for arr in flattened_videos)
  X_padded = pad_sequences(flattened_videos, maxlen=max_sequence_length, padding='post', dtype='int')
  X = np.array(X_padded)
  normalized_data_list = []
  scaler = StandardScaler()

  for arr in X:
    arr = arr.reshape(-1, 1)
    normalized_arr = scaler.fit_transform(arr).flatten()
    normalized_data_list.append(normalized_arr)

  X_normalized = np.array(normalized_data_list)
  labels2 = np.array(labels)
  X_train, X_test, y_train, y_test = train_test_split(X_normalized, labels2, test_size=0.3, random_state=42)
  return X_train,X_test,y_train,y_test






