from google.colab import drive
drive.mount('/content/drive')

!pip install mediapipe opencv-python
from google.colab.patches import cv2_imshow
import cv2
import mediapipe as mp
import numpy as np
import glob

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def take_coordinates(landmarks, array_of_landmarks):
    frame_data = []
    for i in range(32):
        if i == 0:
            continue
        frame_data.append(landmarks[i].x)
        frame_data.append(landmarks[i].y)
        frame_data.append(landmarks[i].z)
    array_of_landmarks.append(frame_data)

def draw_skeleton(frame, landmarks):
    mp_drawing.draw_landmarks(frame, landmarks, mp_pose.POSE_CONNECTIONS)

good_shot_directory = '/content/drive/MyDrive/Good_Shots2/'
bad_shot_directory = '/content/drive/MyDrive/Bad_Shots2/'

video_extensions = ['*.mp4', '*.avi', '*.mov']

final_data = good_shots + bad_shots

for x in range(len(final_data)):
    array_of_landmarks = []
    cap = cv2.VideoCapture(final_data[x])

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
                take_coordinates(landmarks, array_of_landmarks)
                # Draw skeleton on a blank image
                blank_image = np.zeros_like(frame)
                draw_skeleton(blank_image, results.pose_landmarks)
                cv2_imshow(blank_image)  # Show the skeletonized portion

        cap.release()
        cv2.destroyAllWindows()
        training_data.append(array_of_landmarks)


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
        if(i == 0):
          continue;
        nose_landmark = landmarks[0]
        nose_x, nose_y, nose_z = nose_landmark.x, nose_landmark.y, nose_landmark.z
        frame_data.append(landmarks[i].x)
        frame_data.append(landmarks[i].y)
        frame_data.append(landmarks[i].z)
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


good_shot_directory = '/content/drive/MyDrive/Good_Shots2/'
bad_shot_directory = '/content/drive/MyDrive/Bad_Shots2/'

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
          # blank_image = np.zeros_like(frame)
          # draw_skeleton(blank_image, results.pose_landmarks)
          # cv2_imshow(blank_image)  # Show the skeletonized portion


        cap.release()
        cv2.destroyAllWindows()
        training_data.append(array_of_landmarks)




from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences


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



model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(max_sequence_length,)))
model.add(Dropout(.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(.3))
model.add(Dense(32, activation='relu'))
model.add(Dropout(.3))
model.add(Dense(1, activation='sigmoid'))




model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


model.fit(X_train, y_train, epochs=40, batch_size=32, validation_data = (X_test,y_test))


loss, accuracy = model.evaluate(X_test, y_test)
print("Test accuracy:", accuracy)


import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Get predictions for the test data
y_pred_probs = model.predict(X_test)
y_pred = np.round(y_pred_probs).flatten()

# Compute the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=['Predicted Optimal Form', 'Predicted Sub-Optimal Form'],
            yticklabels=['Actual Optimal Form', 'Predicted Optimal Form'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.regularizers import l2
import numpy as np

# ... (Loading and preprocessing your data)

flattened_videos = [sum(frames, []) for frames in training_data]
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

learning_rate = 0.001
num_hidden_layers = 3
hidden_units = [128, 64, 32]
activation_function = 'relu'
dropout_rate = .3
batch_size = 32
epochs = 20

# Create the model
model = Sequential()

# Add input layer
model.add(Dense(hidden_units[0], activation=activation_function, input_shape=(max_sequence_length,)))

# Add hidden layers
for i in range(1, num_hidden_layers):
    model.add(Dense(hidden_units[i], activation=activation_function, kernel_regularizer=l2(0.01)))  # Adding L2 regularization

# Add dropout layer
model.add(Dropout(dropout_rate))

# Add output layer
model.add(Dense(1, activation='sigmoid'))

# Compile the model with a custom optimizer (e.g., Adam with a specific learning rate)
custom_optimizer = Adam(learning_rate=learning_rate)
model.compile(optimizer=custom_optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Implement early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model with early stopping
history = model.fit(X_train, y_train, epochs=epochs,batch_size = batch_size,
                    validation_data=(X_test, y_test), callbacks=[early_stopping])

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print("Test accuracy:", accuracy)



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
max_sequence_length = max(len(arr) for arr in training_data)
X_padded = pad_sequences(training_data, maxlen=max_sequence_length, padding='post', dtype='int')

X = np.array(X_padded)

normalized_data_list = []
scaler = StandardScaler()
normalized_data_list = X.reshape(-1,X.shape[-1])
normalized_data_list = scaler.fit_transform(normalized_data_list)
normalized_data_list = normalized_data_list.reshape(X.shape)

X_normalized = np.array(normalized_data_list)
labels2 = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X_normalized, labels2, test_size=0.3, random_state=42)

model = Sequential()

# Replace Dense layers with LSTM layers
model.add(LSTM(128, activation='relu', input_shape=(X_normalized.shape[1],X_normalized.shape[2])))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=40, batch_size=20, validation_split=0.3)

loss, accuracy = model.evaluate(X_test, y_test)
print("Test accuracy:", accuracy)



import matplotlib.pyplot as plt

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=20, validation_data=(X_test, y_test))

# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()



import matplotlib.pyplot as plt

# Assuming you have 'labels', 'training_data', and 'flattened_videos' defined

# Define the joint number for the elbow (13 in this case)
elbow_joint = 13

# Extract and flatten x, y, and z values for the elbow joint
x_values = [data[3 * elbow_joint] for data in flattened_videos]
y_values = [data[3 * elbow_joint + 1] for data in flattened_videos]
z_values = [data[3 * elbow_joint + 2] for data in flattened_videos]

# Create three separate subplots for x, y, and z values with different colors for good and bad shots
plt.figure(figsize=(15, 5))

# Plot x values
plt.subplot(131)
for i, label in enumerate(labels):
    if label == 1:
        plt.scatter(i, x_values[i], c='blue', label='Good Shot')
    else:
        plt.scatter(i, x_values[i], c='red', label='Bad Shot')
plt.title('Left Elbow X Values')
good_shot_legend = plt.Line2D([0], [0], marker='o', color='w', label='Good Shot', markerfacecolor='blue', markersize=10)
bad_shot_legend = plt.Line2D([0], [0], marker='o', color='w', label='Bad Shot', markerfacecolor='red', markersize=10)
plt.legend(handles=[good_shot_legend, bad_shot_legend], loc='upper right')
# Plot y values
plt.subplot(132)
for i, label in enumerate(labels):
    if label == 1:
        plt.scatter(i, y_values[i], c='blue', label='Good Shot')
    else:
        plt.scatter(i, y_values[i], c='red', label='Bad Shot')
plt.title('Left Elbow Y Values')
good_shot_legend = plt.Line2D([0], [0], marker='o', color='w', label='Good Shot', markerfacecolor='blue', markersize=10)
bad_shot_legend = plt.Line2D([0], [0], marker='o', color='w', label='Bad Shot', markerfacecolor='red', markersize=10)
plt.legend(handles=[good_shot_legend, bad_shot_legend], loc='upper right')
# Plot z values
plt.subplot(133)
for i, label in enumerate(labels):
    if label == 1:
        plt.scatter(i, z_values[i], c='blue', label='Good Shot')
    else:
        plt.scatter(i, z_values[i], c='red', label='Bad Shot')
plt.title('Left Elbow Z Values')

# Create a single legend for all graphs
# Create a single legend for all graphs

good_shot_legend = plt.Line2D([0], [0], marker='o', color='w', label='Good Shot', markerfacecolor='blue', markersize=10)
bad_shot_legend = plt.Line2D([0], [0], marker='o', color='w', label='Bad Shot', markerfacecolor='red', markersize=10)
plt.legend(handles=[good_shot_legend, bad_shot_legend], loc='upper right')

plt.tight_layout()
plt.show()



import matplotlib.pyplot as plt

# Assuming you have 'labels', 'training_data', and 'flattened_videos' defined

# Define the joint number for the elbow (13 in this case)
for i in range(32):
  elbow_joint = i

  # Extract and flatten x, y, and z values for the elbow joint
  x_values = [data[3 * elbow_joint] for data in flattened_videos]
  y_values = [data[3 * elbow_joint + 1] for data in flattened_videos]
  z_values = [data[3 * elbow_joint + 2] for data in flattened_videos]

  # Create three separate subplots for x, y, and z values with different colors for good and bad shots
  plt.figure(figsize=(15, 5))

  # Plot x values
  plt.subplot(131)
  for i, label in enumerate(labels):
      if label == 1:
          plt.scatter(i, x_values[i], c='blue', label='Good Shot')
      else:
          plt.scatter(i, x_values[i], c='red', label='Bad Shot')
  plt.title('Right Elbow X Values')
  good_shot_legend = plt.Line2D([0], [0], marker='o', color='w', label='Good Shot', markerfacecolor='blue', markersize=10)
  bad_shot_legend = plt.Line2D([0], [0], marker='o', color='w', label='Bad Shot', markerfacecolor='red', markersize=10)
  plt.legend(handles=[good_shot_legend, bad_shot_legend], loc='upper right')
  # Plot y values
  plt.subplot(132)
  for i, label in enumerate(labels):
      if label == 1:
          plt.scatter(i, y_values[i], c='blue', label='Good Shot')
      else:
          plt.scatter(i, y_values[i], c='red', label='Bad Shot')
  plt.title('Right Elbow Y Values')
  good_shot_legend = plt.Line2D([0], [0], marker='o', color='w', label='Good Shot', markerfacecolor='blue', markersize=10)
  bad_shot_legend = plt.Line2D([0], [0], marker='o', color='w', label='Bad Shot', markerfacecolor='red', markersize=10)
  plt.legend(handles=[good_shot_legend, bad_shot_legend], loc='upper right')
  # Plot z values
  plt.subplot(133)
  for i, label in enumerate(labels):
      if label == 1:
          plt.scatter(i, z_values[i], c='blue', label='Good Shot')
      else:
          plt.scatter(i, z_values[i], c='red', label='Bad Shot')
  plt.title('Right Elbow Z Values')

  # Create a single legend for all graphs
  # Create a single legend for all graphs

  good_shot_legend = plt.Line2D([0], [0], marker='o', color='w', label='Good Shot', markerfacecolor='blue', markersize=10)
  bad_shot_legend = plt.Line2D([0], [0], marker='o', color='w', label='Bad Shot', markerfacecolor='red', markersize=10)
  plt.legend(handles=[good_shot_legend, bad_shot_legend], loc='upper right')

  plt.tight_layout()
  plt.show()




import matplotlib.pyplot as plt

# Assuming you have 'labels', 'training_data', and 'flattened_videos' defined

# Define the joint number for the elbow (13 in this case)
elbow_joint = 25

# Extract and flatten x, y, and z values for the elbow joint
x_values = [data[3 * elbow_joint] for data in flattened_videos]
y_values = [data[3 * elbow_joint + 1] for data in flattened_videos]
z_values = [data[3 * elbow_joint + 2] for data in flattened_videos]

# Create three separate subplots for x, y, and z values with different colors for good and bad shots
plt.figure(figsize=(15, 5))

# Plot x values
plt.subplot(131)
for i, label in enumerate(labels):
    if label == 1:
        plt.scatter(i, x_values[i], c='blue', label='Good Shot')
    else:
        plt.scatter(i, x_values[i], c='red', label='Bad Shot')
plt.title('Left Knee X Values')
good_shot_legend = plt.Line2D([0], [0], marker='o', color='w', label='Good Shot', markerfacecolor='blue', markersize=10)
bad_shot_legend = plt.Line2D([0], [0], marker='o', color='w', label='Bad Shot', markerfacecolor='red', markersize=10)
plt.legend(handles=[good_shot_legend, bad_shot_legend], loc='upper right')
# Plot y values
plt.subplot(132)
for i, label in enumerate(labels):
    if label == 1:
        plt.scatter(i, y_values[i], c='blue', label='Good Shot')
    else:
        plt.scatter(i, y_values[i], c='red', label='Bad Shot')
plt.title('Left Knee Y Values')
good_shot_legend = plt.Line2D([0], [0], marker='o', color='w', label='Good Shot', markerfacecolor='blue', markersize=10)
bad_shot_legend = plt.Line2D([0], [0], marker='o', color='w', label='Bad Shot', markerfacecolor='red', markersize=10)
plt.legend(handles=[good_shot_legend, bad_shot_legend], loc='upper right')
# Plot z values
plt.subplot(133)
for i, label in enumerate(labels):
    if label == 1:
        plt.scatter(i, z_values[i], c='blue', label='Good Shot')
    else:
        plt.scatter(i, z_values[i], c='red', label='Bad Shot')
plt.title('Left Knee Z Values')

# Create a single legend for all graphs
# Create a single legend for all graphs

good_shot_legend = plt.Line2D([0], [0], marker='o', color='w', label='Good Shot', markerfacecolor='blue', markersize=10)
bad_shot_legend = plt.Line2D([0], [0], marker='o', color='w', label='Bad Shot', markerfacecolor='red', markersize=10)
plt.legend(handles=[good_shot_legend, bad_shot_legend], loc='upper right')

plt.tight_layout()
plt.show()



