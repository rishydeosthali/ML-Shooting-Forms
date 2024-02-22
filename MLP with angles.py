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
final_angles = []
angles = []
numb = 0

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


def take_coordinates(landmarks):
    left_elbow = [[landmarks[14].x,landmarks[14].y],[landmarks[12].x,landmarks[12].y],[landmarks[24].x,landmarks[24].y]]
    right_elbow = [[landmarks[13].x,landmarks[13].y],[landmarks[11].x,landmarks[11].y],[landmarks[23].x,landmarks[23].y]]
    left_arm = [[landmarks[16].x,landmarks[16].y],[landmarks[14].x,landmarks[14].y],[landmarks[12].x,landmarks[12].y]]
    right_arm = [[landmarks[15].x,landmarks[15].y],[landmarks[13].x,landmarks[13].y],[landmarks[11].x,landmarks[11].y]]
    left_leg = [[landmarks[24].x,landmarks[24].y],[landmarks[26].x,landmarks[26].y],[landmarks[28].x,landmarks[28].y]]
    right_leg = [[landmarks[23].x,landmarks[23].y],[landmarks[25].x,landmarks[25].y],[landmarks[27].x,landmarks[27].y]]
    left_foot = [[landmarks[32].x,landmarks[32].y],[landmarks[28].x,landmarks[28].y],[landmarks[26].x,landmarks[26].y]]
    right_foot = [[landmarks[31].x,landmarks[31].y],[landmarks[27].x,landmarks[27].y],[landmarks[25].x,landmarks[25].y]]
    return left_elbow,right_elbow,left_arm,right_arm,left_leg,right_leg,left_foot,right_foot


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
              left_elbow,right_elbow,left_arm,right_arm,left_leg,right_leg,left_foot,right_foot = take_coordinates(landmarks)
              angle_left_elbow = calculate_angle(left_elbow[0],left_elbow[1],left_elbow[2])
              final_angles.append(angle_left_elbow)
              angle_right_elbow = calculate_angle(right_elbow[0], right_elbow[1], right_elbow[2])
              final_angles.append(angle_right_elbow)
              angle_left_arm = calculate_angle(left_arm[0], left_arm[1], left_arm[2])
              final_angles.append(angle_left_arm)
              angle_right_arm = calculate_angle(right_arm[0], right_arm[1], right_arm[2])
              final_angles.append(angle_right_arm)
              angle_left_leg = calculate_angle(left_leg[0], left_leg[1], left_leg[2])
              final_angles.append(angle_left_leg)
              angle_right_leg = calculate_angle(right_leg[0], right_leg[1], right_leg[2])
              final_angles.append(angle_right_leg)
              angle_left_foot = calculate_angle(left_foot[0], left_foot[1], left_foot[2])
              final_angles.append(angle_left_foot)
              angle_right_foot = calculate_angle(right_foot[0], right_foot[1], right_foot[2])
              final_angles.append(angle_right_foot)








          mp_drawing.draw_landmarks(annotated_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cap.release()
        cv2.destroyAllWindows()
        angles.append(final_angles)





from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences


max_sequence_length = max(len(arr) for arr in angles)
X_padded = pad_sequences(angles, maxlen=max_sequence_length, padding='post', dtype='int')


X = np.array(X_padded)
X = X/180
labels2 = np.array(labels)


X_train, X_test, y_train, y_test = train_test_split(X, labels2, test_size=0.3, random_state=42)



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

max_sequence_length = max(len(arr) for arr in angles)
X_padded = pad_sequences(angles, maxlen=max_sequence_length, padding='post', dtype='int')


X = np.array(X_padded)
X = X/180
labels2 = np.array(labels)


X_train, X_test, y_train, y_test = train_test_split(X, labels2, test_size=0.3, random_state=42)


learning_rate = 0.001
num_hidden_layers = 3
hidden_units = [128, 64, 32]
activation_function = 'relu'
dropout_rate = 0.4
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
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                    validation_data=(X_test, y_test), callbacks=[early_stopping])

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print("Test accuracy:", accuracy)



