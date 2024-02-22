## Create a method to crop one of the frames by calculating max width and max height before hand, and THEN cropping the frame to match such dimensions.

from google.colab.patches import cv2_imshow
import cv2
from google.colab import drive
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import glob
import cv2
!pip install mediapipe
import mediapipe as mp


drive.mount('/content/drive')
drive_service = build('drive', 'v3')

def max_width_height(videopath):
  # Initialize MediaPipe Pose model
  mp_pose = mp.solutions.pose
  pose = mp_pose.Pose()
  padding = 170
  max_width = 0
  max_height = 0

  # Open the video capture
  cap = cv2.VideoCapture(videopath)  # Replace with your video file path
  count = True
  while True:
      ret, frame = cap.read()
      if not ret:
          break

      # Convert the frame to RGB (MediaPipe Pose model expects RGB input)
      rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

      # Perform pose estimation on the frame
      results = pose.process(rgb_frame)

      if results.pose_landmarks:
          # Extract nose and legs keypoints
          landmarks = results.pose_landmarks.landmark
          nose = landmarks[mp_pose.PoseLandmark.NOSE]
          left_leg = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
          right_leg = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]

          # Calculate cropping region based on keypoints
          x_coords = [nose.x, left_leg.x, right_leg.x]
          y_coords = [nose.y, left_leg.y, right_leg.y]

          min_x = int(max(0, min(x_coords) * frame.shape[1]))
          max_x = int(min(frame.shape[1], max(x_coords) * frame.shape[1]))
          min_y = int(max(0, min(y_coords) * frame.shape[0]))
          max_y = int(min(frame.shape[0], max(y_coords) * frame.shape[0]))

          # Define the cropping region with padding to keep some background

          x1 = max(0, min_x - padding)
          y1 = max(0, min_y - padding)
          x2 = min(frame.shape[1], max_x + padding)
          y2 = min(frame.shape[0], max_y + padding)
          # Crop the frame using the calculated region
          cropped_frame = frame[y1:y2, x1:x2]
          width = x2-x1
          height = y2-y1
          if(width > max_width):
            max_width = width
          if(height > max_height):
            max_height = height
          # Display the cropped frame

          count = False

      # Press 'q' to exit the loop
      if cv2.waitKey(30) & 0xFF == ord('q'):
          break

  # Release the video capture and close all windows
  cap.release()
  cv2.destroyAllWindows()
  return max_width,max_height
  
  
  def crop_frame(video_path, max_width, max_height):
    count = 0;
    videos = []
    # Initialize MediaPipe Pose model
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    # Open the video capture
    cap = cv2.VideoCapture(video_path)  # Replace with your video file path
    alltrue = True
    while True:
        ret, frame = cap.read()
        if not ret:
            break
  
        # Convert the frame to RGB (MediaPipe Pose model expects RGB input)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  
        # Perform pose estimation on the frame
        results = pose.process(rgb_frame)
  
        if results.pose_landmarks:
            # Extract nose and legs keypoints
            landmarks = results.pose_landmarks.landmark
            nose = landmarks[mp_pose.PoseLandmark.NOSE]
            left_leg = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
            right_leg = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]
  
            # Calculate cropping region based on keypoints
            x_coords = [nose.x, left_leg.x, right_leg.x]
            y_coords = [nose.y, left_leg.y, right_leg.y]
  
            min_x = int(max(0, min(x_coords) * frame.shape[1]))
            max_x = int(min(frame.shape[1], max(x_coords) * frame.shape[1]))
            min_y = int(max(0, min(y_coords) * frame.shape[0]))
            max_y = int(min(frame.shape[0], max(y_coords) * frame.shape[0]))
  
            # Define the cropping region with padding to keep some background
            padding_width = int((max_width - (max_x - min_x))/2)
            padding_height = int((max_height - (max_y - min_y))/2)
            x1 = max(0, min_x - padding_width)
            y1 = max(0, min_y - padding_height)
            x2 = min(frame.shape[1], max_x + padding_width)
            y2 = min(frame.shape[0], max_y + padding_height)
            if(x2-x1 != max_width):
              if(x2+(max_width - (x2-x1)) <= frame.shape[1]):
                x2 = x2+(max_width - (x2-x1))
              else:
                x1 = x1-(max_width - (x2-x1))
            if(y2-y1 != max_height):
              if(y2+(max_height - (y2-y1))<= frame.shape[0]):
                y2 = y2+(max_height - (y2-y1))
              else:
                y1 = y1-(max_height-(y2-y1))
  
  
            # Crop the frame using the calculated region
  
            cropped_frame = frame[y1:y2, x1:x2]
            videos.append(cropped_frame)
  
            # Display the cropped frame
            #cv2_imshow(cropped_frame)
            #print("("+str(x2-x1)+","+str(y2-y1)+")")
  
  
        # Press 'q' to exit the loop
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
  
    # Release the video capture and close all windows
    cap.release()
    cv2.destroyAllWindows()
    return videos


def crop_video(input_video_path,index):
    import shutil
    max_width, max_height = max_width_height(input_video_path)

    cap = cv2.VideoCapture(input_video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # Change the codec here if needed
    output_video_path = 'output_video' + str(index) + '.mp4'

    out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (max_width, max_height))
    video = crop_frame(input_video_path,max_width,max_height)
    for i in video:
        out.write(i)

    cap.release()
    out.release()

    destination_folder = '/content/drive/MyDrive/Bad_Shots2'
    shutil.copy(output_video_path, destination_folder)
    return output_video_path


good_shot_directory = '/content/drive/MyDrive/Good_Shots/'
bad_shot_directory = '/content/drive/MyDrive/Bad_Shots/'

video_extensions = ['*.mp4', '*.avi', '*.mov']

# Search for video files in the directory using glob
good_shots = []
bad_shots = []
for extension in video_extensions:
    good_shots.extend(glob.glob(good_shot_directory + extension))
    bad_shots.extend(glob.glob(bad_shot_directory + extension))

for x in good_shots2:
  cap = cv2.VideoCapture(x)
  while True:
    ret,frame = cap.read()
    if not ret:
      break
    print(str(frame.shape[0]) + " " + str(frame.shape[1]))
    break
  cap.release()
  cv2.destroyAllWindows()



good_shot_directory = '/content/drive/MyDrive/Good_Shots/'
bad_shot_directory = '/content/drive/MyDrive/Bad_Shots/'

video_extensions = ['*.mp4', '*.avi', '*.mov']

# Search for video files in the directory using glob
good_shots = []
bad_shots = []
for extension in video_extensions:
    good_shots.extend(glob.glob(good_shot_directory + extension))
    bad_shots.extend(glob.glob(bad_shot_directory + extension))


good_shot_directory = '/content/drive/MyDrive/Good_Shots2/'
bad_shot_directory = '/content/drive/MyDrive/Bad_Shots2/'

video_extensions = ['*.mp4', '*.avi', '*.mov']

# Search for video files in the directory using glob
good_shots2 = []
bad_shots2 = []
for extension in video_extensions:
    good_shots2.extend(glob.glob(good_shot_directory + extension))
    bad_shots2.extend(glob.glob(bad_shot_directory + extension))


for x in good_shots2:
  cap = cv2.VideoCapture(x)
  while True:
    ret,frame = cap.read()
    if not ret:
      break
    print(str(frame.shape[0]) + " " + str(frame.shape[1]))
    break
  cap.release()
  cv2.destroyAllWindows()
  
  