import math as m
import cv2
import mediapipe as mp
import pandas as pd


def findDistance(x1, y1, x2, y2):
    dist = m.sqrt((x2-x1)**2+(y2-y1)**2)
    return dist


def findAngle(x1, y1, x2, y2):
    theta = m.acos((y2 - y1)*(-y1) / (m.sqrt((x2 - x1)**2 + (y2 - y1)**2) * y1))
    degree = int(180/m.pi)*theta
    return degree


def sendWarning():
    pass


good_frames = 0
bad_frames = 0

# Font type.
font = cv2.FONT_HERSHEY_SIMPLEX

# Colors.
blue = (255, 127, 0)
red = (50, 50, 255)
green = (127, 255, 0)
dark_blue = (127, 20, 0)
light_green = (127, 233, 100)
yellow = (0, 255, 255)
pink = (255, 0, 255)

# Dictionary to store frame number and corresponding offset
offset_data = {'Offset': [], 'Neck Inclination': [], 'Torso Inclination': [], 'Sport': []}


num_frames_per_video = 150  # Set the desired number of frames per video

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose()
cnt = 0
folderNum = 1
path = ('C:\\Users\\Divyansh Bose\\Downloads\\VisualMotionAnalysis\\VisualMotionAnalysis-main\\Videos\\Sample' +
        str(folderNum) + '\\')
classes = ['Basketball', 'Cricket', 'Football', 'Tennis', 'Volleyball']

for i in classes:
    flag = 0
    cnt += 1
    FileName = i+'.mp4'

    cap = cv2.VideoCapture(path + FileName)

    frame_count = 1

    while True:
        ret, img = cap.read()
        fps = cap.get(cv2.CAP_PROP_FPS)

        if img is None or img.size == 0:
            print("Error: Empty frame after resizing.")
            flag = 1
            break

        img = cv2.resize(img, (1050, 850))
        h, w, c = img.shape

        keypoints = pose.process(img)

        if keypoints.pose_landmarks:

            # Convert the image back to BGR.
            image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # Use lm and lmPose as representative of the following methods.
            lm = keypoints.pose_landmarks
            lmPose = mp_pose.PoseLandmark

            # Left shoulder.
            l_shldr_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * w)
            l_shldr_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * h)

            # Right shoulder.
            r_shldr_x = int(lm.landmark[lmPose.RIGHT_SHOULDER].x * w)
            r_shldr_y = int(lm.landmark[lmPose.RIGHT_SHOULDER].y * h)

            # Left ear.
            l_ear_x = int(lm.landmark[lmPose.LEFT_EAR].x * w)
            l_ear_y = int(lm.landmark[lmPose.LEFT_EAR].y * h)

            # Left hip.
            l_hip_x = int(lm.landmark[lmPose.LEFT_HIP].x * w)
            l_hip_y = int(lm.landmark[lmPose.LEFT_HIP].y * h)

            #  Left elbow
            l_elbow_x = int(lm.landmark[lmPose.LEFT_ELBOW].x * w)
            l_elbow_y = int(lm.landmark[lmPose.LEFT_ELBOW].y * w)

            # Calculate distance between left shoulder and right shoulder points.
            offset = findDistance(l_shldr_x, l_shldr_y, r_shldr_x, r_shldr_y)

            # Assist to align the camera to point at the side view of the person.
            # Offset threshold 30 is based on results obtained from analysis over 100 samples.
            if offset < 100:
                cv2.putText(image, str(int(offset)) + ' Aligned', (w - 150, 30), font, 0.9, green, 2)
            else:
                cv2.putText(image, str(int(offset)) + ' Not Aligned', (w - 150, 30), font, 0.9, red, 2)

            # Calculate angles.
            neck_inclination = findAngle(l_shldr_x, l_shldr_y, l_ear_x, l_ear_y)
            torso_inclination = findAngle(l_hip_x, l_hip_y, l_shldr_x, l_shldr_y)

            # Append offset data to dictionary
            # offset_data['Offset'].append(offset)
            # offset_data['Neck Inclination'].append(neck_inclination)
            # offset_data['Torso Inclination'].append(torso_inclination)
            # offset_data['Sport'].append(i)

            cv2.imwrite('C:\\Users\\Divyansh Bose\\Downloads\\VisualMotionAnalysis\\VisualMotionAnalysis-main\\'
                        'frames\\OutputVideo' + str(cnt) + '\\' + 'img%d.jpg' % frame_count, image)

            if frame_count >= num_frames_per_video:
                break
            frame_count += 1

            # Draw landmarks.
            cv2.circle(image, (l_shldr_x, l_shldr_y), 7, yellow, -1)
            cv2.circle(image, (l_ear_x, l_ear_y), 7, yellow, -1)

            # Let's take y - coordinate of P3 100px above x1,  for display elegance.
            # Although we are taking y = 0 while calculating angle between P1,P2,P3.
            cv2.circle(image, (l_shldr_x, l_shldr_y - 100), 7, yellow, -1)
            cv2.circle(image, (r_shldr_x, r_shldr_y), 7, pink, -1)
            cv2.circle(image, (l_hip_x, l_hip_y), 7, yellow, -1)

            # Similarly, here we are taking y - coordinate 100px above x1. Note that
            # you can take any value for y, not necessarily 100 or 200 pixels.
            cv2.circle(image, (l_hip_x, l_hip_y - 100), 7, yellow, -1)

            # Put text, Posture and angle inclination.
            # Text string for display.
            angle_text_string = 'Neck : ' + str(int(neck_inclination)) + '  Torso : ' + str(int(torso_inclination))

            # Determine whether good posture or bad posture.
            # The threshold angles have been set based on intuition.
            if neck_inclination < 40 and torso_inclination < 10:
                bad_frames = 0
                good_frames += 1

                cv2.putText(image, angle_text_string, (10, 30), font, 0.9, light_green, 2)
                cv2.putText(image, str(int(neck_inclination)), (l_shldr_x + 10, l_shldr_y), font, 0.9, light_green, 2)
                cv2.putText(image, str(int(torso_inclination)), (l_hip_x + 10, l_hip_y), font, 0.9, light_green, 2)

                # Join landmarks.
                cv2.line(image, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), green, 2)
                cv2.line(image, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y - 100), green, 2)
                cv2.line(image, (l_hip_x, l_hip_y), (l_shldr_x, l_shldr_y), green, 2)
                cv2.line(image, (l_hip_x, l_hip_y), (l_hip_x, l_hip_y - 100), green, 2)

            else:
                good_frames = 0
                bad_frames += 1

                cv2.putText(image, angle_text_string, (10, 30), font, 0.9, red, 2)
                cv2.putText(image, str(int(neck_inclination)), (l_shldr_x + 10, l_shldr_y), font, 0.9, red, 2)
                cv2.putText(image, str(int(torso_inclination)), (l_hip_x + 10, l_hip_y), font, 0.9, red, 2)

                # Join landmarks.
                cv2.line(image, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), red, 2)
                cv2.line(image, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y - 100), red, 2)
                cv2.line(image, (l_hip_x, l_hip_y), (l_shldr_x, l_shldr_y), red, 2)
                cv2.line(image, (l_hip_x, l_hip_y), (l_hip_x, l_hip_y - 100), red, 2)

            # Calculate the time of remaining in a particular posture.
            good_time = (1 / fps) * good_frames
            bad_time = (1 / fps) * bad_frames

            # Pose time.
            if good_time > 0:
                time_string_good = 'Good Posture Time : ' + str(round(good_time, 1)) + 's'
                cv2.putText(image, time_string_good, (10, h - 20), font, 0.9, green, 2)
            else:
                time_string_bad = 'Bad Posture Time : ' + str(round(bad_time, 1)) + 's'
                cv2.putText(image, time_string_bad, (10, h - 20), font, 0.9, red, 2)

            # If you stay in bad posture for more than 3 minutes (180s) send an alert.
            if bad_time > 180:
                sendWarning()

            cv2.imshow('Posture Detection', image)

            k = cv2.waitKey(1)
            if k == 27:  # Press 'Esc' to exit
                break

    # Convert offset data to DataFrame
    offset_df = pd.DataFrame(offset_data)

    # Save DataFrame to CSV
    if cnt == len(classes):
        offset_df.to_csv(f'Dataset' + str(folderNum) + '.csv', index=False)

    if flag == 1:
        continue

    # Release the VideoCapture object and video writer if used
    cap.release()

    # Close the windows.
    cv2.destroyAllWindows()
