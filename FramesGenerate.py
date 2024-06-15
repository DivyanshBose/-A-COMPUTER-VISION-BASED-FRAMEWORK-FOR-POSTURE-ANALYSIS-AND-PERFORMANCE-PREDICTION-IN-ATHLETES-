import cv2
folderNum = 1
path = ('C:\\Users\\Divyansh Bose\\Downloads\\VisualMotionAnalysis\\VisualMotionAnalysis-main\\Videos\\Sample'
        + str(folderNum) + '\\')
classes = ['Badminton', 'Basketball', 'Cricket', 'Football', 'Tennis']
cnt = 0
num_frames_per_video = 150

for i in classes:
    FileName = i+'.mp4'
    cap = cv2.VideoCapture(path + FileName)
    cnt += 1
    frame_count = 1

    while True:
        ret, frame = cap.read()

        if ret:
            frame = cv2.resize(frame, (1050, 850))
            cv2.imwrite('C:\\Users\\Divyansh Bose\\Downloads\\VisualMotionAnalysis\\VisualMotionAnalysis-main\\'
                        'frames\\Video' + str(cnt) + '\\' + 'img%d.jpg' % frame_count, frame)

            cv2.imshow("res", frame)

            if frame_count >= num_frames_per_video:
                break

            frame_count += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
