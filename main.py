import cv2

face_ref = cv2.CascadeClassifier('face_ref.xml') # reference to the classifier (face recognition)
camera = cv2.VideoCapture(1) # 0 means default camera (device camera)

def face_detection(frame):
    optimized_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) # improve perfromance: RGB to grayscale
    faces = face_ref.detectMultiScale(optimized_frame, scaleFactor=1.1, minSize=(500, 500), minNeighbors=5) # scaleFactor: downscalling image for performance (1.1 means 10%), minSize: minimum size of face (500px x 500px)
    return faces

def drawer_box(frame):
    for x, y, w, h in face_detection(frame):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 4) # GBR color format
    pass

def close_window():
    camera.release()
    cv2.destroyAllWindows()
    exit()

def main():
    while True: # infinite take picture from camera until [*]
        _, frame = camera.read() # returns a boolean and a frame
        drawer_box(frame)
        cv2.imshow('Face Detection Edo', frame) # show image to screen
        if cv2.waitKey(1) & 0xFF == ord('q'): # wait for keypress [*]
            close_window()

if __name__ == '__main__':
    main()