import cv2
import mediapipe as mp

mp_holistic=mp.solutions.holistic
mp_drawing=mp.solutions.drawing_utils
mp_face_mesh=mp.solutions.face_mesh

def mediapipe_detection(image,model):
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB) #color conversion BGR 2 RGB
    image.flags.writeable=False # image is no longer writable
    results=model.process(image) # make prediction 
    image.flags.writeable=True # image is writable
    image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR) # color conversion RGB 2 BGR
    return image,results 

def draw_styled_landmark(image,result):
    mp_drawing.draw_landmarks(image,result.face_landmarks,mp_face_mesh.FACEMESH_TESSELATION,
                              mp_drawing.DrawingSpec(color=(250,210,150),thickness=1,circle_radius=1),
                              mp_drawing.DrawingSpec(color=(255,240,180),thickness=1,circle_radius=1)) #Draw face connection 
    mp_drawing.draw_landmarks(image,result.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(0,165,255),thickness=2,circle_radius=2),
                              mp_drawing.DrawingSpec(color=(0,140,255),thickness=2,circle_radius=2)) #Draw pose connection
    mp_drawing.draw_landmarks(image,result.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(255,0,191),thickness=2,circle_radius=3),
                              mp_drawing.DrawingSpec(color=(200,0,120),thickness=2,circle_radius=2)) #Draw left hand connnection
    mp_drawing.draw_landmarks(image,result.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(255,200,100),thickness=2,circle_radius=2),
                              mp_drawing.DrawingSpec(color=(255,240,180),thickness=2,circle_radius=2)) #Draw right hand connection

view=cv2.VideoCapture(0) #grabing the web cam
# set mediapipe model
with mp_holistic.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5) as holistic:
    while view.isOpened():# double checking if we are still accesing the web cam and initiatiating the loop
        #Read feed
        ret, frame =view.read() # reading the fram from the web cam
        image,result=mediapipe_detection(frame, holistic)
        print(result)
        draw_styled_landmark(image,result) #Draw LAndmarks
        cv2.imshow('LOOK CAM',image) # showing to user
        if cv2.waitKey(10) & 0xFF ==ord('q'): # breaking
            break
    view.release() # releasre the web cam
    cv2.destroyAllWindows() # close down the frame