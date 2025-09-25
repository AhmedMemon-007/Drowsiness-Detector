import cv2
import mediapipe as mp
import numpy as np
import time
import sys

#  helper functions 
def beep():
    """Simple cross-platform beep"""
    try:
        if sys.platform.startswith("win"):
            import winsound
            winsound.Beep(1000, 500)
        else:
            print("\a", end="")
    except:
        pass

def eye_aspect_ratio(pts):
    #Compute Eye Aspect Ratio (EAR) from 6 landmarks
    p1, p2, p3, p4, p5, p6 = pts
    v1 = np.linalg.norm(p2 - p6)
    v2 = np.linalg.norm(p3 - p5)
    h = np.linalg.norm(p1 - p4)
    return (v1 + v2) / (2.0 * h + 1e-6)

# detection class 
class DrowsinessDetector:
    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]

    def __init__(self, ear_thresh=0.22, perclos_window=50, perclos_limit=30):
        self.ear_thresh = ear_thresh
        self.perclos_window = perclos_window
        self.perclos_limit = perclos_limit
        self.buffer = []
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

    def process(self, frame):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.face_mesh.process(rgb)

        if not res.multi_face_landmarks:
            return frame, 0, False

        lm = res.multi_face_landmarks[0]
        pts = np.array([[p.x * w, p.y * h] for p in lm.landmark])

        left = eye_aspect_ratio([pts[i] for i in self.LEFT_EYE])
        right = eye_aspect_ratio([pts[i] for i in self.RIGHT_EYE])
        ear = (left + right) / 2.0

        # update buffer
        self.buffer.append(1 if ear < self.ear_thresh else 0)
        if len(self.buffer) > self.perclos_window:
            self.buffer.pop(0)
        perclos = 100 * sum(self.buffer) / len(self.buffer)

        # draw stats
        cv2.putText(frame, f"EAR: {ear:.2f}", (30,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(frame, f"PERCLOS: {perclos:.1f}%", (30,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        closed = perclos > self.perclos_limit
        return frame, perclos, closed

# main loop
def main():
    cap = cv2.VideoCapture(0)
    detector = DrowsinessDetector()
    last_beep = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame, perclos, closed = detector.process(frame)

        if closed and time.time() - last_beep > 3:
            beep()
            last_beep = time.time()
            cv2.putText(frame, "DROWSY ALERT!", (50,100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)

        cv2.imshow("Driver Drowsiness Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()