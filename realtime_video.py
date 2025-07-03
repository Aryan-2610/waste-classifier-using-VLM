import cv2
import time
import tempfile
from waste import answer_dict  

FRAME_INTERVAL_SEC = 3 

def classify_video_stream(video_source=0):
    cap = cv2.VideoCapture(video_source)
    last_classified = time.time() - FRAME_INTERVAL_SEC
    result = {"caption": "", "category": "", "bin": "", "explain": ""}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()
        if current_time - last_classified >= FRAME_INTERVAL_SEC:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                cv2.imwrite(tmp.name, frame)
                try:
                    result = answer_dict(tmp.name)
                    print(result)
                except Exception as e:
                    print("❌ Error during inference:", e)
                last_classified = current_time

        # Show prediction on video
        display_text = f"{result['category']} | {result['bin']}"
        caption_text = result['caption']

        cv2.putText(frame, display_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, caption_text, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

        cv2.imshow("Waste Classifier", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    classify_video_stream(0) 
