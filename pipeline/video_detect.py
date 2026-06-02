import cv2
from ultralytics import YOLO

# 1. Load the pre-trained YOLOv8 Nano model
model = YOLO("yolov8n.pt")

# 2. Open the test video file
video_path = "data/test_video.mp4"
cap = cv2.VideoCapture(video_path)

# 3. Get video properties to set up the output video writer
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# 4. Set up the video writer to save the processed output
# 'mp4v' is a standard codec used for generating .mp4 files
output_path = "data/output_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

frame_count = 0

# 5. Start a loop to process the video frame by frame
while cap.isOpened():
    # Read the next frame
    ret, frame = cap.read()
    
    # If 'ret' is False, it means we have reached the end of the video
    if not ret:
        break
        
    frame_count += 1
    
    # 6. Run YOLO detection on this single frame
    # classes=[0] only detects people. verbose=False stops YOLO from printing long logs every frame
    results = model(frame, classes=[0], verbose=False)
    result = results[0]
    
    # 7. Count how many people were found in this frame
    person_count = len(result.boxes)
    
    # 8. Draw the bounding boxes onto the frame
    # YOLO provides .plot() which draws all boxes and returns the updated image
    annotated_frame = result.plot()
    
    # 9. Draw our live person count text on the top left of the frame
    cv2.putText(
        img=annotated_frame, 
        text=f"People Count: {person_count}", 
        org=(50, 50),                  # X, Y coordinates
        fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
        fontScale=1.5,                 # Font size
        color=(0, 0, 255),             # Red text (OpenCV uses BGR format)
        thickness=3                    # Text thickness
    )
    
    # 10. Print the frame number and people count to the terminal
    print(f"Frame {frame_count}: {person_count} people detected")
    
    # 11. Write the fully annotated frame into our output video file
    out.write(annotated_frame)

# 12. Release resources when finished
cap.release()
out.release()
print(f"Finished processing! Saved to {output_path}")
