import cv2
from openvino.runtime import Core
import os

# Initialize OpenVINO core
ie = Core()

# Read and compile the model
model = ie.read_model(model="/home/raptor1/Downloads/openvino_model/best_openvino_model/best.xml")
compiled_model = ie.compile_model(model=model, device_name="CPU")

# Get input and output layers
input_layer_ir = compiled_model.input(0)
output_layer_ir = compiled_model.output()

# Define input and output video paths
input_video_path = "/home/raptor1/Downloads/controlfire.mp4"
output_video_path = "runs/openvino/test_openvino_runtime.mp4"

# Open the input video
cap = cv2.VideoCapture(input_video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    N, C, H, W = 1, 1, 640, 640
    resized_image = cv2.resize(frame, (W, H))
    input_image = cv2.dnn.blobFromImage(resized_image, 1/255, (W, H), [0, 0, 0], 1, crop=False)

    # Perform inference
    output = compiled_model([input_image])[output_layer_ir]
    output = output[0]

    boxes = []
    scores = []
    class_ids = []

    # Process the output data
    for row in output:
        classes_scores = row[5:]
        conf = row[4]
        (_min_score, max_score, _min_class_loc, (_x, max_class_index)) = cv2.minMaxLoc(classes_scores)
        if conf >= 0.25 and max_class_index != 6:
            box = [row[0] - 0.5 * row[2], row[1] - 0.5 * row[3], row[2], row[3]]
            boxes.append(box)
            scores.append(max_score)
            class_ids.append(max_class_index)

    # Get original frame dimensions
    [height, width, _] = frame.shape
    scale_h = height / 640
    scale_w = width / 640

    # Perform Non-Maximum Suppression
    RESULT_BOXES = cv2.dnn.NMSBoxes(boxes, scores, 0.5, 0.6, 0.5)
    for index in RESULT_BOXES:
        box = boxes[index]
        x, y = round(box[0] * scale_w), round(box[1] * scale_h)
        x_plus_w, y_plus_h = round((box[0] + box[2]) * scale_w), round((box[1] + box[3]) * scale_h)
        frame = cv2.rectangle(cv2.UMat(frame), (x, y), (x_plus_w, y_plus_h), (0, 0, 255), 4)

    # Write the frame to the output video
    out.write(frame)
    
    # Display the frame
    cv2.imshow('Processed Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

# Ensure output directory exists
os.makedirs("runs/openvino", exist_ok=True)
