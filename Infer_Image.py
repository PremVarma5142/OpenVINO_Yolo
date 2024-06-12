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

# Load and preprocess the image
image = cv2.imread("/home/raptor1/Downloads/demo2.jpg")
N, C, H, W = 1, 1, 640, 640
resized_image = cv2.resize(image, (W, H))
input_image = cv2.dnn.blobFromImage(resized_image, 1/255, (W, H), [0, 0, 0], 1, crop=False)

# Perform inference
output = compiled_model([input_image])[output_layer_ir]
output = output[0]

# Initialize lists for boxes, scores, and class IDs
boxes = []
scores = []
class_ids = []

# Process output data
for row in output:
    classes_scores = row[5:]
    conf = row[4]
    
    (_min_score, max_score, _min_class_loc, (_x, max_class_index)) = cv2.minMaxLoc(classes_scores)
    if conf >= 0.2 and max_class_index != 6:
        box = [row[0] - 0.5 * row[2], row[1] - 0.5 * row[3], row[2], row[3]]
        boxes.append(box)
        scores.append(max_score)
        class_ids.append(max_class_index)

# Get original image dimensions
[height, width, _] = image.shape

# Calculate scale factors
scale_h = height / 640
scale_w = width / 640

# Perform Non-Maximum Suppression
RESULT_BOXES = cv2.dnn.NMSBoxes(boxes, scores, 0.5, 0.6, 0.5)

# Draw bounding boxes on the image
for index in RESULT_BOXES:
    box = boxes[index]
    x, y = round(box[0] * scale_w), round(box[1] * scale_h)
    x_plus_w, y_plus_h = round((box[0] + box[2]) * scale_w), round((box[1] + box[3]) * scale_h)
    image = cv2.rectangle(cv2.UMat(image), (x, y), (x_plus_w, y_plus_h), (0, 0, 255), 8)

# Save the result
os.makedirs("runs/openvino", exist_ok=True)
cv2.imwrite("runs/openvino/test_openvino_runtime.jpg", image)
