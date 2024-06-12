OpenVINO Inference on Video

This project demonstrates how to perform inference on a video using a pre-trained model with the OpenVINO toolkit. The input video is processed frame-by-frame, and detected objects are highlighted with bounding boxes in the output video.

Project Structure

```

.
├── README.md               # Project documentation
├── main.py                 # Main script for video processing and inference
├── runs
│   └── openvino
│       └── test_openvino_runtime.mp4  # Output video
└── model
    └── best.xml            # Pre-trained model file

Requirements

    Python 3.8+
    OpenVINO toolkit
    OpenCV
    Required Python packages (install using the provided requirements.txt)

Installing Requirements

```

pip install -r requirements.txt

Usage

    Model Setup: Place your pre-trained OpenVINO model XML file (best.xml) in the model directory.
    Input Video: Ensure your input video file (controlfire.mp4) is in the specified directory.
    Run the Script: Execute the main script to process the video and perform inference.

    ```

    python Infer_Video.py

Script Details

The main script (main.py) performs the following steps:

    Initialize OpenVINO Core: Load the model and compile it for the CPU.

    python

ie = Core()
model = ie.read_model(model="/path/to/your/model/best.xml")
compiled_model = ie.compile_model(model=model, device_name="CPU")

Video Processing: Read the input video, process each frame, and perform inference to detect objects.

python

cap = cv2.VideoCapture(input_video_path)

Preprocessing: Resize the frames and prepare them for inference.

python

resized_image = cv2.resize(frame, (W, H))
input_image = cv2.dnn.blobFromImage(resized_image, 1/255, (W, H), [0, 0, 0], 1, crop=False)

Inference: Perform inference on each frame and process the output to extract bounding boxes and class IDs.

python

output = compiled_model([input_image])[output_layer_ir]

Post-Processing: Apply Non-Maximum Suppression (NMS) to filter out overlapping boxes and draw bounding boxes on the frames.

python

RESULT_BOXES = cv2.dnn.NMSBoxes(boxes, scores, 0.5, 0.6, 0.5)

Output: Write the processed frames to an output video file and display the frames in real-time.

python

    out.write(frame)

Output

The output video with detected objects is saved in the runs/openvino directory with the name test_openvino_runtime.mp4.
Notes

    Adjust the confidence_threshold and nms_threshold in the script for better results based on your model and data.
    Ensure the input video and model paths are correctly specified in the script.

License

This project is licensed under the MIT License. See the LICENSE file for more details.
Acknowledgments

    OpenVINO Toolkit
    OpenCV

Contributing

Contributions are welcome! Please open an issue or submit a pull request.
