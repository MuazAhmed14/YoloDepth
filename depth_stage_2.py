import cv2
from ultralytics import YOLO
import time
import os
import torch
import numpy as np
import PIL.Image as pil
from torchvision import transforms
from networks import ResnetEncoder, DepthDecoder
from layers import disp_to_depth
from utils import download_model_if_doesnt_exist
from evaluate_depth import STEREO_SCALE_FACTOR

# Initialize the YOLO model
yolo_model = YOLO('yolov8n.pt')  # Make sure the model path is correct

# Load the pre-trained model
model_name = "mono+stereo_640x192"
download_model_if_doesnt_exist(model_name)
model_path = os.path.join("models", model_name)
encoder_path = os.path.join(model_path, "encoder.pth")
depth_decoder_path = os.path.join(model_path, "depth.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = ResnetEncoder(18, False)
loaded_dict_enc = torch.load(encoder_path, map_location=device)
encoder.load_state_dict({k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()})
encoder.to(device)
encoder.eval()

depth_decoder = DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))
loaded_dict = torch.load(depth_decoder_path, map_location=device)
depth_decoder.load_state_dict(loaded_dict)
depth_decoder.to(device)
depth_decoder.eval()

# Define preprocessing transformations for Monodepth2
preprocess = transforms.Compose([
    transforms.Resize((192, 640)),  # Monodepth2's expected input resolution
    transforms.ToTensor(),
])

# Define the video stream
cap = cv2.VideoCapture("http://192.168.100.11:4747/video")

# Original dimensions used by Monodepth2
mono_height = 192
mono_width = 640

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Initialize variables for calculating FPS
start_time = time.time()
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Process frame with YOLO
    yolo_results = yolo_model(frame)
    # Add a print statement to inspect the structure of yolo_resultsq
    print(type(yolo_results), yolo_results)
    # Preprocess and predict depth with Monodepth2
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_image = pil.fromarray(frame_rgb)
    input_image = preprocess(input_image).unsqueeze(0).to(device)

    with torch.no_grad():
        features = encoder(input_image)
        outputs = depth_decoder(features)
        disp = outputs[("disp", 0)]
        scaled_disp, depth = disp_to_depth(disp, 0.1, 100)
        metric_depth = STEREO_SCALE_FACTOR * depth.cpu().numpy()

    # Squeeze metric depth to remove unnecessary dimensions
    metric_depth_squeezed = metric_depth.squeeze()

    # Calculate the resize ratios for width and height
    resize_ratio_width = mono_width / frame.shape[1]
    resize_ratio_height = mono_height / frame.shape[0]

    # Draw bounding boxes and labels from YOLO results
    for i, det in enumerate(yolo_results.xyxy[0]):
        x1, y1, x2, y2, conf, cls_id = map(int, det.tolist()[:6])
        label = f'{yolo_model.names[int(cls_id)]} {conf:.2f}'
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Map YOLO detection coordinates to the original Monodepth2 resolution
        x1_depth = int(x1 * resize_ratio_width)
        x2_depth = int(x2 * resize_ratio_width)
        y1_depth = int(y1 * resize_ratio_height)
        y2_depth = int(y2 * resize_ratio_height)

        # Extract the corresponding depth information from Monodepth2
        depth_for_box = metric_depth_squeezed[y1_depth:y2_depth, x1_depth:x2_depth]
        mean_depth = np.mean(depth_for_box)  # Mean depth in meters

        # Display the mean depth on the frame
        depth_label = f'Depth: {mean_depth:.2f}m'
        cv2.putText(frame, depth_label, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    # Calculate and display FPS
    frame_count += 1
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display the frame with detections and depth map
    cv2.imshow('YOLO Object Detection with Orientation', frame)
    
    # Normalize the metric depth for visualization purposes only
    min_val, max_val = metric_depth_squeezed.min(), metric_depth_squeezed.max()
    metric_depth_vis = (metric_depth_squeezed - min_val) / (max_val - min_val)  # Scales values to [0, 1]
    metric_depth_vis = (metric_depth_vis * 255).astype(np.uint8)  # Scales values to [0, 255]
    depth_colormap = cv2.applyColorMap(metric_depth_vis, cv2.COLORMAP_MAGMA)
    cv2.imshow('Depth', depth_colormap)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything when job is finished
cap.release()
cv2.destroyAllWindows()
