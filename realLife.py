import cv2
import os
import torch
import numpy as np
import PIL.Image as pil
from torchvision import transforms
import matplotlib.pyplot as plt
from networks import ResnetEncoder, DepthDecoder
from layers import disp_to_depth
from utils import download_model_if_doesnt_exist
from evaluate_depth import STEREO_SCALE_FACTOR

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

# Define preprocessing transformations
preprocess = transforms.Compose([
    transforms.Resize((192, 640)),
    transforms.ToTensor(),
])

# Initialize video stream
cap = cv2.VideoCapture("http://192.168.100.11:4747/video")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the image
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_image = pil.fromarray(frame_rgb)
    input_image = preprocess(input_image).unsqueeze(0).to(device)

    # Predict depth
    with torch.no_grad():
        features = encoder(input_image)
        outputs = depth_decoder(features)
        disp = outputs[("disp", 0)]
        disp_resized = torch.nn.functional.interpolate(disp, (frame.shape[0], frame.shape[1]),
                                                       mode="bilinear", align_corners=False)
        # Apply depth scaling
        scaled_disp, depth = disp_to_depth(disp, 0.1, 100)
        metric_depth = STEREO_SCALE_FACTOR * depth.cpu().numpy()
    # Assuming 'metric_depth' is the numpy array with shape [1, 1, H, W]
    metric_depth_squeezed = metric_depth.squeeze()  # Removes the single-dimensional entries from the shape

    # Normalize the metric depth for visualization purposes only
    min_val, max_val = metric_depth_squeezed.min(), metric_depth_squeezed.max()
    metric_depth_vis = (metric_depth_squeezed - min_val) / (max_val - min_val)  # Scales values to [0, 1]
    metric_depth_vis = (metric_depth_vis * 255).astype(np.uint8)  # Scales values to [0, 255]

    
    

    # Convert to colormap for visualization
    depth_colormap = (metric_depth_squeezed / metric_depth.max() * 255).astype(np.uint8)
    depth_colormap = cv2.applyColorMap(depth_colormap, cv2.COLORMAP_MAGMA)

    # Show images
    cv2.imshow('Original', frame)
    cv2.imshow('Depth', depth_colormap)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything when job is finished
cap.release()
cv2.destroyAllWindows()
