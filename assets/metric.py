import numpy as np
import matplotlib.pyplot as plt

# Load the depth map
depth_map = np.load('bus_depth.npy')

# Remove extra dimensions
depth_map_squeezed = np.squeeze(depth_map)

# Check the new shape of depth_map, it should be (192, 640) or similar
print("New shape:", depth_map_squeezed.shape)

# Visualize the depth map
plt.imshow(depth_map_squeezed, cmap='plasma')
plt.colorbar()  # Show color scale
plt.title('Depth Map')
plt.show()

#python test_simple.py --image_path assets\bus.jpg --model_name mono+stereo_640x192 --pred_metric_depth