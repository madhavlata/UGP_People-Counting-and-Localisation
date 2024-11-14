import cv2
import numpy as np
import torch
import random
import matplotlib.pyplot as plt

# Assuming the rest of your code remains the same above this point

frame = cv2.imread(file)
print(frame.shape)
scale_factor = 0.5
frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
ori_img = frame.copy()
frame = frame.copy()
image = tensor_transform(frame)
image = img_transform(image).unsqueeze(0)

with torch.no_grad():
    d6 = model(image)

    # Save the model output d6 as a segmentation map
    # Convert d6 to a NumPy array
    d6_np = d6.data.cpu().numpy()
    
    # Assume d6 is in shape (1, num_classes, height, width)
    # We want to visualize the first class (for example)
    segmentation_map = d6_np[0, 0]  # Taking the first class for visualization

    # Normalize the segmentation map to the range [0, 255]
    segmentation_map_normalized = (segmentation_map - np.min(segmentation_map)) / (np.max(segmentation_map) - np.min(segmentation_map))
    segmentation_map_normalized = (segmentation_map_normalized * 255).astype(np.uint8)

    # Save the segmentation map as an image
    cv2.imwrite('./segmentation_map.jpg', segmentation_map_normalized)

    count, pred_kpoint = counting(d6)
    point_map = generate_point_map(pred_kpoint)
    box_img = generate_bounding_boxes(pred_kpoint, frame)
    show_fidt = show_fidt_func(d6.data.cpu().numpy())

    res1 = np.hstack((ori_img, show_fidt))
    res2 = np.hstack((box_img, point_map))
    res = np.vstack((res1, res2))

    density = './density_map.jpg' 
    cv2.imwrite('./demo.jpg', res)
    print("pred:%.3f" % count)

    x = random.randint(1,100000) 
    density = 'static/density_map'+str(x)+'.jpg' 
    plt.imsave(density, show_fidt)

return count, density