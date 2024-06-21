import matplotlib.pyplot as plt
import cv2
import numpy as np
# import spynet


def save_flow_to_file(flow, height, width, path):
    # Convert the structured array to a regular 2D array
    flow_2d = np.zeros((height * width, 2), dtype=int)
    index = 0
    for i in range(height):
        for j in range(width):
            flow_2d[index, 0] = flow[i, j][0]
            flow_2d[index, 1] = flow[i, j][1]
            index += 1

    # Save the 2D array to a text file
    np.savetxt(path + 'flow.csv', flow_2d,
               fmt='%d %d', header='f0 f1', comments='')


def save_flow_image(frame1, frame2, flow, path):
    flow = np.reshape(flow, frame1.shape)
    height = frame1.shape[0]
    width = frame1.shape[1]
    save_flow_to_file(flow, height, width, path)
    step = 1  # 2
    plt.figure(figsize=(20, 5))

    plt.subplot(131)
    plt.title('Frame 1')
    plt.imshow(frame1, cmap='gray')
    plt.axis('off')

    plt.subplot(132)
    plt.title('Frame 2')
    plt.imshow(frame2, cmap='gray')
    plt.axis('off')

    plt.subplot(133)
    plt.title('Optical Flow')
    hsv = np.zeros((frame1.shape[0], frame1.shape[1], 3), dtype='uint8')
    hsv = cv2.cvtColor(hsv, cv2.COLOR_BGR2HSV)

    # Efficiently convert the (128, 128) array of tuples to a (128, 128, 2) array
    reshaped_flow_array = np.array(flow.tolist()).astype('float32')

    # Create the coordinate grids
    x_grid = np.arange(0, np.array(frame1).shape[1], step)
    y_grid = np.arange(np.array(frame1).shape[0] - 1, -1, -step)
    plt.quiver(x_grid, y_grid,
               reshaped_flow_array[::step, ::step, 0], -reshaped_flow_array[::step, ::step, 1])  # Invert Y component for correct orientation

    mag, ang = cv2.cartToPolar(
        reshaped_flow_array[..., 0], reshaped_flow_array[..., 1])

    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    plt.axis('off')
    plt.savefig(path + 'flow.png')
    cv2.imwrite(path + 'flow_colors.png', rgb)
