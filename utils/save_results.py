import matplotlib.pyplot as plt
import numpy as np


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
    step = 2  # 2
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
    # Create the coordinate grids
    x = np.arange(0, width, step)
    y = np.arange(height - 1, -1, -step)

    # Initialize arrays to store the flow components
    u = np.zeros((height // step, width // step))
    v = np.zeros((height // step, width // step))

    # Extract the flow components
    for i in range(0, height, step):
        for j in range(0, width, step):
            u[i // step, j // step] = flow[i, j][0]  # horizontal component
            v[i // step, j // step] = flow[i, j][1]  # vertical component

    # Create the quiver plot
    plt.quiver(x, y, u, v)

    plt.axis('off')
    plt.savefig(path + 'flow.png')
