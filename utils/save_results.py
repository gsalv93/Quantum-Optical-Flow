import cv2
import flow_vis
import imageio
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_flow_to_file(flow, height, width, path):
    # Convert the structured array to a regular 2D array
    flow_2d = np.zeros((height * width, 2), dtype=int)
    index = 0
    for i in range(height):
        for j in range(width):
            flow_2d[index, 0] = flow[i, j, 0]
            flow_2d[index, 1] = flow[i, j, 1]
            index += 1

    # Save the 2D array to a text file
    np.savetxt(path + 'flow.csv', flow_2d,
               fmt='%d %d', header='f0 f1', comments='')


def save_to_mp4(path, video_frames, flow_frames_est, fps):
    print("Saving mp4 file")
    with imageio.get_writer(path + 'out.mp4', mode="I", fps=fps) as writer:
        for idx, flow_est in enumerate(flow_frames_est):
            print("Adding frame to mp4 file: ", idx + 1)
            flow_color = flow_vis.flow_to_color(flow_est, convert_to_bgr=False)
            # flow_color = cv2.cvtColor(flow_color, cv2.COLOR_BGR2RGB)
            video_frame = video_frames[idx]
            video_frame = cv2.cvtColor(video_frame, cv2.COLOR_GRAY2BGR)
            added_image = cv2.addWeighted(video_frame, 0.5, flow_color, 0.3, 0)
            writer.append_data(added_image)
            print(idx + 1)
    print("Done")


def save_to_gif(path, video_frames, flow_frames_est, fps):
    print("Saving GIF file")
    with imageio.get_writer(path + 'out.gif', mode="I", fps=fps) as writer:
        for idx, flow_est in enumerate(flow_frames_est):
            print("Adding frame to gif file: ", idx + 1)
            flow_color = flow_vis.flow_to_color(flow_est, convert_to_bgr=False)
            # flow_color = cv2.cvtColor(flow_color, cv2.COLOR_BGR2RGB)
            video_frame = video_frames[idx]
            video_frame = cv2.cvtColor(video_frame, cv2.COLOR_GRAY2BGR)
            added_image = cv2.addWeighted(video_frame, 0.5, flow_color, 0.3, 0)
            writer.append_data(added_image)
            print(idx + 1)
    print("Done")


def writeFlow(name, flow):
    f = open(name, 'wb')
    f.write('PIEH'.encode('utf-8'))
    np.array([flow.shape[1], flow.shape[0]], dtype=np.int32).tofile(f)
    flow = flow.astype(np.float32)
    flow.tofile(f)


def save_bqm(path, bqm):
    print("Saving BQM to file...")
    tmp_file = bqm.to_file()
    with open(path, 'wb') as f:
        tmp_file.seek(0)
        f.write(tmp_file.read())
        f.close()
    print("Saved bqm to file!")


def save_flow_image(frame_1, frame_2, flow_uv, path):
    
    print("Saving results to folder...")
    rows = frame_1.shape[0]
    cols = frame_1.shape[1]
    step = 1  # 2
    plt.figure(figsize=(20, 5))

    plt.subplot(131)
    plt.title('Frame 1')
    plt.imshow(frame_1, cmap='gray')
    plt.axis('off')

    plt.subplot(132)
    plt.title('Frame 2')
    plt.imshow(frame_2, cmap='gray')
    plt.axis('off')

    plt.subplot(133)
    plt.title('Optical Flow')

    x_grid = np.arange(1, cols-1, step)
    y_grid = np.arange(rows-2, 0, -step)
    flow_uv_x = flow_uv[1:rows-1:step, 1:cols-1:step, 0]
    flow_uv_y = -flow_uv[1:rows-1:step, 1:cols-1:step, 1]

    plt.quiver(x_grid, y_grid,
               flow_uv_x, flow_uv_y)

    flow_color = flow_vis.flow_to_color(flow_uv, convert_to_bgr=False)
    f_1c = cv2.cvtColor(frame_1, cv2.COLOR_GRAY2BGR)

    plt.axis('off')
    flow_color = cv2.cvtColor(flow_color, cv2.COLOR_BGR2RGB)

    plt.savefig(path + 'flow.png')
    cv2.imwrite(path + 'flow_colors.png', flow_color)

    added_image = cv2.addWeighted(f_1c, 0.5, flow_color, 0.3, 0)

    cv2.imwrite(path + 'combined.png', added_image)
    print("Results are in the work directory.")
