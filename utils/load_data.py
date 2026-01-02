import cv2
import dimod


def load_frames(base_folder, scale):
    print("Loading frames...")
    # circleframe1_32x32.png
    frame1_dir = base_folder + \
        '64_1.png'
    # circleframe2_32x32.png
    frame2_dir = base_folder + \
        '64_2.png' 

    frame_1 = cv2.imread(frame1_dir, cv2.IMREAD_GRAYSCALE)
    frame_2 = cv2.imread(frame2_dir, cv2.IMREAD_GRAYSCALE)

    f1_rows, f1_cols = frame_1.shape
    frame1_resized = cv2.resize(
        frame_1, (int(f1_cols / scale), int(f1_rows / scale)))
    f2_rows, f2_cols = frame_2.shape
    frame2_resized = cv2.resize(
        frame_2, (int(f2_cols / scale), int(f2_rows / scale)))

    print("Frames loaded!")
    return frame1_resized, frame2_resized


def load_video(base_folder, scale=1):
    #  pedestrians_80x80.mp4, seq1_64.gif, TestVideoSC80T.mp4 [-a 1 -l 0.1] eps .4
    path = base_folder + 'TestVideoSC80T.mp4'
    extension = path[-4:]
    videocap = cv2.VideoCapture(path)
    fps = videocap.get(cv2.CAP_PROP_FPS)
    frames = []
    boolean, frame = videocap.read()
    while (boolean):
        f_rows, f_cols, c = frame.shape
        frame_resized = cv2.resize(
            frame, (int(f_cols / scale), int(f_rows / scale)))

        gray_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        frames.append(gray_frame)
        boolean, frame = videocap.read()

    return frames, fps, extension


def load_bqm(path):
    print("Loading BQM from file...")
    with open(path, 'rb') as f:
        loaded_bqm = dimod.BinaryQuadraticModel.from_file(f)
        f.close()
    print("Loaded BQM from file!")
    return loaded_bqm
