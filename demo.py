import os
import random
import argparse
import sys
import numpy as np
import cv2
import math
from utils.qof import perform_optical_flow
from utils.load_data import load_frames, load_video, load_bqm
from utils.save_results import save_flow_image, save_to_gif, save_to_mp4
from utils.qubos import classic_QUBO_computation, parallel_QUBO_computation, pixel_block_QOF, para_pixel_block_QOF
from utils.IO import read, write
from utils.FlowStats.stats import FlowStats

random.seed()
np.set_printoptions(threshold=sys.maxsize)
dataset_dir = 'dataset/'
OUTFOLDER = "demo_output/"

if not os.path.exists(OUTFOLDER):
    os.makedirs(OUTFOLDER)


def initialize_difference_matrix(labels):
    size = len(labels)
    differences_matrix = np.zeros((size, size, 2))
    for i, d1 in enumerate(labels):
        for j, d2 in enumerate(labels):
            f_difference = (d1[0] - d2[0], d1[1] - d2[1])
            differences_matrix[i, j, 0] = f_difference[0]
            differences_matrix[i, j, 1] = f_difference[1]
    return differences_matrix


def generate_labels(max_disparity, ratio):
    range_val = np.arange(-np.floor(max_disparity/ratio),
                          np.floor(max_disparity/ratio) + 1)

    dcol = np.reshape(np.tile(range_val, (len(range_val), 1)), (1, -1))
    drow = np.reshape(
        np.tile(range_val[:, np.newaxis], (1, len(range_val))), (1, -1))

    labels = np.stack((dcol.flatten().astype(dtype='int8'),
                      drow.flatten().astype(dtype='int8')), axis=-1)
    difference_matrix = initialize_difference_matrix(labels)
    return labels, difference_matrix

def get_flow_from_solution(frame, best_sample, labels, step):
    rows = int(frame.shape[0]) 
    cols = int(frame.shape[1])
    n_d = labels.shape[0]
    flow_uv = np.zeros([frame.shape[0], frame.shape[1], 2])
    for i in range(0, rows, step):
        for j in range(0, cols, step):
            for k in range(n_d):
                if (best_sample.sample[f'x[{int(i/step)}][{int(j/step)}][{k}]'] == 1):

                    temp1 = np.full((step, step), labels[k, 1])
                    flow_uv[i:i+step, j:j+step, 0] = temp1
                    temp2 = np.full((step, step), labels[k, 0])
                    flow_uv[i:i+step, j:j+step, 1] = temp2

    return flow_uv


def main():
    # Input parameters for the energy function.
    # Usage:
    # python demo.py -r L1 -d 1 -s 1 -a 0.5 -b 1 -l 2 -ta 1 -m c
    parser = argparse.ArgumentParser(
        description='Parameters for the energy function.')
    parser.add_argument('-r', '--reg_type', nargs='?', default='L1',
                        help='Regularization type.')
    parser.add_argument('-d', '--disp', nargs='?', default=1,
                        help='Maximum displacement of the optical flow')
    parser.add_argument('-s', '--scale', nargs='?', default=1,
                        help='Image scaling value')
    parser.add_argument('-a', '--alpha', nargs='?', default=0.5,
                        help='Penalities weight.')
    parser.add_argument('-b', '--beta', nargs='?', default=7.5,
                        help='Value used when computing Laplacian weights in the energy function.')
    parser.add_argument('-l', '--Lambda', nargs='?', default=2,
                        help='Value that\'s multiplied to the second member in the energy function.')
    parser.add_argument('-ta', '--tao', nargs='?', default=1,
                        help='Thershold used when computing regularization.')
    parser.add_argument('-m', '--mode', nargs='?', default='c',
                        help='Mode for OF computation.')

    args = parser.parse_args()

    ### --- additional parameters --- ###
    reg_type = args.reg_type.upper()
    scale = float(args.scale)
    max_disp = int(args.disp) * scale
    alpha = float(args.alpha)
    beta = float(args.beta)
    Lambda = float(args.Lambda)
    tao = float(args.tao)
    mode = args.mode.lower()  # c: classical, hq hybrid, q quantum

    # If an invalid regularization parameter is entered, I just default to L1.
    if reg_type != 'L1' and reg_type != 'L2' and reg_type != 'C':
        reg_type = 'L1'

    # If an invalid mode parameter is entered, I just default to c (classical).
    if mode != 'c' and mode != 'hq' and mode != 'q':
        mode = 'c'

    # Generating labels
    labels, diff_matrix = generate_labels(max_disp, scale)

    # Preparing parameters
    offset = int(np.floor(max_disp/scale))
    parameters = {
        "reg_type": reg_type,
        "offset": offset,
        "alpha": alpha,
        "beta": beta,
        "lambda": Lambda,
        "tao": tao,
        "mode": mode
    }
    # 1, 4, 16, 64...
    block_size = 1  # Set to 1 when not using Block Optical Flow
    block_side = math.ceil(block_size/2)
    step = int(math.sqrt(block_size))

    # Images used for testing purposes
    """  
    N = 8
    M = 8
    frame_1 = np.zeros([N, M], dtype='uint8')
    frame_2 = np.zeros([N, M], dtype='uint8')
    gt = np.zeros([N, M, 2], dtype='int8')

    frame_1[5, 5] = 255
    frame_2[5, 4] = 255

    # frame_1[0, 0:N] = 128
    # frame_2[0, 0:N] = 128

    # frame_1[0:N, 5] = 128
    # frame_2[0:N, 5] = 128

    # actual gt (-1 0)
    gt[3, 3, 0] = -1
    gt[3, 3, 1] = 0
    """
    #############################################
    ### PERFORMING OPTICAL FLOW ON TWO FRAMES ###
    #############################################
    """ """
    # Loading and preparing frames
    frame_1, frame_2 = load_frames(dataset_dir, scale)

    rows = frame_1.shape[0]
    cols = frame_1.shape[1]

    # pad frames
    p_frame1 = np.pad(frame_1, pad_width=offset*block_side, mode='edge')
    p_frame2 = np.pad(frame_2, pad_width=(
        (offset*2)*block_side), mode='edge')

    # Old:
    # p_frame1 = np.pad(frame_1, pad_width=offset, mode='edge')
    # p_frame2 = np.pad(frame_2, pad_width=((offset*2)), mode='edge')

    # gaussian blur
    # frame_1 = cv2.GaussianBlur(frame_1, (3, 3), 3)
    # frame_2 = cv2.GaussianBlur(frame_2, (3, 3), 3)

    # Pixelwise Optical Flow
    # Classic
    # bqm = classic_QUBO_computation(
    #     p_frame1, p_frame2, labels, parameters, diff_matrix)

    # Parallelized
    bqm = parallel_QUBO_computation(
        p_frame1, p_frame2, labels, parameters, diff_matrix)

    # Block Optical Flow
    # bqm = pixel_block_QOF(p_frame1, p_frame2, labels,
    #                       parameters, step, diff_matrix)

    # Parallelized computation of Block Optical Flow
    # bqm = para_pixel_block_QOF(
    #     p_frame1, p_frame2, labels, parameters, step, diff_matrix)

    # Load BQM from file if needed.
    # bqm = load_bqm('bqms/bqm_model.bqm')

    solution = perform_optical_flow(bqm, mode, rows, cols)
    flow_est = get_flow_from_solution(frame_1, solution, labels, step)

    # Saving results
    save_flow_image(frame_1, frame_2, flow_est, OUTFOLDER)
    write(OUTFOLDER + 'computed_flows/flow.flo', flow_est)

    ##############################################
    #### PERFORMING OPTICAL FLOW ON GIF/VIDEO ####
    ##############################################
    # Uncomment this block of code when computing the BQM on a gif or video

    """          
    video_frames, fps, extension = load_video(dataset_dir, scale)
    flow_frames_est = []
    num_frames = 2  # len(video_frames)
    print(f"Total video frames: {num_frames}")
    for i in range(0, num_frames-1):
        print(f"Computing frames {i+1} and {i+2}...")
        frame_1 = video_frames[i]
        frame_2 = video_frames[i+1]
        rows, cols = frame_1.shape
        # frame_1 = cv2.GaussianBlur(frame_1, (3, 3), 3)
        # frame_2 = cv2.GaussianBlur(frame_2, (3, 3), 3)

        # p_frame1 = np.pad(frame_1, pad_width=offset, mode='edge')
        # p_frame2 = np.pad(frame_2, pad_width=((offset*2)), mode='edge')
        p_frame1 = np.pad(frame_1, pad_width=offset*block_side, mode='edge')
        p_frame2 = np.pad(frame_2, pad_width=(
            (offset*2)*block_side), mode='edge')

        # bqm computation strats
        # Classic
        # bqm = classic_QUBO_computation(p_frame1, p_frame2, labels, parameters, diff_matrix)

        # Parallelized
        # bqm = parallel_QUBO_computation(
        #     p_frame1, p_frame2, labels, parameters, diff_matrix)

        #  Block classic
        # bqm = pixel_block_QOF(p_frame1, p_frame2, labels, parameters, step, diff_matrix)

        # Block parallelized
        bqm = para_pixel_block_QOF(
            p_frame1, p_frame2, labels, parameters, step, diff_matrix)

        #  Getting colored flow
        solution = perform_optical_flow(bqm, mode, rows, cols)
        flow_est = get_flow_from_solution(
            frame_1, solution, labels, step)

        curr_path = 'frame_' + str(i+1) + '_'
        print(OUTFOLDER + curr_path)
        save_flow_image(frame_1, frame_2, flow_est, OUTFOLDER + curr_path)

        flow_frames_est.append(flow_est)
        write(OUTFOLDER+'computed_flows/flow' + str(i+1) + '.flo', flow_est)

    # Check for file format and call the function according to it
    if extension == '.mp4':
        save_to_mp4(OUTFOLDER, video_frames, flow_frames_est, fps)
    elif extension == '.gif':
        save_to_gif(OUTFOLDER, video_frames, flow_frames_est, fps)
    """


if __name__ == "__main__":
    main()
