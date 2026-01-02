import time
import numpy as np
import math
from tqdm import tqdm
from pyqubo import Array
from utils.save_results import save_bqm
from multiprocessing import Process, cpu_count, set_start_method
from utils.qubo_computation import QUBO, para_QUBO, block_QUBO, para_block_QUBO, merge_bqm


def create_subarrays(arr, step):
    subarrays = []
    for i in range(0, len(arr), step):
        subarray = arr[i:i + step]
        subarrays.append(subarray)
    return subarrays


def create_frame1_patches(frame, rows, cols, offset):
    patches = []
    coordinates_list = []
    for i in range(offset, rows+offset):
        for j in range(offset, cols+offset):
            x1 = i-offset
            y1 = j-offset
            x2 = i+1+offset
            y2 = j+1+offset
            # Regular Patch
            patch = frame[x1:x2, y1:y2]
            # Smaller Patch
            # patch = frame[x1+offset:x2-offset, y1+offset:y2-offset]

            patches.append(patch)
            coordinates_list.append((x1, y1))

    return patches, coordinates_list


def create_frame2_patches(frame, rows, cols, labels, offset):
    patches = []
    n_d = labels.shape[0]
    for i in range(offset, rows+offset):
        for j in range(offset, cols+offset):
            x1 = i-offset
            x2 = i+1+offset
            y1 = j-offset
            y2 = j+1+offset

            patch_per_label = []
            for k in range(n_d):
                l_0 = int(labels[k, 0])
                l_1 = int(labels[k, 1])
                # Regular Patch
                patch = frame[x1+offset+l_0:x2+offset +
                              l_0, y1+offset+l_1:y2+offset+l_1]

                # Smaller Patch
                # patch = frame[x1+(offset*2)+l_0:x2+
                #                  l_0, y1+(offset*2)+l_1:y2+l_1]

                patch_per_label.append(patch)

            patches.append(patch_per_label)
    return patches


def classic_QUBO_computation(frame_1, frame_2, labels, parameters, diff_matrix):

    start_time = time.time()
    print("Creating QUBO... [Classic]")
    H_pyqubo = QUBO(frame_1, frame_2, labels, parameters, diff_matrix)
    print("Done!")
    print("--- %s seconds ---" % (time.time() - start_time))

    model = H_pyqubo.compile()

    bqm = model.to_bqm()
    # Save BQM file if needed
    # save_bqm('bqms/bqm_model.bqm', bqm)
    return bqm


def parallel_QUBO_computation(frame_1, frame_2, labels, parameters, diff_matrix):
    set_start_method('fork', force=True)
    offset = parameters["offset"]

    rows = frame_1.shape[0]-(2*offset)
    cols = frame_1.shape[1]-(2*offset)
    n_d = labels.shape[0]
    f1_patches, coordinates_list = create_frame1_patches(
        frame_1, rows, cols, offset)
    f2_patches = create_frame2_patches(frame_2, rows, cols, labels, offset)
    x_pyqubo = Array.create('x', shape=(rows, cols, n_d), vartype='BINARY')
    print((rows, cols, n_d))
    start_time = time.time()
    print("Creating QUBO... [Parallel]")
    sub_array = create_subarrays(coordinates_list, cpu_count())
    processes = []
    i = 0
    bqm = 0

    for coords in tqdm(sub_array):
        partial_bqm = 0
        for c in coords:
            f1_patch = f1_patches[i]
            f2_patch_group = f2_patches[i]
            coordinates = c
            i = i + 1
            process = Process(target=para_QUBO, args=(
                frame_1, f1_patch, f2_patch_group, labels, parameters, coordinates, x_pyqubo, diff_matrix))
            processes.append(process)
            process.start()

        count = 0
        while count < len(processes):
            partial_bqm, n = merge_bqm()
            if not n:
                continue
            bqm += partial_bqm
            count += n

        for process in processes:
            process.join()
        processes = []
    print("Done!")
    print("--- %s seconds ---" % (time.time() - start_time))
    # Save BQM file if needed
    # save_bqm('bqms/bqm_model.bqm', bqm)
    return bqm


def pixel_block_QOF(frame_1, frame_2, labels, parameters, step, diff_matrix):
    start_time = time.time()
    print("Creating QUBO with blocks... [Classic]")
    H_pyqubo = block_QUBO(frame_1, frame_2, labels,
                          parameters, step, diff_matrix)
    print("Done!")
    print("--- %s seconds ---" % (time.time() - start_time))

    model = H_pyqubo.compile()

    bqm = model.to_bqm()
    # Save BQM file if needed
    #  save_bqm('bqms/bqm_model.bqm', bqm)
    return bqm


def create_block_frame1_patches(frame, rows, cols, offset, step):
    patches = []
    coordinates_list = []
    block_len = math.ceil((step*step) / 2)
    offset = offset*block_len
    for i in range(offset, rows+offset, step):
        for j in range(offset, cols+offset, step):
            x1 = i-offset
            x2 = i+block_len+offset
            y1 = j-offset
            y2 = j+block_len+offset
            patch = frame[x1:x2, y1:y2]
            # Regular Patch
            # patch = frame[x1:x1+step, y1:y1+step]
            # Smaller Patch
            # patch = frame[x1+offset:x2-offset, y1+offset:y2-offset]

            patches.append(patch)
            coordinates_list.append((int(x1), int(y1)))

    return patches, coordinates_list


def create_block_frame2_patches(frame, rows, cols, labels, offset, step):
    patches = []
    n_d = labels.shape[0]
    block_len = math.ceil((step*step) / 2)
    offset = offset*block_len
    for i in range(offset, rows+offset, step):
        for j in range(offset, cols+offset, step):
            x1 = i-offset
            x2 = i+block_len+offset
            y1 = j-offset
            y2 = j+block_len+offset

            patch_per_label = []
            for k in range(n_d):
                l_0 = int(labels[k, 0])
                l_1 = int(labels[k, 1])
                # Regular Patch
                patch = frame[x1+offset+l_0:x2+offset +
                              l_0, y1+offset+l_1:y2+offset+l_1]
                # patch = frame[x1+offset+l_0:x1+offset+step +
                #               l_0, y1+offset+l_1:y1+offset+step+l_1]

                patch_per_label.append(patch)

            patches.append(patch_per_label)
    return patches


def para_pixel_block_QOF(frame_1, frame_2, labels, parameters, step, diff_matrix):
    set_start_method('fork', force=True)
    offset = parameters["offset"]
    block_len = math.ceil((step*step) / 2)

    # rows = frame_1.shape[0]-(2*offset)
    # cols = frame_1.shape[1]-(2*offset)
    rows = int((frame_1.shape[0]-(2*offset*block_len)))
    cols = int((frame_1.shape[1]-(2*offset*block_len)))
    n_d = labels.shape[0]
    f1_patches, coordinates_list = create_block_frame1_patches(
        frame_1, rows, cols, offset, step)
    f2_patches = create_block_frame2_patches(
        frame_2, rows, cols, labels, offset, step)
    x_pyqubo = Array.create('x', shape=(
        int(rows/step), int(cols/step), n_d), vartype='BINARY')

    start_time = time.time()
    print("Creating QUBO with blocks... [Parallel]")
    sub_array = create_subarrays(coordinates_list, cpu_count())
    processes = []
    i = 0
    bqm = 0

    for coords in tqdm(sub_array):
        partial_bqm = 0
        for c in coords:
            f1_patch = f1_patches[i]
            f2_patch_group = f2_patches[i]
            coordinates = c
            i = i + 1
            process = Process(target=para_block_QUBO, args=(
                frame_1, f1_patch, f2_patch_group, labels, parameters, coordinates, x_pyqubo, step, diff_matrix))
            processes.append(process)
            process.start()

        count = 0

        while count < len(processes):
            partial_bqm, n = merge_bqm()
            if not n:
                continue
            bqm += partial_bqm
            count += n

        for process in processes:
            process.join()
        processes = []
    print("Done!")
    print("--- %s seconds ---" % (time.time() - start_time))
    # Save BQM file if needed
    # save_bqm('bqms/bqm_model.bqm', bqm)
    return bqm
