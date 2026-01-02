import cv2
import sys
import time
import math
import numpy as np
from pyqubo import Array
from multiprocessing import Process, Queue, cpu_count

np.set_printoptions(threshold=sys.maxsize)
result_queue = Queue(maxsize=cpu_count())


def L1_reg(f_star):
    penalty_1 = abs(f_star[0])
    penalty_2 = abs(f_star[1])
    return penalty_1 + penalty_2


def L2_reg(f_star):
    penalty_1 = (f_star[0]**2)
    penalty_2 = (f_star[1]**2)
    return penalty_1 + penalty_2


def Charbonnier_reg(f_star):
    eps = 0.5
    penalty_1 = math.sqrt(f_star[0]**2 + eps**2)
    penalty_2 = math.sqrt(f_star[1]**2 + eps**2)
    return penalty_1 + penalty_2


def regularization(f_star, tao, reg_type):
    if reg_type == 'L1':
        tot_penalty = L1_reg(f_star)
    if reg_type == 'L2':
        tot_penalty = L2_reg(f_star)
    if reg_type == 'C':
        tot_penalty = Charbonnier_reg(f_star)

    return min(tot_penalty, tao)


def weight(I1p, I1q, beta):
    inv_beta = 1 / beta
    diff = np.linalg.norm(np.array(I1p) - np.array(I1q))
    return math.exp(- (diff * inv_beta))


def is_patch_flat(patch, threshold=1e-6):
    # return (np.sum(patch) == 0)
    variance = np.var(patch)
    return variance < threshold


def delta(padded_frame_1, reg_type, f_difference, pixel_neighbour, pixel_coordinates, beta, tao, step):
    rho_S = 0

    # f_difference = (d1[0] - d2[0],
    #                 d1[1] - d2[1])

    reg_factor = regularization(f_difference, tao, reg_type)

    # I1 at pixel_coord, I2 at pixel_neighbours
    I1p = padded_frame_1[pixel_coordinates[0]:pixel_coordinates[0]+step,
                         pixel_coordinates[1]:pixel_coordinates[1]+step]
    I1q = padded_frame_1[pixel_neighbour[0]:pixel_neighbour[0]+step,
                         pixel_neighbour[1]:pixel_neighbour[1]+step]

    w = weight(I1p, I1q, beta)

    rho_S = w*reg_factor
    return rho_S


def get_neighbours(i, j, step):

    north = (i-step, j)
    south = (i+step, j)
    west = (i, j-step)
    east = (i, j+step)
    N = [north, south, west, east]
    return N


def initialize_difference_matrix(labels):
    size = len(labels)
    differences_matrix = np.zeros((size, size, 2))
    for i, d1 in enumerate(labels):
        for j, d2 in enumerate(labels):
            f_difference = (d1[0] - d2[0], d1[1] - d2[1])
            differences_matrix[i, j, 0] = f_difference[0]
            differences_matrix[i, j, 1] = f_difference[1]
    return differences_matrix


def QUBO(frame_1, frame_2, labels, parameters, diff_matrix):

    n_d = labels.shape[0]
    # Unpacking parameters
    reg_type = parameters["reg_type"]
    offset = parameters["offset"]
    alpha = parameters["alpha"]
    beta = parameters["beta"]
    lambd = parameters["lambda"]
    tao = parameters["tao"]

    rows = frame_1.shape[0]-(2*offset)
    cols = frame_1.shape[1]-(2*offset)

    x_pyqubo = Array.create('x', shape=(rows, cols, n_d), vartype='BINARY')

    H1_pyqubo = sum(sum(
        alpha*((1 - sum(x_pyqubo[i][j][d] for d in range(n_d)))**2) for j in range(cols)) for i in range(rows))

    H2_pyqubo = 0
    for i in range(offset, rows+offset):
        for j in range(offset, cols+offset):
            for k in range(n_d):
                x1 = i-offset
                x2 = i+1+offset
                y1 = j-offset
                y2 = j+1+offset

                l_0 = int(labels[k, 0])
                l_1 = int(labels[k, 1])

                # Regular patches
                patch1 = frame_1[x1:x2, y1:y2]

                patch2 = frame_2[x1+offset+l_0:x2+offset +
                                 l_0, y1+offset+l_1:y2+offset+l_1]

                # Smaller patches
                # patch1 = frame_1[x1+offset:x2-offset, y1+offset:y2-offset]

                # patch2 = frame_2[x1+(offset*2)+labels[k, 0]:x2+
                #                   labels[k, 0], y1+(offset*2)+labels[k, 1]:y2+labels[k, 1]]

                ncc_matrix = cv2.matchTemplate(
                    patch1, patch2, cv2.TM_CCORR_NORMED)
                # ncc = np.min(np.max(correlation_coefficient(patch1,patch2),initial=0),initial=1)

                rho_d = 1 - ncc_matrix[0, 0]
                # rho_d = 1 - np.sum(ncc_matrix[ncc_matrix > 0])

                H2_pyqubo += rho_d*x_pyqubo[x1][y1][k]
    """"""
    H3_pyqubo = 0
    for i in range(offset, rows+offset):
        for j in range(offset, cols+offset):
            x1 = i-offset
            y1 = j-offset
            neigh = get_neighbours(x1, y1, step=1)
            for d1 in range(n_d):
                for d2 in range(n_d):
                    usable_coordinates = get_usable_coords(rows, cols, neigh)

                    # usable_coordinates = [sub for sub in neigh if all(
                    #     ele >= 0 and ele < rows and ele < cols for ele in sub)]
                    for c in usable_coordinates:
                        i_prime = c[0]
                        j_prime = c[1]
                        H3_pyqubo += (lambd * (delta(frame_1, reg_type, diff_matrix[d1, d2], c, (
                            x1, y1), beta, tao, step=0)) * x_pyqubo[x1][y1][d1] * x_pyqubo[i_prime][j_prime][d2])

    H_pyqubo = H1_pyqubo + H2_pyqubo + H3_pyqubo

    return H_pyqubo


def para_QUBO(frame_1, patch1, patch2, labels, parameters, coordinates, x_pyqubo, diff_matrix):
    (i, j) = coordinates
    n_d = labels.shape[0]

    # Unpacking parameters
    reg_type = parameters["reg_type"]
    offset = parameters["offset"]
    alpha = parameters["alpha"]
    beta = parameters["beta"]
    lambd = parameters["lambda"]
    tao = parameters["tao"]

    rows = frame_1.shape[0]-(2*offset)
    cols = frame_1.shape[1]-(2*offset)

    H1_pyqubo = alpha*(1 - sum(x_pyqubo[i][j][d] for d in range(n_d)))**2

    H2_pyqubo = 0
    eps = 0.45  # 0.20
    for k in range(n_d):

        p2 = patch2[k]
        # If botch patches equal 0, NCC is 0.
        ncc_matrix = cv2.matchTemplate(patch1, p2, cv2.TM_CCORR_NORMED)
        ncc = ncc_matrix[0, 0]

        # if is_patch_flat(patch1):
        #     ncc = 0
        if ncc != 1:
            ncc = ncc - eps  # penalizing incorrect flows.

        rho_d = 1 - ncc
        H2_pyqubo += rho_d*x_pyqubo[i][j][k]
    """"""

    H3_pyqubo = 0
    neigh = get_neighbours(i, j, step=1)

    for d1 in range(n_d):
        for d2 in range(n_d):
            usable_coordinates = get_usable_coords(rows, cols, neigh)
            # usable_coordinates = [sub for sub in neigh if all(
            #     ele >= 0 and ele < rows and ele < cols for ele in sub)]
            for c in usable_coordinates:
                i_prime = c[0]
                j_prime = c[1]
                H3_pyqubo += (lambd * (delta(frame_1, reg_type, diff_matrix[d1, d2], c, (i, j), beta, tao, step=0))
                              * x_pyqubo[i][j][d1] * x_pyqubo[i_prime][j_prime][d2])

    H_pyqubo = H2_pyqubo + H3_pyqubo + H1_pyqubo

    model = H_pyqubo.compile()
    bqm = model.to_bqm()
    result_queue.put(bqm)


def block_QUBO(frame_1, frame_2, labels, parameters, step, diff_matrix):

    n_d = labels.shape[0]
    # Unpacking parameters
    reg_type = parameters["reg_type"]
    offset = parameters["offset"]
    alpha = parameters["alpha"]
    beta = parameters["beta"]
    lambd = parameters["lambda"]
    tao = parameters["tao"]
    block_len = math.ceil((step*step) / 2)
    # Old: leva block_len
    rows = int((frame_1.shape[0]-(2*offset*block_len)))
    cols = int((frame_1.shape[1]-(2*offset*block_len)))
    diff_matrix = initialize_difference_matrix(labels)
    x_pyqubo = Array.create('x', shape=(
        int(rows/step), int(cols/step), n_d), vartype='BINARY')

    H1_pyqubo = sum(sum(alpha*((1 - sum(x_pyqubo[i][j][d] for d in range(n_d)))**2)
                    for j in range(int(cols/step))) for i in range(int(rows/step)))
    print("H1 created")
    H2_pyqubo = 0
    # Old: togli offset
    offset = offset*block_len

    for i in range(offset, rows+offset, step):
        for j in range(offset, cols+offset, step):
            # Old: scommenta le righe commentate e x2, y2
            x1 = i-offset
            x2 = i+block_len+offset
            y1 = j-offset
            y2 = j+block_len+offset
            pixel_block_1 = frame_1[x1:x2, y1:y2]
            # pixel_block_1 = frame_1[x1:x1+step, y1:y1+step]

            for k in range(n_d):
                l_0 = int(labels[k, 0])
                l_1 = int(labels[k, 1])
                pixel_block_2 = frame_2[x1+offset+l_0:x2+offset +
                                        l_0, y1+offset+l_1:y2+offset+l_1]
                # pixel_block_2 = frame_2[x1+offset+l_0:x1+offset +
                #                         step+l_0, y1+offset+l_1:y1+offset+step+l_1]

                ncc_matrix = cv2.matchTemplate(
                    pixel_block_1, pixel_block_2, cv2.TM_CCORR_NORMED)

                ncc = ncc_matrix[0, 0]
                # Old: non ci sono
                eps = 0.4  # 0.20
                if ncc != 1:
                    ncc = ncc - eps  # penalizing incorrect flows.
                if is_patch_flat(pixel_block_1):
                    ncc = 0
                # ncc = np.min(np.max(correlation_coefficient(patch1,patch2),initial=0),initial=1)

                rho_d = 1 - ncc
                # rho_d = 1 - np.sum(ncc_matrix[ncc_matrix > 0])

                H2_pyqubo += rho_d*x_pyqubo[int(x1/step)][int(y1/step)][k]
    print("H2 created")
    H3_pyqubo = 0

    for i in range(offset, rows+offset, step):
        for j in range(offset, cols+offset, step):
            x1 = i-offset
            y1 = j-offset

            neigh = get_neighbours(x1, y1, step)

            # print("** CURRENT COORDINATES **")
            # print(int(x1/step), int(y1/step))
            for d1 in range(n_d):
                for d2 in range(n_d):
                    usable_coordinates = get_usable_coords(rows, cols, neigh)
                    # usable_coordinates2 = [sub for sub in neigh if all(
                    #     ele >= 0 and ele < rows and ele < cols for ele in sub)]
                    for c in usable_coordinates:
                        # print("ADJACENT COORDINATES")
                        i_prime = c[0]
                        j_prime = c[1]
                        # difference = diff_matrix[d1, d2]
                        H3_pyqubo += (lambd * (delta(frame_1, reg_type, diff_matrix[d1, d2], c, (
                            x1, y1), beta, tao, step=0)) * x_pyqubo[int(x1/step)][int(y1/step)][d1] * x_pyqubo[int(i_prime/step)][int(j_prime/step)][d2])
    print("H3 created")

    H_pyqubo = H1_pyqubo + H2_pyqubo + H3_pyqubo

    return H_pyqubo


def get_usable_coords(rows, cols, neighbours):
    usable_coordinates = []
    for n in neighbours:
        ni = n[0]
        nj = n[1]
        if (ni >= 0 and nj >= 0) and ni < rows and nj < cols:
            usable_coordinates.append(n)
    return usable_coordinates


def para_block_QUBO(frame_1, patch1, patch2, labels, parameters, coordinates, x_pyqubo, step, diff_matrix):
    (i, j) = coordinates
    n_d = labels.shape[0]
    # Unpacking parameters
    reg_type = parameters["reg_type"]
    offset = parameters["offset"]
    alpha = parameters["alpha"]
    beta = parameters["beta"]
    lambd = parameters["lambda"]
    tao = parameters["tao"]

    block_len = math.ceil((step*step) / 2)
    rows = int((frame_1.shape[0]-(2*offset*block_len)))
    cols = int((frame_1.shape[1]-(2*offset*block_len)))

    H1_pyqubo = alpha * \
        (1 - sum(x_pyqubo[int(i/step)][int(j/step)][d] for d in range(n_d)))**2

    eps = 0.45  # 0.20
    H2_pyqubo = 0
    for k in range(n_d):
        p2 = patch2[k]
        ncc_matrix = cv2.matchTemplate(patch1, p2, cv2.TM_CCORR_NORMED)
        ncc = ncc_matrix[0, 0]

        if ncc != 1:  # and not is_patch_flat(patch1):
            ncc = ncc - eps  # penalizing incorrect flows.
        if is_patch_flat(patch1):
            ncc = 0
        rho_d = 1 - ncc
        H2_pyqubo += rho_d*x_pyqubo[int(i/step)][int(j/step)][k]

    """"""
    H3_pyqubo = 0
    neigh = get_neighbours(i, j, step)

    for d1 in range(n_d):
        for d2 in range(n_d):
            usable_coordinates = get_usable_coords(rows, cols, neigh)
            # usable_coordinates = [sub for sub in neigh if all(
            #     ele >= 0 and ele < rows and ele < cols for ele in sub)]
            for c in usable_coordinates:
                i_prime = c[0]
                j_prime = c[1]
                H3_pyqubo += (lambd * (delta(frame_1, reg_type, diff_matrix[d1, d2], c, (i, j), beta, tao, step=0))
                              * x_pyqubo[int(i/step)][int(j/step)][d1] * x_pyqubo[int(i_prime/step)][int(j_prime/step)][d2])

    H_pyqubo = H2_pyqubo + H3_pyqubo + H1_pyqubo

    model = H_pyqubo.compile()
    bqm = model.to_bqm()
    result_queue.put(bqm)


def get_queue():
    results = []
    while not result_queue.empty():
        results.append(result_queue.get())
    return results


def merge_bqm():
    bqm_total = 0
    count = 0
    while not result_queue.empty():
        partial = result_queue.get()
        bqm_total += partial
        count += 1
    return bqm_total, count
