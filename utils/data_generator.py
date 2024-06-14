import cv2
import sys
import math
import numpy as np

dataset_dir = 'dataset/'
epsilon = sys.float_info.min

# MISC
def get_pad_size(counter):
    if 0 <= counter <= 1:
        return 1
    elif 2 <= counter <= 5:
        return 2
    elif 6 <= counter <= 11:
        return 3
    elif 12 <= counter <= 19:
        return 4
    return 5


def pad_images(image_1, image_2, pad_size):
    p_frame1 = cv2.copyMakeBorder(
        image_1, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_CONSTANT, 0)

    # Padding value for image_2 is bigger since out of bounds situations during I2 patch calculation for NCC may otherwise occur.
    p_frame2 = cv2.copyMakeBorder(
        image_2, pad_size*2, pad_size*2, pad_size*2, pad_size*2, cv2.BORDER_CONSTANT, 0)
    
    return p_frame1, p_frame2


# RHO_S STUFF

def regularization(f_star):
    tao = 1
    penalty_1 = abs(f_star[0])
    penalty_2 = abs(f_star[1])

    return min(penalty_1 + penalty_2, tao)


def weight(I1_value, I2_value):
    beta = 0.01

    return math.exp(-((np.linalg.norm(np.array(I1_value) - np.array(I2_value)))/beta))


# RHO_D STUFF

"""
NCC is the normalized cross-correlation between
two patches, one centered at p in I1 and one centered at
(p+fp) in I2, computed in each color channel and averaged.
"""
# Normalized CrossCorrelation computation
# https://stackoverflow.com/questions/53436231/normalized-cross-correlation-in-python


def ncc_computation(patch_I1, patch_I2):

    linear_patch_I1 = patch_I1.flatten()
    linear_patch_I2 = patch_I2.flatten()
    # Epsilon is set in order to avoid division by zero
    norm_patch_I1 = np.linalg.norm(linear_patch_I1)
    linear_patch_I1 = linear_patch_I1 / (norm_patch_I1 + epsilon)
    norm_patch_I2 = np.linalg.norm(linear_patch_I2)
    linear_patch_I2 = linear_patch_I2 / (norm_patch_I2 + epsilon)

    correlation = np.correlate(linear_patch_I1, linear_patch_I2, mode='full')

    return correlation

# Computing the penalty factor.
# Patches from the two frames are extracted and then passed to the NCC function.

def penalty(pixel_coordinates, label, padded_I1, padded_I2, pad_size):

    penalty_value = 10000

    # PREPARING I1 AND I2 PATCHES FOR NCC
    patch_size = pad_size*2+1
    # I'm computing the pixel coordinates of the padded image patch, given the original pixel coordinates
    pixel_coords_of_padded_I1 = tuple(
        map(sum, zip(pixel_coordinates, (pad_size, pad_size))))

    # Computing the patch around the center pixel, this depends on pad_size.
    patch_x = int(pixel_coords_of_padded_I1[0] - pad_size)
    patch_y = int(pixel_coords_of_padded_I1[1] - pad_size)

    patch_I1 = padded_I1[patch_x:patch_x +
                         patch_size, patch_y:patch_y+patch_size].astype('int32')

    # I do the same for I2
    pixel_coords_in_I2 = tuple(
        map(sum, zip(pixel_coordinates, label)))  # p+fp

    pixel_coords_of_padded_I2 = tuple(
        map(sum, zip(pixel_coords_in_I2, (pad_size+pad_size, pad_size+pad_size))))

    patch_x = int(pixel_coords_of_padded_I2[0] - pad_size + pad_size)
    patch_y = int(pixel_coords_of_padded_I2[1] - pad_size + pad_size)

    patch_I2 = padded_I2[patch_x:patch_x +
                         patch_size, patch_y:patch_y+patch_size].astype('int32')

    # If p+fp is in the buffer zone, I set its value to a big penalty value
    if (pixel_coords_in_I2[0] < 0 or pixel_coords_in_I2[1] < 0) or (pixel_coords_in_I2[0] >= padded_I2.shape[0]-(pad_size+3) or pixel_coords_in_I2[1] >= padded_I2.shape[1]-(pad_size+3)):
        # print(patch_I1)
        # print(patch_I2)
        patch_I2[1][1] = penalty_value

    ncc = np.mean(ncc_computation(patch_I1, patch_I2))
    output = 1 - np.maximum(ncc, 0)

    return output

# Computing Preference matrix P.
def get_preference_matrix_fm(frame_1, frame_2, labels, theta):

    # Preference Consensus Matrix
    P = np.zeros((labels.shape[0], labels.shape[1]))
    # Energies matrix
    Energy = np.zeros_like(P)

    Lambda = 1

    """
    E(f) = sum_[p in I1]{rho_d(p, fp, I1, I2)} + lambda*sum_[(p, q) in N]{wpq * rho_s(fp-fq)}
    p -> pixel I'm currently analyzing.
    fp -> flow of pixel p.
    p + fp -> pixel after applying flow fp.
    N -> 4-connected pixel grid.
    q -> neighbour pixels of p (4 pixel NSWE neighbour).
    fq -> flow of pixel q.
    rho_d computes NCC between I1 patch and I2 patch.
    rho_s computes norm 1 or Charbonnier Penalty of fp-fq.
    wpq -> weight value (weight function).
    lambda -> arbitrary value.
    """
    
    for label_index, label in enumerate(labels.T):

        iteration = 0
        label = np.reshape(label, frame_1.shape)
        # PADDING IMAGES
        pad_size = get_pad_size(label_index)
        padded_frame_1, padded_frame_2 = pad_images(frame_1, frame_2, pad_size)
        print(
            f"--- Evaluating label for patch size of {pad_size*2+1} for each pixel ---")
        for pixel_row in range(0, frame_1.shape[0]):
            for pixel_col in range(0, frame_1.shape[1]):

                pixel_coordinates = (pixel_row, pixel_col)  # p
                pixel_label = label[pixel_row][pixel_col]  # fp
                pixel_neighbours = [(pixel_row, pixel_col+1), (pixel_row+1, pixel_col),
                                    (pixel_row, pixel_col-1), (pixel_row-1, pixel_col)]  # q

                # Computing rho_D for pixel (i,j)
                rho_D = penalty(pixel_coordinates,
                                pixel_label, padded_frame_1, padded_frame_2, pad_size)

                # Computing rho_S for pixel (i,j)
                rho_S = 0
                for i in range(0, 4):
                    neighbour_coordinates = pixel_neighbours[i]

                    # Check if the 4-pixel neighbour is inside the matrix
                    # If it's not, I leave its coordinates to zero.
                    current_label_neighbour = np.zeros(1, dtype='i,i')
                    if (neighbour_coordinates[0] > 0 or neighbour_coordinates[1] > 0) or (neighbour_coordinates[0] < labels.shape[0] or neighbour_coordinates[1] < labels.shape[1]):
                        current_label_neighbour[0] = (
                            neighbour_coordinates[0], neighbour_coordinates[1])

                    f_difference = (pixel_label[0] - current_label_neighbour[0][0],
                                    pixel_label[1] - current_label_neighbour[0][1])  # fp - fq
                    reg_factor = regularization(f_difference)
                    # I1 at pixel_coord, I2 at pixel_neighbours
                    w = weight(padded_frame_1[pixel_coordinates[0]][pixel_coordinates[1]],
                               padded_frame_2[neighbour_coordinates[0]
                                              ][neighbour_coordinates[1]])
                    rho_S += w*reg_factor

                # Computing objective for pixel (i,j)
                r = rho_D + (Lambda * rho_S)
                Energy[iteration][label_index] = r
                # Checking if the label is suitable for pixel (i,j)

                if r < theta:
                    P[iteration][label_index] = 1

                iteration += 1
    np.savetxt('demo_output/P.csv', P, delimiter=',')
    np.savetxt('demo_output/Energy.csv', Energy, delimiter=',')

    return P, Energy
