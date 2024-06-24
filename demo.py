import os
import cv2
import time
import glob
import random
import argparse
import numpy as np
from scipy.io import loadmat
import sys
from problems.disjoint_set_cover import DisjointSetCover
from utils.data_generator import get_preference_matrix_fm
from utils.save_results import save_flow_image

dataset_dir = 'dataset/'
OUTFOLDER = "demo_output/"
random.seed()
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")
np.set_printoptions(threshold=sys.maxsize)

if not os.path.exists(OUTFOLDER):
    os.makedirs(OUTFOLDER)

# Function to load image frames. (Tentative)


def load_frames(base_folder):
    # 0_img, 385_img
    frame1_dir = base_folder + '0_img1.png'
    frame2_dir = base_folder + '0_img2.png'
    frame1 = cv2.imread(frame1_dir, cv2.IMREAD_GRAYSCALE)
    frame2 = cv2.imread(frame2_dir, cv2.IMREAD_GRAYSCALE)
    scale = 1

    f1_rows, f1_cols = frame1.shape
    frame1_resized = cv2.resize(
        frame1, (int(f1_rows / scale), int(f1_cols / scale)))
    f2_rows, f2_cols = frame2.shape
    frame2_resized = cv2.resize(
        frame2, (int(f2_rows / scale), int(f2_cols / scale)))

    return frame1_resized, frame2_resized


"""
Tentative labels generation function.
We start by generating random labels in a 3x3 neighborhood and then gradually using larger
and larger neighborhoods (5x5, 7x7, 9x9 and 11x11). For each neighborhood, 2+2*i labels are
generated (where i is the number of neighborhood, e.g. for 3x3 i=0, for 5x5 i=1, for 7x7 i=2 and so on).
When extracting labels, I make sure to exclude labels for smaller neighbors (e.g. when computing the labels for the 5x5
neighbor, labels that fall into the 3x3 neighborhood are excluded).
"""


def generate_random_coordinates(patch_size, number_of_coordinates):
    coord_set = []
    biggest_value = int(patch_size/2)
    for i in range(0, number_of_coordinates):
        a = biggest_value if random.random() < 0.5 else -biggest_value
        b = random.randint(-biggest_value, biggest_value)
        coords = (a, b) if random.random() > 0.5 else (a, b)
        coord_set.append(coords)
    return coord_set


def generate_labels(image_shape):

    num_pixels = image_shape[0] * image_shape[1]
    # tentative flow field
    labels_matrix = np.zeros((num_pixels, 30), dtype='i,i')

    for pixel in range(0, num_pixels):
        # 2 labels are first generated
        labels_to_generate = 2
        # I keep track of generated labels
        number_of_generated_labels = 0

        current_patch_size = 3
        list_of_labels = []
        # Looping until 30 as I'm generating 30 labels.
        while number_of_generated_labels < 30:

            labels = generate_random_coordinates(
                current_patch_size, labels_to_generate)
            number_of_generated_labels += labels_to_generate

            labels_to_generate += 2
            current_patch_size += 2
            list_of_labels.append(labels)

        # List of labels is then flattened and converted to an np array of type (int, int).
        flattened_list_of_labels = [
            item for items in list_of_labels for item in items]
        labels_matrix[pixel] = np.asarray(
            flattened_list_of_labels, dtype='i,i')

    return labels_matrix


"""
The computed preference matrix and z vector are passed to this function. The goal is to extract the columns
of the matrix P whose value in the corresponding position of the z vector is 1.
"""


def union_function(P, z_hat):

    possible_candidates = []
    possible_indices = []
    for index, element in enumerate(z_hat):
        if element == 0:
            continue
        possible_indices.append(index)
        possible_candidates.append(P[:, [index]])

    possible_candidates = np.array(np.reshape(possible_candidates,
                                              (P.shape[0], len(possible_indices))))

    return possible_candidates, possible_indices


"""
Extracting the best labels from the possible_candidates matrix. The chosen ones are the
ones with the lowest energy computed when generating P.
"""


def extracting_optical_flow(possible_candidates, possible_indices, Energy, labels):
    optical_flow = np.zeros(labels.shape[0], dtype='i,i')

    for row in range(possible_candidates.shape[0]):
        min_energy = float('inf')
        min_index = -1
        candidate_indices = np.nonzero(possible_candidates[row])[0]
        # Checking the row element with the lowest energy

        for col in candidate_indices:
            current_energy = Energy[row][col]
            if current_energy < min_energy:
                min_energy = current_energy
                min_index = possible_indices[col]

        # The label corresponding to the minimum energy is saved to the optical flow vector.
        if min_index != -1:
            optical_flow[row] = labels[row][min_index]

    return optical_flow


def evaluating_preference_matrix(P, Energy, labels, z_hat):
    possible_candidates, possible_indices = union_function(P, z_hat)
    optical_flow = extracting_optical_flow(
        possible_candidates, possible_indices, Energy, labels)

    return optical_flow


def main():
    # Input parameters for the energy function. Default values are still tentative.
    parser = argparse.ArgumentParser(
        description='Parameters for the energy function.')
    parser.add_argument('-r', '--reg_type', nargs='?', default='L1',
                        help='Regularization type.')
    parser.add_argument('-th', '--theta', nargs='?', default=1.5,
                        help='Threshold value used when populating P matrix.')
    parser.add_argument('-b', '--beta', nargs='?', default=1,
                        help='Value used when computing weights in the energy function.')
    parser.add_argument('-l', '--Lambda', nargs='?', default=1,
                        help='Value that\'s multiplied to the second member in the energy function.')
    parser.add_argument('-ta', '--tao', nargs='?', default=1,
                        help='Thershold used for the first member of the energy function.')

    args = parser.parse_args()
    # Loading and preparing frames (attualmente sono grayscale per semplicità)
    frame_1, frame_2 = load_frames(dataset_dir)
    # frame_1 = cv2.Canny(frame_1, 100, 200)
    # frame_2 = cv2.Canny(frame_2, 100, 200)
    # Generating labels
    labels = generate_labels(frame_1.shape)

    #### --- configurations for the algorithm (by default DeQuMF(SA) is set) --- ####

    # to use DeQuMF (SA), uncomment here
    # dsc = DisjointSetCover(sampler_type="sa", decompose=True)

    # to use QuMF, uncomment here
    # dsc = DisjointSetCover(sampler_type="qa", decompose=False)

    # to use DeQuMF, uncomment here
    # dsc = DisjointSetCover(sampler_type="qa", decompose=True)

    # to use QuMF (SA), uncomment here
    dsc = DisjointSetCover(sampler_type="sa", decompose=False)

    ### --- additional parameters --- ###
    reg_type = args.reg_type.upper()
    theta = float(args.theta)
    beta = float(args.beta)
    Lambda = float(args.Lambda)
    tao = float(args.tao)
    # If an invalid regularization parameter is entered, I just default to L1.
    if reg_type != 'L1' and reg_type != 'L2' and reg_type != 'C':
        reg_type = 'L1'
    print("Chosen values:")
    print(
        f"Theta: {theta}\nBeta: {beta}\nLambda: {Lambda}\nTao: {tao}\nRegularization: {reg_type}")
    # Preference-Consensus is obtained
    P, Energy = get_preference_matrix_fm(
        frame_1, frame_2, labels, reg_type, theta, beta, Lambda, tao)
    print("Done!")
    print("Computing z vector with chosen sampler...")

    # P = np.loadtxt('demo_output/P.csv', delimiter=',')
    # Energy = np.loadtxt('demo_output/Energy.csv', delimiter=',')

    z_hat = dsc(P)
    print(z_hat)
    print("Done!")

    print("Computing optical flow...")
    optical_flow = evaluating_preference_matrix(P, Energy, labels, z_hat)

    print("Done!")
    print("Results are in the work directory.")
    # Checking for results

    save_flow_image(frame_1, frame_2, optical_flow, OUTFOLDER)


if __name__ == "__main__":
    main()
