import numpy as np
import matplotlib.pyplot as plt
from skimage import draw

# Other files in the project
from NCC import find_correspondances
from homography_RANSAC import estimate_homography


def plot_correspondaces(img1, img2, corresp):
    img3 = np.zeros((img1.shape[0]+img2.shape[0], max(img1.shape[1], img2.shape[1])))
    img3[:img1.shape[0], :img1.shape[1]] = img1
    img3[img1.shape[0]:, :img2.shape[1]] = img2

    fig, ax = plt.subplots(figsize=(10,10))
    ax.imshow(img3, cmap='gray')

    for match in corresp:
        pt1 = match[0]
        pt2 = match[1]
        r, c = draw.line(pt1[0], pt1[1], pt2[0] + img1.shape[0], pt2[1])
        ax.plot(c, r, linewidth=0.4, color='blue')

    plt.show()


def homography(grey_img1, grey_img2, features1, features2):

    ##################
    # Find correspondaces between the two sets of cornes
    ##################

    # get a list containing all the feature correspondances
    corresp = find_correspondances(grey_img1, grey_img2, features1, features2)

    print("\nNumber of initial correspondances: ", len(corresp))
    # plot the correspondaces between the two images
    plot_correspondaces(grey_img1, grey_img2, corresp)

    ##################
    # Estimate the homography using RANSAC to ignore outliers
    ##################

    # estimate the homography matrix from the computed correspondances
    homography, inliers = estimate_homography(corresp)

    print("\nNumber of inliers after RANSAC: ", len(inliers))
    # plot the correspondaces between the two images
    plot_correspondaces(grey_img1, grey_img2, inliers)

    return homography
