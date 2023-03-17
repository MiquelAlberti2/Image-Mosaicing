import numpy as np
import imageio.v3 as iio # to read and write images
import matplotlib.pyplot as plt
from skimage import draw
import cv2

# Other files in the project
from harris_corner_detector import detect_features
from NCC import find_correspondances
from homography_RANSAC import estimate_homography
from merge_images import warp_image1_onto_image2

def rgb_to_gray(rgb):

    # compute a weighted average of RGB colors to obtain a greyscale value
    # weights correspond to the luminosity of each color channel
    # we also normalize the image
    return (1/255)*np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

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


##################
# Read the two images
##################

images = []

for i in range(8, 18):
    num_image = str(i)
    if i<10:
        num_image ='0'+num_image       

    images.append(iio.imread(uri='DanaOffice/DSC_03'+num_image+'.JPG'))

# convert to grey images and reduce them by a scaling factor
grey_images = []

scale = 0.75
for i in range(len(images)):
    h, w = images[i].shape[0], images[i].shape[1]
    images[i] = cv2.resize(images[i], (int(scale*w), int(scale*h)))
    grey_images.append(rgb_to_gray(images[i]))

##################
# Detect features of both images
##################

grey_img1 = grey_images[0]
grey_img2 = grey_images[1]
image1 = images[0]
image2 = images[1]

# get masks with corner locations
features1 = detect_features(grey_img1)
features2 = detect_features(grey_img2)

# visualize the detected corners
plt.imshow(image1, cmap='gray')
plt.plot(features1[:, 1], features1[:, 0], 'r.', markersize=5)
plt.show()

plt.imshow(image2, cmap='gray')
plt.plot(features2[:, 1], features2[:, 0], 'r.', markersize=5)
plt.show()

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


##################
# Warp one image onto the other to obtain the final mosaic
##################

# obtain the final mosaic with the computed homography
mosaic = warp_image1_onto_image2(image1, image2, homography)

# Display the output mosaic
plt.imshow(mosaic)
plt.show()

# write the mosaic to disk
iio.imwrite(uri="output/mosaic.png", image=mosaic)

