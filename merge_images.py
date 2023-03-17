import numpy as np
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt

def compute_image(h,pt):
    homog_pt=[pt[0], pt[1], 1]
    homog_image=np.matmul(h,homog_pt)

    return [homog_image[0]/homog_image[2], homog_image[1]/homog_image[2]]

def warp_image1_onto_image2(img1, img2, homography):
    """
    INPUT
     - img1: RGB image represented as a Numpy array
     - img2: RGB image represented as a Numpy array
     - homography that maps img1 into img2
    OUTPUT
     - final mosaic with img1 warped into img2
    """

    # create output image with the appropiate size to fit both original images
    n1, c1 = img1.shape[0], img1.shape[1]
    n2, c2 = img2.shape[0], img2.shape[1]

    # get the images of the corners to see how big the output should be
    # we warp image 1 into image 2, so we just need the images of image 1
    img1_corner1 = compute_image(homography, [0,0])
    img1_corner2 = compute_image(homography, [0,c1])
    img1_corner3 = compute_image(homography, [n1,c1])
    img1_corner4 = compute_image(homography, [n1,0])

    min_row = int(min(img1_corner1[0], img1_corner2[0], img1_corner3[0], img1_corner4[0],0))
    max_row = int(max(img1_corner1[0], img1_corner2[0], img1_corner3[0], img1_corner4[0],n2))
    min_col = int(min(img1_corner1[1], img1_corner2[1], img1_corner3[1], img1_corner4[1],0))
    max_col = int(max(img1_corner1[1], img1_corner2[1], img1_corner3[1], img1_corner4[1],c2))

    mosaic = np.zeros((max_row - min_row, max_col - min_col, 3), dtype='uint8')

    # copy image 2 into the solution
    mosaic[-min_row:n2-min_row, -min_col:c2-min_col,:] = img2 # min_row and min_col could be negative

    # Display the output mosaic
    plt.imshow(mosaic)
    plt.show()
    

    # warp image 1 using the inverse of the homography (backward warping)
    h_inverse = np.linalg.inv(homography)

    # Interpolate the pixel values of each input image at the transformed coordinates of the meshgrid
    interpR = interp2d(np.arange(c1), np.arange(n1), img1[:, :, 0], kind='linear') # it first takes the y component
    interpG = interp2d(np.arange(c1), np.arange(n1), img1[:, :, 1], kind='linear')
    interpB = interp2d(np.arange(c1), np.arange(n1), img1[:, :, 2], kind='linear')

    for i in range(mosaic.shape[0]):
        for j in range(mosaic.shape[1]):
            # Let's continue with the assumption that the camera is rotating along the Y_axis
            # So we can distinguish the following cases:
            coord_img1_y, coord_img1_x = compute_image(h_inverse,[i+min_row, j+min_col])

            in_img1, in_img2 = 0,0
            # if it's outside the frame, no blending
            if coord_img1_x > 0 and coord_img1_x < c1 and coord_img1_y > 0 and coord_img1_y < n1:
                in_img1=0.5

                # make sure that the values are between 0 and 255
                img1_R = max(0, min(interpR(coord_img1_x, coord_img1_y), 255)) 
                img1_G = max(0, min(interpG(coord_img1_x, coord_img1_y), 255))
                img1_B = max(0, min(interpB(coord_img1_x, coord_img1_y), 255))

            if i > 0 and i < n2 and j > 0 and j < c2:
                in_img2=0.5

            if in_img1+in_img2>0: # If the value is not in any of the images, it has to remain black
                mosaic[i,j,0] = int((in_img2*mosaic[i,j,0] + in_img1*img1_R)/(in_img1+in_img2))
                mosaic[i,j,1] = int((in_img2*mosaic[i,j,1] + in_img1*img1_G)/(in_img1+in_img2))
                mosaic[i,j,2] = int((in_img2*mosaic[i,j,2] + in_img1*img1_B)/(in_img1+in_img2))
           
    return mosaic