import numpy as np
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt

def compute_image(h,pt):
    homog_pt=[pt[0], pt[1], 1]
    homog_image=np.matmul(h,homog_pt)

    return [homog_image[0]/homog_image[2], homog_image[1]/homog_image[2]]

def warp_images(img1, img2, img3, homography1, homography3):
    """
    INPUT
     - img1: RGB image represented as a Numpy array
     - img2: RGB image represented as a Numpy array
     - img3: RGB image represented as a Numpy array
     - homography1: that maps img1 into img2
     - homography3: that maps img3 into img2
    OUTPUT
     - final mosaic with img1 and img3 warped into img2
    """

    # create output image with the appropiate size to fit all original images
    n1, c1 = img1.shape[0], img1.shape[1]
    n2, c2 = img2.shape[0], img2.shape[1]
    n3, c3 = img3.shape[0], img3.shape[1]

    # get the images of the corners to see how big the output should be
    # we use the coordinate system of img2, so we don't need to transform its coordinates
    img1_corner1 = compute_image(homography1, [0,0])
    img1_corner2 = compute_image(homography1, [0,c1])
    img1_corner3 = compute_image(homography1, [n1,c1])
    img1_corner4 = compute_image(homography1, [n1,0])

    img3_corner1 = compute_image(homography3, [0,0])
    img3_corner2 = compute_image(homography3, [0,c3])
    img3_corner3 = compute_image(homography3, [n3,c3])
    img3_corner4 = compute_image(homography3, [n3,0])

    min_row = int(min(img1_corner1[0], img1_corner2[0], img1_corner3[0], img1_corner4[0],
                      img3_corner1[0], img3_corner2[0], img3_corner3[0], img3_corner4[0],
                      0))
    max_row = int(max(img1_corner1[0], img1_corner2[0], img1_corner3[0], img1_corner4[0],
                      img3_corner1[0], img3_corner2[0], img3_corner3[0], img3_corner4[0],
                      n2))
    min_col = int(min(img1_corner1[1], img1_corner2[1], img1_corner3[1], img1_corner4[1],
                      img3_corner1[1], img3_corner2[1], img3_corner3[1], img3_corner4[1],
                      0))
    max_col = int(max(img1_corner1[1], img1_corner2[1], img1_corner3[1], img1_corner4[1],
                      img3_corner1[1], img3_corner2[1], img3_corner3[1], img3_corner4[1],
                      c2))

    mosaic = np.zeros((max_row - min_row, max_col - min_col, 3), dtype='uint8')

    # copy img2 into the solution
    mosaic[-min_row:n2-min_row, -min_col:c2-min_col,:] = img2 # min_row and min_col could be negative

    # warp image 1 using the inverse of the homography (backward warping)
    h_inverse1 = np.linalg.inv(homography1)
    h_inverse3 = np.linalg.inv(homography3)

    # Interpolate the pixel values of img1 and img3
    interpR1 = interp2d(np.arange(c1), np.arange(n1), img1[:, :, 0], kind='linear')
    interpG1 = interp2d(np.arange(c1), np.arange(n1), img1[:, :, 1], kind='linear')
    interpB1 = interp2d(np.arange(c1), np.arange(n1), img1[:, :, 2], kind='linear')

    interpR3 = interp2d(np.arange(c3), np.arange(n3), img3[:, :, 0], kind='linear')
    interpG3 = interp2d(np.arange(c3), np.arange(n3), img3[:, :, 1], kind='linear')
    interpB3 = interp2d(np.arange(c3), np.arange(n3), img3[:, :, 2], kind='linear')

    for i in range(mosaic.shape[0]):
        for j in range(mosaic.shape[1]):
            # express the point in the coordinate system of img1 and img3
            coord_img1_y, coord_img1_x = compute_image(h_inverse1,[i+min_row, j+min_col])
            coord_img3_y, coord_img3_x = compute_image(h_inverse3,[i+min_row, j+min_col])

            img1_R, img1_G, img1_B, img3_R, img3_G, img3_B = 0,0,0,0,0,0
            in_img1, in_img2, in_img3 = 0,0,0
            # if it's outside the frame, no blending
            if coord_img1_x > 0 and coord_img1_x < c1 and coord_img1_y > 0 and coord_img1_y < n1:
                in_img1=0.3 # if its inside the frame, we have to consider its color in the final mosaic

                # to make sure that the values are between 0 and 255
                img1_R = max(0, min(interpR1(coord_img1_x, coord_img1_y), 255)) 
                img1_G = max(0, min(interpG1(coord_img1_x, coord_img1_y), 255))
                img1_B = max(0, min(interpB1(coord_img1_x, coord_img1_y), 255))

            if coord_img3_x > 0 and coord_img3_x < c3 and coord_img3_y > 0 and coord_img3_y < n3:
                in_img3=0.3

                # to make sure that the values are between 0 and 255
                img3_R = max(0, min(interpR3(coord_img3_x, coord_img3_y), 255)) 
                img3_G = max(0, min(interpG3(coord_img3_x, coord_img3_y), 255))
                img3_B = max(0, min(interpB3(coord_img3_x, coord_img3_y), 255))

            if i > -min_row and i < n2-min_row and j > -min_col and j < c2-min_col:
                in_img2=0.3

            if in_img1+in_img2+in_img3>0: # If the value is not in any of the images, it has to remain black
                mosaic[i,j,0] = int((in_img2*mosaic[i,j,0] + in_img1*img1_R + in_img3*img3_R)/(in_img1+in_img2+in_img3))
                mosaic[i,j,1] = int((in_img2*mosaic[i,j,1] + in_img1*img1_G + in_img3*img3_G)/(in_img1+in_img2+in_img3))
                mosaic[i,j,2] = int((in_img2*mosaic[i,j,2] + in_img1*img1_B + in_img3*img3_B)/(in_img1+in_img2+in_img3))
           
    return mosaic