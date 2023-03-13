import numpy as np

# Non-maximum suppression
from skimage.feature import peak_local_max


def apply_Kernel(img, kernel):

    nrow=img.shape[0]
    ncol=img.shape[1]

    filt_img = np.zeros_like(img)

    pad_size = int(kernel.shape[0]/2) 
    pad_image = np.pad(img, pad_size, mode='constant')

    for i in range(pad_size, nrow + pad_size):
        for j in range(pad_size, ncol + pad_size):
                filt_img[i-pad_size,j-pad_size] = (kernel*pad_image[i-pad_size:i+pad_size+1, j-pad_size:j+pad_size+1]).sum()

    return filt_img


def detect_features(img):
    """
    INPUT
     - numpy array representing a grey img
    OUTPUT
     - mask of the same size as the img with 1s at the corners locations
    """
    # create kernel for Sobel mask
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # apply Sobel mask to compute the gradients I_x, I_y
    I_x = apply_Kernel(img, kernel_x)
    I_y = apply_Kernel(img, kernel_y)

    # compute the C matrix
    I_x2 = np.square(I_x)
    I_y2 = np.square(I_y)
    I_xy = I_x * I_y

    # Smooth out with a 5x5 box filter
    box_kernel = np.zeros((5,5))
    box_kernel.fill(1./(5*5))

    I_x2 = apply_Kernel(I_x2, box_kernel)
    I_y2 = apply_Kernel(I_y2, box_kernel)
    I_xy = apply_Kernel(I_xy, box_kernel)

    # we will use a C matrix of 1 × 1 neighborhood to save computations
    # so it is just C=[[I_x2, I_xy],[I_xy, I_y2]] at the corresponding pixel value

    # compute the score R instead of the eigenvalues
    det=(I_x2*I_y2)-(I_xy*I_xy)
    trace=I_x2+I_y2

    R=det-0.05*(trace**2)

    # ¿¿Threshold the harris response???
    # when R>0 and relativelly big, we have a corner
    threshold = 0.1 * np.max(R)
    corners = peak_local_max(R, min_distance=5, threshold_abs=threshold, exclude_border=True)
    

    return corners