import numpy as np

# Non-maximum suppression
from skimage.feature import peak_local_max

def size_gaussian_mask(std):

    size=5*std

    # the number should be integer
    if not size.is_integer():
        size=int(5*std) + 1
        
    # we want an odd size
    if size%2==0:
        size+=1

    size = int(size)

    return size, int(size/2)
    
def compute_Gauss_filter(std):
    
    size, half_s = size_gaussian_mask(std)

    mask = np.zeros((size,size))

    half_s = int(size/2)

    for i in range(-half_s,half_s+1):
        for j in range(-half_s,half_s+1):
            mask[i+half_s,j+half_s] = np.exp((-i*i - j*j)/(2*std*std))

    # Factor to normalize the mask
    k = np.sum(mask)

    # to get the separable filter, we just take the first row
    d_mask = (1/(k**(1/2)))*mask[0]

    return d_mask, half_s


def apply_Gauss_smoothing(img,std):

    #create a kernel for a separated gaussian filter
    kernel, padding_size = compute_Gauss_filter(std)

    nrow=img.shape[0]
    ncol=img.shape[1]

    # create a "horizontal" 0 padding
    padded_image = np.pad(img, ((0, 0), (padding_size, padding_size)), mode='constant')

    
    vertical_filt_img = np.zeros_like(img)

    # apply the horizontal filter
    for i in range(nrow):
        for j in range(padding_size, ncol + padding_size):
                vertical_filt_img[i,j-padding_size] = (kernel*padded_image[i, j-padding_size:j+padding_size+1]).sum()


    # create a "vertical" 0 padding
    vertical_filt_img = np.pad(vertical_filt_img, ((padding_size, padding_size), (0, 0)), mode='constant')
    
    filt_img = np.zeros_like(img)

    # apply the vertical filter
    for i in range(padding_size, nrow + padding_size):
        for j in range(ncol):
                filt_img[i-padding_size,j] = (kernel*vertical_filt_img[i-padding_size:i+padding_size+1, j]).sum()

    return filt_img

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
     - list of coordinates of the detected features
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

    # apply a smoothing filter, because derivatives are very sensitive to noise
    std=0.7
    I_x2 = apply_Gauss_smoothing(I_x2, std)
    I_y2 = apply_Gauss_smoothing(I_y2, std)
    I_xy = apply_Gauss_smoothing(I_xy, std)

    R = np.zeros_like(img)

    # we compute the C matrix using a neighborhood of 7x7 (ignoring borders)
    for i in range(3, img.shape[0]-3):
        for j in range(3, img.shape[1]-3):
            I_x2_sum=I_x2[i-3:i+4, j-3:j+4].sum()
            I_y2_sum=I_y2[i-3:i+4, j-3:j+4].sum()
            I_xy_sum=I_xy[i-3:i+4, j-3:j+4].sum()

            # compute the score R instead of the eigenvalues
            det=(I_x2_sum*I_y2_sum)-(I_xy_sum*I_xy_sum)
            trace=I_x2_sum+I_y2_sum

            R[i,j]=det-0.05*(trace**2)

    # when R>0 and relativelly big (bigger than threshold), we have a corner
    # return a list of the [x y] coordinates where there is a corner
    thr = np.mean(R) + np.std(R)
    corners = peak_local_max(R, min_distance=5, threshold_abs=thr, exclude_border=True)
    
    return corners