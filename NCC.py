import numpy as np


def compute_NCC(img1, img2, pt1, pt2):
    # compute NCC of a neighbourhood around pt1 and pt2 of size 2n+1
    n = 1

    x1, y1 = pt1[0], pt1[1]
    x2, y2 = pt2[0], pt2[1]

    if x1>n-1 and x1<img1.shape[0]-n and y1>n-1 and y1<img1.shape[1]-n:
        patch1=img1[x1-n:x1+n+1, y1-n:y1+n+1]

    if x2>n-1 and x2<img2.shape[0]-n and y2>n-1 and y2<img2.shape[1]-n:
        patch2=img2[x2-n:x2+n+1, y2-n:y2+n+1] 

    # Compute the mean and variance of each patch
    mean1 = np.mean(patch1)
    mean2 = np.mean(patch2)

    n1 = ((patch1-mean1)**2).sum()
    n2 = ((patch2-mean2)**2).sum()

    N_corr = (((patch1 - mean1)*(patch2 - mean2)).sum())/(n1*n2)**(1/2)

    return N_corr


def find_correspondances(img1, img2, features1, features2):
    """
    INPUT
     - img1, img2: numpy array representing grey images
     - features1, features2: lists of coordinates where there is a corner
    OUTPUT
     - list of pairs (tuples) of matching points
    """
    corresp = []
    threshold = 0.8

    for pt1 in features1:
        for pt2 in features2:
            ncc = compute_NCC(img1, img2, pt1, pt2)

            # check match
            if ncc > threshold:
                corresp.append((pt1,pt2))

            # TODO, remove points when added to corresp
            # OR if there are two matches with the same point, just save the one with the higher NCC 

    return corresp