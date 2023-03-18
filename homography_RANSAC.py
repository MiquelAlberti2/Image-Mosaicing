import numpy as np



def compute_homography(corresp):
    """
    Computes the coefficients of the homography that maps
    the correspondances from image 1 to image 2
    """
    n=len(corresp)

    # build the system of equations Ah=0
    A = np.empty((2*n, 9))

    for i in range(n):
        pt1, pt2 =corresp[i][0], corresp[i][1]
        x1, y1 = pt1[0], pt1[1]
        x2, y2 = pt2[0], pt2[1]

        A[2*i] = np.array([x1, y1, 1, 0, 0, 0, -x1*x2, -y1*x2, -x2])
        A[2*i+1] = np.array([0, 0, 0, x1, y1, 1, -x1*y2, -y1*y2, -y2])

    # now we can solve the system
    At = np.transpose(A)

    # compute the SVD of A^t A
    u, s, vh = np.linalg.svd(np.matmul(At,A))

    # h is the column of U associated with the smallest singular value in S
    min_i = np.argmin(s)
    column_result = u[:,min_i]

    return np.reshape(column_result, (3, 3))

def compute_image(h,pt):
    homog_pt=[pt[0], pt[1], 1]
    homog_image=np.matmul(h,homog_pt)

    return [homog_image[0]/homog_image[2], homog_image[1]/homog_image[2]]

def estimate_homography(corresp):
    """
    INPUT
     - list of pairs (tuples) of matching points
    OUTPUT
     - Matrix of the homography that maps the first points in corresp
                                     into the second points in corresp
    """
    best_inliers = None
    best_score = 0

    thr = 5
    max_iter = 100 # number of iterations 
    iter=0
    exit = False
    while iter < max_iter and not exit:
        inliers = []
        # Randomly sample 4 points (minimum required to determine homography in P2)
        i_sample = np.random.choice(len(corresp), 4, replace=False)

        corresp_sample = [corresp[i_sample[0]],
                          corresp[i_sample[1]],
                          corresp[i_sample[2]],
                          corresp[i_sample[3]]]


        # Fit an homography with the small sample
        h = compute_homography(corresp_sample)

        # Compute inliers
        for match in corresp:
            pt1, pt2 = match[0], match[1]
            # compute image 
            image2=compute_image(h,pt1)
            dist = ((image2[0] - pt2[0])**2 + (image2[1] - pt2[1])**2)**(1/2)

            if dist < thr:
                inliers.append((pt1,pt2))

        score = len(inliers)      
        
        # Update best parameters if score is better
        if score > best_score:
            best_inliers = inliers
            best_score = score
            print("current best score: ", best_score)

            if best_score > max((4*len(corresp)/5),4):
                # Is good enough
                exit=True

        iter+=1
        
    if not exit:
        print("maximum number of iterations reached with RANSAC!")
        
    # Compute the final homography with all inliers
    return compute_homography(best_inliers), best_inliers

