import cv2

def image_to_sift(img):
    # create SIFT
    sift = cv2.SIFT_create()
    # Detect keypoints
    kp = []
    desc = []
    for i in range(len(img)):
        tmp_kp, tmp_desc = sift.detectAndCompute(img[i], None)
        kp.append(tmp_kp)
        desc.append(tmp_desc)

    return kp, desc