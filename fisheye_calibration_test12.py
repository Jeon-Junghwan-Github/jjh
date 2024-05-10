import numpy as np
import cv2
import glob

import numpy as np
import cv2
import glob

# Step 0 (Setting)
# input the images
image_path = 'calibration_images/*.jpg'

# Define the chess board rows and columns
CHECKERBOARD = (6, 9)
subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW
objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane

# Step 1
# Getting the images including checker points, information about checker points from original images
for i,path in enumerate(glob.glob(image_path)):
    # Load the image and convert it to gray scale
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_NORMALIZE_IMAGE)
    # Make sure the chess board pattern was found in the image
    if ret:
        objpoints.append(objp)
        better_corners = cv2.cornerSubPix(gray,corners,(3,3),(-1,-1),subpix_criteria)
        imgpoints.append(better_corners)
        #cv2.drawChessboardCorners(img, (rows, cols), corners, ret)
    print(f"{i+1} : {str(path)}")
    # count of images
    N_imm = i+1

# Step 2
# Function >> (checker points information of images, checker points) -> parameters
# Input >> image information : objpoints, imgpoints, image size
# Output >> parameters : rms, K, D, rvecs, tvecs
rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(objpoints, imgpoints, gray.shape[::-1], None, None, None, None,
    calibration_flags,(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6))

# Step 3
# Undistort images and display
vstack_images = []
for i, fname in enumerate(glob.glob(image_path)):
    img = cv2.imread(fname)
    h, w = img.shape[:2]

    # 새로운 카메라 매트릭스를 계산하여 왜곡 보정
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        K, D, (w, h), 1, (w, h)
    )

    # 왜곡 보정
    undistorted_img = cv2.fisheye.undistortImage(
        img, K, D, None, new_camera_matrix
    )

    # ROI(Region of Interest)에 맞게 이미지 잘라내기
    '''
    x, y, w, h = roi
    undistorted_image = undistorted_image[y : y + h, x : x + w]
    '''
    print(f"{i+1} : {roi}")

    img_half1 = cv2.pyrDown(img)
    undistorted_img_half1 = cv2.pyrDown(undistorted_img)

    img_half2 = cv2.pyrDown(img_half1)
    undistorted_img_half2 = cv2.pyrDown(undistorted_img_half1)
    
    print(img_half2.shape[:2], undistorted_img_half2.shape[:2])
    result_vertical = np.vstack([img_half2, undistorted_img_half2])
    vstack_images.append(result_vertical)

final_result = np.hstack([image for image in vstack_images])
cv2.imshow('final result', final_result)
cv2.waitKey(0)


cv2.destroyAllWindows()
