import numpy as np
import cv2
import glob

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

K = np.zeros((3, 3))
D = np.zeros((4, 1))
rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_imm)]
tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_imm)]
rms, _, _, _, _ = cv2.fisheye.calibrate(
    objpoints,
    imgpoints,
    gray.shape[::-1],
    K,
    D,
    rvecs,
    tvecs,
    calibration_flags,
    (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6))

print(f"K")
print(K)
print()
print(f"D")
print(D)
map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, (1280, 800), cv2.CV_16SC2)

for path in glob.glob(image_path):
    img =cv2.imread(path)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    cv2.imshow('Original Image', img)
    cv2.imshow('Undistort Image', undistorted_img)
    cv2.waitKey(5000)

cv2.destroyAllWindows()