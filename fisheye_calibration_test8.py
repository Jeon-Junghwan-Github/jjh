import numpy as np
import cv2
import glob

# 캘리브레이션을 위한 체스보드 그리드 크기
chessboard_size = (9, 6)

# 체스보드 코너를 탐지할 이미지들의 경로
images_path = "calibration_images/*.jpg"

# 체스보드 코너 탐지를 위한 객체 포인트와 이미지 포인트 초기화
obj_points = []  # 실제 3D 포인트
img_points = []  # 이미지 상의 2D 포인트

# 객체 포인트 생성 (0,0,0), (1,0,0), ..., (8,5,0)
objp = np.zeros((1,chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

# 보다 정확한 체스 포인트 매칭을 위한 튜플
# cv2.TERM_CRITERIA_EPS : 알고리즘에서의 상대적인 정확도를 나타냅니다.
# cv2.TERM_CRITERIA_MAX_ITER : 알고리즘에서의 최대 반복 횟수를 나타냅니다.
subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)

# 모든 이미지에서 체스보드 코너를 찾아 포인트들을 수집
images = glob.glob(images_path)
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 체스보드 코너 탐지
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    # 탐지된 코너가 있을 경우, 포인트 추가
    if ret == True:
        obj_points.append(objp)
        real_corners = cv2.cornerSubPix(gray,corners,(3,3),(-1,-1),subpix_criteria)
        img_points.append(real_corners)

# 카메라 매트릭스와 왜곡 계수 계산
ret, camera_matrix, dist_coeffs, _, _ = cv2.fisheye.calibrate(
    obj_points, img_points, gray.shape[::-1], None, None
)

# 보정된 이미지 생성
for i, fname in enumerate(images):
    img = cv2.imread(fname)
    h, w = img.shape[:2]

    # 새로운 카메라 매트릭스를 계산하여 왜곡 보정
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 1, (w, h)
    )

    # 왜곡 보정
    undistorted_image = cv2.fisheye.undistortImage(
        img, camera_matrix, dist_coeffs, None, new_camera_matrix
    )

    # ROI(Region of Interest)에 맞게 이미지 잘라내기
    '''
    x, y, w, h = roi
    undistorted_image = undistorted_image[y : y + h, x : x + w]
    '''
    
    # 결과 이미지 출력 또는 저장
    cv2.imshow(f"Undistorted Image {i}", undistorted_image)
    cv2.waitKey(3000)

cv2.destroyAllWindows()
