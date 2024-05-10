import numpy as np
import cv2

# 체스보드의 가로 및 세로 내부 코너의 수
# 체스보드에 대한 코너 검출
CHESSBOARD_SIZE = (6, 9)

# 각 체스보드 코너의 실제 3D 좌표 생성
objp = np.zeros((np.prod(CHESSBOARD_SIZE), 3), dtype=np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)

# 3D 포인트 저장할 배열과 2D 포인트 저장할 배열 생성
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

# 이미지 디렉토리 루프
for i in range(1, 6):
    # 이미지 파일 로드
    img = cv2.imread(f'calibration_images/{i}.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 체스보드 코너 찾기
    ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

    # 코너가 발견되면 3D 포인트와 2D 포인트 추가
    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

        # 코너 표시
        cv2.drawChessboardCorners(img, CHESSBOARD_SIZE, corners, ret)
        cv2.imshow('img', img)
        cv2.imwrite(f'result_images/{i}.jpg', img)
        cv2.waitKey(5000)

cv2.destroyAllWindows()

# 카메라 캘리브레이션 수행
ret, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
    objpoints,
    imgpoints,
    gray.shape[::-1],
    None,
    None
)

print("K = ", K)
print("D = ", D)