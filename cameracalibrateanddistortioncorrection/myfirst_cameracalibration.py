import cv2 as cv
import numpy as np

import numpy as np
import cv2

def select_img_from_video(video_file, board_pattern, wnd_name='Camera Calibration'):
    video = cv2.VideoCapture(video_file)
    assert video.isOpened(), "Video file couldn't be opened"

    img_select = []

    while True:
        ret, img = video.read()
        if not ret:
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, board_pattern, None)

        # 체스보드 코너가 발견된 경우, 이미지를 선택 목록에 추가
        if ret:
            cv2.drawChessboardCorners(img, board_pattern, corners, ret)
            img_select.append(img)

            cv2.imshow(wnd_name, img)
            cv2.waitKey(1)  # 잠시 딜레이를 줘서 이미지를 보여줌

    cv2.destroyAllWindows()
    return img_select

def calib_camera_from_chessboard(images, board_pattern, board_cellsize, K=None, dist_coeff=None, calib_flags=None):
    objp = np.zeros((board_pattern[0]*board_pattern[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_pattern[0], 0:board_pattern[1]].T.reshape(-1, 2) * board_cellsize

    objpoints = []
    imgpoints = []

    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        complete, corners = cv2.findChessboardCorners(gray, board_pattern)
        if complete:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            objpoints.append(objp)
            imgpoints.append(corners2)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], K, dist_coeff, flags=calib_flags)
    return ret, mtx, dist, rvecs, tvecs

if __name__ == '__main__':
    video_file = 'comeon.mp4'  # 비디오 파일 경로를 여러분의 비디오 경로로 수정하세요.
    board_pattern = (10, 7)
    board_cellsize = 0.012  # 실제 체스보드 셀의 크기 (미터 단위)

    # 비디오에서 자동으로 체스보드 코너를 찾는 프레임 선택
    img_select = select_img_from_video(video_file, board_pattern)
    assert len(img_select) > 0, 'No images with detectable chessboard corners were selected!'

    # 카메라 캘리브레이션 수행
    rms, K, dist_coeff, rvecs, tvecs = calib_camera_from_chessboard(img_select, board_pattern, board_cellsize)

    # 카메라 내부 매개 변수 출력 (행렬 형태)
    print("카메라 내부 매개 변수 (행렬 형태):")
    print(K)

    # 카메라 내부 매개 변수 출력 (fx, fy, cx, cy, ...)
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    # RMSE 값 출력
    print("\n카메라 내부 매개 변수:")
    print(f"(fx, fy, cx, cy, ..., rmse): ({fx}, {fy}, {cx}, {cy}, ..., {rms})")

    # 렌즈 왜곡 매개 변수 출력
    print("\n렌즈 왜곡 매개 변수:")
    print(dist_coeff)



'''
import numpy as np
import cv2 as cv

def select_img_from_video(video_file, board_pattern, select_all=False, wait_msec=10, wnd_name='Camera Calibration'):
    # Open a video
    video = cv.VideoCapture(video_file)
    assert video.isOpened()

    # Select images
    img_select = []
    while True:
        # Grab an images from the video
        valid, img = video.read()
        if not valid:
            break

        if select_all:
            img_select.append(img)
        else:
            # Show the image
            display = img.copy()
            cv.putText(display, f'NSelect: {len(img_select)}', (10, 25), cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0))
            cv.imshow(wnd_name, display)

            # Process the key event
            key = cv.waitKey(wait_msec)
            if key == ord(' '):             # Space: Pause and show corners
                complete, pts = cv.findChessboardCorners(img, board_pattern)
                cv.drawChessboardCorners(display, board_pattern, pts, complete)
                cv.imshow(wnd_name, display)
                key = cv.waitKey()
                if key == ord('\r'):
                    img_select.append(img) # Enter: Select the image
            if key == 27:                  # ESC: Exit (Complete image selection)
                break

    cv.destroyAllWindows()
    return img_select

def calib_camera_from_chessboard(images, board_pattern, board_cellsize, K=None, dist_coeff=None, calib_flags=None):
    # Find 2D corner points from given images
    img_points = []
    for img in images:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        complete, pts = cv.findChessboardCorners(gray, board_pattern)
        if complete:
            img_points.append(pts)
    assert len(img_points) > 0

    # Prepare 3D points of the chess board
    obj_pts = [[c, r, 0] for r in range(board_pattern[1]) for c in range(board_pattern[0])]
    obj_points = [np.array(obj_pts, dtype=np.float32) * board_cellsize] * len(img_points) # Must be `np.float32`

    # Calibrate the camera
    return cv.calibrateCamera(obj_points, img_points, gray.shape[::-1], K, dist_coeff, flags=calib_flags)

if __name__ == '__main__':
    video_file = 'comeon.mp4'
    board_pattern = (10, 7)
    board_cellsize = 0.012

    img_select = select_img_from_video(video_file, board_pattern)
    assert len(img_select) > 0, 'There is no selected images!'
    rms, K, dist_coeff, rvecs, tvecs = calib_camera_from_chessboard(img_select, board_pattern, board_cellsize)

    # Print calibration results
    print('## Camera Calibration Results')
    print(f'* The number of selected images = {len(img_select)}')
    print(f'* RMS error = {rms}')
    print(f'* Camera matrix (K) = \n{K}')
    print(f'* Distortion coefficient (k1, k2, p1, p2, k3, ...) = {dist_coeff.flatten()}')

'''



'''
def select_img_from_video(video_file, board_pattern, select_all=False, wait_msec=10):
    """
    Open a video and select images that contain a complete chessboard.
    """
    video = cv.VideoCapture(video_file)
    if not video.isOpened():
        raise IOError("Error opening video file.")

    img_select = []
    while True:
        ret, frame = video.read()
        if not ret:
            break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        complete, _ = cv.findChessboardCorners(gray, board_pattern)

        if complete or select_all:
            cv.imshow('Select Image', frame)
            key = cv.waitKey(wait_msec) & 0xFF
            if key == ord('s') or select_all:
                img_select.append(frame)
            elif key == ord('q'):
                break

    cv.destroyAllWindows()
    video.release()
    return img_select


def calib_camera_from_chessboard(images, board_pattern, board_cellsize, K=None, dist_coeff=None, calib_flags=None):
    global img_points, obj_points_all  # 전역 변수로 사용될 리스트를 선언합니다.
    img_points = []  # 2D 이미지 포인트를 저장할 리스트
    obj_points_all = []  # 3D 오브젝트 포인트를 저장할 리스트

    obj_pts = [[c, r, 0] for r in range(board_pattern[1]) for c in range(board_pattern[0])]
    obj_points = np.array(obj_pts, dtype=np.float32) * board_cellsize

    for img in images:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        complete, pts = cv.findChessboardCorners(gray, board_pattern)
        if complete:
            cv.cornerSubPix(gray, pts, (11, 11), (-1, -1),
                            criteria=(cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            img_points.append(pts)
            obj_points_all.append(obj_points)  # 각 이미지에 대응하는 3D 포인트 세트를 추가합니다.

    assert len(img_points) > 0, 'There is no set of complete chessboard points!'

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(obj_points_all, img_points, gray.shape[::-1], K, dist_coeff,
                                                      flags=calib_flags)
    return ret, mtx, dist, rvecs, tvecs


import cv2 as cv
import numpy as np


def undistort_image(image, camera_matrix, dist_coeffs):
    """
    이미지의 렌즈 왜곡을 보정합니다.

    :param image: 왜곡을 보정할 이미지
    :param camera_matrix: 카메라 캘리브레이션으로 얻은 카메라 매트릭스
    :param dist_coeffs: 카메라 캘리브레이션으로 얻은 왜곡 계수
    :return: 보정된 이미지
    """
    h, w = image.shape[:2]
    new_camera_mtx, roi = cv.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))

    # 이미지 왜곡 보정
    undistorted_img = cv.undistort(image, camera_matrix, dist_coeffs, None, new_camera_mtx)

    # 옵션: ROI를 사용하여 결과 이미지를 자를 수 있음
    x, y, w, h = roi
    undistorted_img = undistorted_img[y:y + h, x:x + w]

    return undistorted_img


# Example usage
# Example usage
if __name__ == "__main__":
    video_file = "camcali.mp4"

    board_pattern = (10, 7)
    board_cellsize = 0.012

    selected_images = select_img_from_video(video_file, board_pattern, select_all=False, wait_msec=500)
    ret, mtx, dist, rvecs, tvecs = calib_camera_from_chessboard(selected_images, board_pattern, board_cellsize)

    # RMSE 계산 부분을 수정합니다.
    total_error = 0
    for i in range(len(selected_images)):
        imgpoints2, _ = cv.projectPoints(obj_points_all[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(img_points[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
        total_error += error

    rmse = np.sqrt(total_error / len(selected_images))

    print(f"Calibration successful: {ret}")
    print(f"Camera matrix:\n{mtx}")
    print(f"Distortion coefficients:\n{dist.ravel()}")
    print(f"RMSE: {rmse}")

'''