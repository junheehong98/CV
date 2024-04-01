import numpy as np
import cv2 as cv

# 수정된 카메라 내부 매개변수와 렌즈 왜곡 매개변수
K = np.array([[1109.82750, 0.00000000, 387.259466],
              [0.00000000, 1062.77784, 505.511935],
              [0.00000000, 0.00000000, 1.00000000]])
dist_coeff = np.array([[-0.256707375, 3.59427512, 0.00686362362, 0.00643457911, -12.7418562]])

# 비디오 파일 경로
video_file = 'comeon.mp4'  # 비디오 파일 경로를 실제 비디오 경로로 수정하세요.

video = cv.VideoCapture(video_file)
assert video.isOpened(), 'Cannot read the given input, ' + video_file

# 렌즈 왜곡 보정 실행
show_rectify = True
map1, map2 = None, None
while True:
    valid, img = video.read()
    if not valid:
        break

    info = "Original"
    if show_rectify:
        if map1 is None or map2 is None:
            # 수정된 카메라 내부 매개변수와 렌즈 왜곡 매개변수를 사용하여 매핑 계산
            map1, map2 = cv.initUndistortRectifyMap(K, dist_coeff, None, K, (img.shape[1], img.shape[0]), cv.CV_32FC1)
        img = cv.remap(img, map1, map2, interpolation=cv.INTER_LINEAR)
        info = "Rectified"
    cv.putText(img, info, (10, 25), cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0))

    cv.imshow("Geometric Distortion Correction", img)
    key = cv.waitKey(10)
    if key == ord(' '):     # Space: Pause
        key = cv.waitKey()
    if key == 27:           # ESC: Exit
        break
    elif key == ord('\t'):  # Tab: Toggle the mode
        show_rectify = not show_rectify

video.release()
cv.destroyAllWindows()
