import numpy as np
import cv2 as cv
import open3d as o3d


# Load the 3D model and its texture
model = o3d.io.read_triangle_mesh("./SkinBodyFinal.obj")
model.compute_vertex_normals()

# Calculate the axis-aligned bounding box of the model
bbox = model.get_axis_aligned_bounding_box()
min_bound = bbox.min_bound
max_bound = bbox.max_bound
max_size = max(max_bound - min_bound)

# Chessboard settings
board_cellsize = 0.012  # Size of a chessboard square (in meters)

# Calculate scale to fit the model in one chessboard cell
scale_factor = board_cellsize / max_size

# Scale the model to fit it within one chessboard square
model.vertices = o3d.utility.Vector3dVector(np.asarray(model.vertices) * scale_factor)
model.compute_vertex_normals()  # Recompute normals for proper lighting


# 비디오 및 칼리브레이션 데이터
video_file = './comeon.mp4'
K = np.array([[1109.82750, 0.00000000, 387.259466],
              [0.00000000, 1062.77784, 505.511935],
              [0.00000000, 0.00000000, 1.00000000]])
dist_coeff = np.array([[-0.256707375, 3.59427512, 0.00686362362, 0.00643457911, -12.7418562]])
board_pattern = (10, 7)
board_cellsize = 0.012
board_criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)  # 수정된 부분



# 비디오 열기
video = cv.VideoCapture(video_file)
assert video.isOpened(), 'Cannot read the given input, ' + video_file

# 체스보드의 3D 포인트 준비
obj_points = board_cellsize * np.array([[c, r, 0] for r in range(board_pattern[1]) for c in range(board_pattern[0])])

# Mesh to Points
vertices = np.asarray(model.vertices)
colors = np.asarray(model.vertex_colors)


while True:
    # 비디오 프레임 읽기
    ret, img = video.read()
    if not ret:
        break
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # 카메라 자세 추정
    success, img_points = cv.findChessboardCorners(gray, board_pattern, board_criteria)

    if success:
        img_points = cv.cornerSubPix(gray, img_points, (11, 11), (-1, -1),board_criteria)
        ret, rvec, tvec = cv.solvePnP(obj_points, img_points, K, dist_coeff)

        # Project the vertices of the 3D model to the 2D image plane

        imgpts, _ = cv.projectPoints(vertices, rvec, tvec, K, dist_coeff)

        # Draw the mesh as wireframe
        imgpts = np.int32(imgpts).reshape(-1, 2)
        for i, j, k in model.triangles:
            cv.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (0, 255, 0), 1)
            cv.line(img, tuple(imgpts[j]), tuple(imgpts[k]), (0, 255, 0), 1)
            cv.line(img, tuple(imgpts[k]), tuple(imgpts[i]), (0, 255, 0), 1)


        # 카메라 포즈 정보 출력
        p = (-np.linalg.inv(K) @ tvec).flatten()
        info = f'XYZ: [{p[0]:.3f} {p[1]:.3f} {p[2]:.3f}]'
        cv.putText(img, info, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 이미지 표시
    cv.imshow('Pose Estimation (Chessboard)', img)
    if cv.waitKey(1) == 27:  # ESC 키로 종료
        break

# 자원 해제
video.release()
cv.destroyAllWindows()

