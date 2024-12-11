import pyzed.sl as sl
import cv2
import numpy as np

# Initialize the ZED camera
camera = sl.Camera()

# Create configuration parameters
init_params = sl.InitParameters()
init_params.camera_fps = 30
init_params.camera_resolution = sl.RESOLUTION.HD1080
init_params.depth_mode = sl.DEPTH_MODE.NEURAL  # Use high-accuracy depth mode
init_params.coordinate_units = sl.UNIT.MILLIMETER  # Depth in millimeters
init_params.depth_minimum_distance = 100
init_params.depth_maximum_distance = 5000  # Maximum depth is 5 meters (5000 mm)

runtime_params = sl.RuntimeParameters()
runtime_params.enable_fill_mode = True

# ArUco detection setup
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# Open the camera
if camera.open(init_params) != sl.ERROR_CODE.SUCCESS:
    print("Failed to open camera")
    exit(1)

# Get camera calibration parameters for triangulation
calibration_params = camera.get_camera_information().camera_configuration.calibration_parameters
left_intrinsics = calibration_params.left_cam
right_intrinsics = calibration_params.right_cam

# Convert intrinsics to OpenCV format
# Construct intrinsic matrices
K_left = np.array([
    [left_intrinsics.fx, 0, left_intrinsics.cx],
    [0, left_intrinsics.fy, left_intrinsics.cy],
    [0, 0, 1]
])

K_right = np.array([
    [right_intrinsics.fx, 0, right_intrinsics.cx],
    [0, right_intrinsics.fy, right_intrinsics.cy],
    [0, 0, 1]
])
D_left = np.array(left_intrinsics.disto)
D_right = np.array(right_intrinsics.disto)
T = np.array([calibration_params.stereo_transform[0, -1], 0, 0])  # Only x translation is
T /= 1000

# Construct projection matrices
P_left = np.hstack((K_left, np.zeros((3, 1))))  # [K_left | 0]
T_mat = np.array([[1, 0, 0, -T[0]],  # Translation for right camera
                  [0, 1, 0, 0],
                  [0, 0, 1, 0]])
P_right = K_right @ T_mat

try:
    while True:
        left_image = sl.Mat()
        right_image = sl.Mat()

        if camera.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            # Retrieve LEFT and RIGHT images
            camera.retrieve_image(left_image, sl.VIEW.LEFT)
            camera.retrieve_image(right_image, sl.VIEW.RIGHT)

            left_ocv = cv2.cvtColor(left_image.get_data(), cv2.COLOR_RGBA2RGB)
            right_ocv = cv2.cvtColor(right_image.get_data(), cv2.COLOR_RGBA2RGB)

            # Detect ArUco markers in both views
            gray_left = cv2.cvtColor(left_ocv, cv2.COLOR_BGR2GRAY)
            gray_right = cv2.cvtColor(right_ocv, cv2.COLOR_BGR2GRAY)

            corners_left, ids_left, _ = detector.detectMarkers(gray_left)
            corners_right, ids_right, _ = detector.detectMarkers(gray_right)

            # Draw detected markers
            left_ocv = cv2.aruco.drawDetectedMarkers(left_ocv, corners_left, ids_left)
            right_ocv = cv2.aruco.drawDetectedMarkers(right_ocv, corners_right, ids_right)

            if ids_left is not None and ids_right is not None:
                # Match markers detected in both views
                for i, id_left in enumerate(ids_left.flatten()):
                    if id_left in ids_right:
                        idx_right = np.where(ids_right == id_left)[0][0]

                        # Get corresponding corners
                        left_corners = np.mean(corners_left[i][0], axis=0)  # Average marker corners
                        right_corners = np.mean(corners_right[idx_right][0], axis=0)

                        # Triangulate 3D position
                        left_points = np.array([[left_corners[0], left_corners[1]]], dtype=np.float32).T
                        right_points = np.array([[right_corners[0], right_corners[1]]], dtype=np.float32).T

                        points_4d = cv2.triangulatePoints(P_left, P_right, left_points, right_points)
                        points_3d = points_4d[:3] / points_4d[3]  # Convert to 3D coordinates

                        # Annotate 3D position on the LEFT view
                        cv2.putText(left_ocv, f"ID {id_left}: {points_3d.ravel()}",
                                    (int(left_corners[0]), int(left_corners[1])),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Display the images
            cv2.imshow('ZED Camera Left View', left_ocv)
            cv2.imshow('ZED Camera Right View', right_ocv)

            # Exit on 'q' key
            if cv2.waitKey(33) & 0xFF == ord('q'):  # ~30 FPS
                break
        else:
            print("Failed to grab frame!")
except KeyboardInterrupt:
    pass
finally:
    camera.close()
    cv2.destroyAllWindows()
