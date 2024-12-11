import pyzed.sl as sl
import cv2
import numpy as np

camera = sl.Camera()

# Create configuration parameters
init_params = sl.InitParameters()
init_params.camera_fps = 30
init_params.camera_resolution = sl.RESOLUTION.HD1080
init_params.depth_mode = sl.DEPTH_MODE.NEURAL  # Use high-accuracy depth mode
# init_params.depth_mode = sl.DEPTH_MODE.ULTRA  # Use high-accuracy depth mode
init_params.coordinate_units = sl.UNIT.MILLIMETER  # Depth in millimeters
init_params.depth_minimum_distance = 100
init_params.depth_maximum_distance = 5000  # Maximum depth is 5 meters (5000 mm)

runtime_params = sl.RuntimeParameters()
runtime_params.enable_fill_mode = True

# ArUco detection setup
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)  # Use a 4x4 dictionary
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

# Open the camera
if camera.open(init_params) != sl.ERROR_CODE.SUCCESS:
    print("Failed to open camera")
    exit(1)


try:
    while True:
        image = sl.Mat()
        depth = sl.Mat()

        if camera.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            # Retrieve RGB image
            camera.retrieve_image(image, sl.VIEW.LEFT)
            image_ocv = cv2.cvtColor(image.get_data(), cv2.COLOR_RGBA2RGB)

            # # Retrieve depth map
            # camera.retrieve_measure(depth, sl.MEASURE.DEPTH)
            #
            # # Cap depth at 5000 mm and scale to 8-bit
            # depth_data = depth.get_data()
            # depth_data[depth_data > 5000] = 5000  # Cap at 5 meters
            # depth_ocv = (depth_data / 5000 * 255).astype(np.uint8)  # Scale to 8-bit range
            
            # ArUco tag detection
            gray_image = cv2.cvtColor(image_ocv, cv2.COLOR_BGR2GRAY)
            corners, ids, rejected = detector.detectMarkers(gray_image)

            if ids is not None:
                # Draw detected markers
                image_ocv = cv2.aruco.drawDetectedMarkers(image_ocv, corners, ids)

                # Extract 3D positions of marker corners
                point_cloud = sl.Mat()
                camera.retrieve_measure(point_cloud, sl.MEASURE.XYZ)

                for marker_corners in corners:
                    for corner in marker_corners[0]:  # Each marker has 4 corners
                        x, y = int(corner[0]), int(corner[1])
                        if 0 <= x < point_cloud.get_width() and 0 <= y < point_cloud.get_height():
                            # Get the 3D position of the corner
                            success, point3d = point_cloud.get_value(x, y)  # Returns a tuple: (x, y, z, confidence)
                            x3d, y3d, z3d, confidence = point3d
                            # print(confidence)
                            # if confidence > 0:  # Check if confidence is high enough
                            cv2.putText(
                                image_ocv,
                                f"({x3d:.1f}, {y3d:.1f}, {z3d:.1f})",
                                (x, y),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (255, 0, 0),
                                1,
                            )

                # # Optionally, get depth for marker corners
                # for marker_corners in corners:
                #     for corner in marker_corners[0]:  # Each marker has 4 corners
                #         x, y = int(corner[0]), int(corner[1])
                #         if 0 <= x < depth_ocv.shape[1] and 0 <= y < depth_ocv.shape[0]:
                #             depth_value = depth_data[y, x]
                #             cv2.putText(image_ocv, f"{depth_value:.0f}mm", (x, y),
                #                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Display the images
            cv2.imshow('ZED Camera Live Feed', image_ocv)
            # cv2.imshow('ZED Camera Depth', depth_ocv)

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
