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

            # Retrieve depth map
            camera.retrieve_measure(depth, sl.MEASURE.DEPTH)

            # Cap depth at 5000 mm and scale to 8-bit
            depth_data = depth.get_data()
            depth_data[depth_data > 5000] = 5000  # Cap at 5 meters
            depth_ocv = (depth_data / 5000 * 255).astype(np.uint8)  # Scale to 8-bit range

            # Display the images
            cv2.imshow('ZED Camera Live Feed', image_ocv)
            cv2.imshow('ZED Camera Depth', depth_ocv)

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
