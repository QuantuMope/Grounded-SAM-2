import pyzed.sl as sl
import cv2
import numpy as np
import time

camera = sl.Camera()

# Create configuration parameters
init_params = sl.InitParameters()
init_params.camera_fps = 100
init_params.camera_resolution = sl.RESOLUTION.VGA

init_params.depth_mode = sl.DEPTH_MODE.NEURAL
# init_params.depth_mode = sl.DEPTH_MODE.QUALITY
# init_params.depth_mode = sl.DEPTH_MODE.NEURAL_PLUS
# init_params.depth_mode = sl.DEPTH_MODE.ULTRA
# init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE

init_params.coordinate_units = sl.UNIT.MILLIMETER  # Depth in millimeters
init_params.depth_minimum_distance = 100
init_params.depth_maximum_distance = 2000  # Maximum depth is 5 meters (5000 mm)
init_params.depth_stabilization = 100

runtime_params = sl.RuntimeParameters()
runtime_params.enable_fill_mode = True

# Open the camera
if camera.open(init_params) != sl.ERROR_CODE.SUCCESS:
    print("Failed to open camera")
    exit(1)

capture_time = 1. / init_params.camera_fps
try:
    while True:
        image = sl.Mat()
        depth = sl.Mat()

        s = time.perf_counter()
        if camera.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            # Retrieve RGB image
            camera.retrieve_image(image, sl.VIEW.LEFT)
            image_ocv = cv2.cvtColor(image.get_data(), cv2.COLOR_RGBA2RGB)

            # Retrieve depth map
            camera.retrieve_measure(depth, sl.MEASURE.DEPTH)

            # Cap depth at 5000 mm and scale to 8-bit
            depth_data = depth.get_data()
            holes_mask = np.isnan(depth_data)
            depth_ocv = (depth_data / init_params.depth_maximum_distance * 255).astype(np.uint8)  # Scale to 8-bit range
            
            depth_colormap = cv2.applyColorMap(depth_ocv, cv2.COLORMAP_JET)
            depth_colormap[holes_mask] = [ 0.0, 0.0, 255 ]
            # depth_colormap[holes_mask] = [ 255, 0.0, 0.0]

            # Display the images
            cv2.imshow('ZED Camera Live Feed', image_ocv)
            cv2.imshow('ZED Camera Depth', depth_colormap)

            time_passed = (time.perf_counter() - s)
            wait_time = 1
            if time_passed < capture_time:
                wait_time = capture_time - time_passed
                wait_time = int(wait_time * 1000)
                if wait_time == 0:
                    wait_time = 1

            if cv2.waitKey(wait_time) & 0xFF == ord("q"):
                break
        else:
            print("Failed to grab frame!")
except KeyboardInterrupt:
    pass
finally:
    camera.close()
    cv2.destroyAllWindows()
