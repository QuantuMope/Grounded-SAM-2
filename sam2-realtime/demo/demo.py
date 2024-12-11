import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import imageio
from torchvision.ops import box_convert
import pyzed.sl as sl
import sys


# use bfloat16 for the entire notebook
autocast_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
autocast_context.__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

from sam2.build_sam import build_sam2_camera_predictor
import time
sys.path.append("..")
from grounding_dino.groundingdino.util.inference import (load_model,
                                                         load_image, predict)


""" Setup GROUNDING DINO model """
GROUNDING_DINO_CONFIG = "../grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "../gdino_checkpoints/groundingdino_swint_ogc.pth"
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
# TEXT_PROMPT = "big shark."
TEXT_PROMPT = sys.argv[1]

grounding_model = load_model(
    model_config_path=GROUNDING_DINO_CONFIG,
    model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
    device="cuda"
)

""" Setup SAM2 model """
model_size = "large"
if model_size == "small":
    sam2_checkpoint = "checkpoints/sam2_hiera_small.pt"
    model_cfg = "sam2_hiera_s.yaml"
else:
    sam2_checkpoint = "checkpoints/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"
predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)

""" Setup ZED camera """
# debug
# cap = cv2.VideoCapture("notebooks/videos/aquarium/aquarium.mp4")
camera = sl.Camera()

# Create configuration parameters
init_params = sl.InitParameters()
init_params.camera_fps = 15
init_params.camera_resolution = sl.RESOLUTION.HD1080
# init_params.camera_resolution = sl.RESOLUTION.HD720
# init_params.depth_mode = sl.DEPTH_MODE.NEURAL  # Use high-accuracy depth mode
init_params.depth_mode = sl.DEPTH_MODE.NONE  # Use high-accuracy depth mode
# init_params.depth_mode = sl.DEPTH_MODE.ULTRA  # Use high-accuracy depth mode
init_params.coordinate_units = sl.UNIT.MILLIMETER  # Depth in millimeters
init_params.depth_minimum_distance = 100
init_params.depth_maximum_distance = 5000  # Maximum depth is 5 meters (5000 mm)

runtime_params = sl.RuntimeParameters()
runtime_params.enable_fill_mode = True

capture_time = 1. / init_params.camera_fps

# Open the camera
if camera.open(init_params) != sl.ERROR_CODE.SUCCESS:
    print("Failed to open camera")
    exit(1)


if_init = False


def query_dino(frame: np.ndarray, verbose: bool = True):
    image_source, image = load_image(frame)

    autocast_context.__exit__(None, None, None)
    boxes, confidences, labels = predict(
        model=grounding_model,
        image=image,
        caption=TEXT_PROMPT,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
    )
    autocast_context.__enter__()

    # process the box prompt for SAM 2
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    # confidences = confidences.numpy().tolist()
    class_names = labels

    if verbose:
        print(input_boxes)
        print(class_names)
    return input_boxes, class_names


while True:
    image = sl.Mat()
    depth = sl.Mat()

    if camera.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
        # Retrieve RGB image
        camera.retrieve_image(image, sl.VIEW.LEFT)
        frame = cv2.cvtColor(image.get_data(), cv2.COLOR_RGBA2RGB)

        # Retrieve depth map
        # camera.retrieve_measure(depth, sl.MEASURE.DEPTH)
        #
        # # Cap depth at 5000 mm and scale to 8-bit
        # depth_data = depth.get_data()
        # depth_data[depth_data > 5000] = 5000  # Cap at 5 meters
        # depth_ocv = (depth_data / 5000 * 255).astype(np.uint8)  # Scale to 8-bit range

        # # Display the images
        # cv2.imshow('ZED Camera Live Feed', image_ocv)
        # cv2.imshow('ZED Camera Depth', depth_ocv)
        #
        # # Exit on 'q' key
        # if cv2.waitKey(33) & 0xFF == ord('q'):  # ~30 FPS
        #     break

    else:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    width, height = frame.shape[:2][::-1]
    s = time.perf_counter()
    if not if_init:
        s = time.perf_counter()
        input_boxes, class_names = query_dino(frame)
        print("grounding dino: ", time.perf_counter() - s)

        predictor.load_first_frame(frame)
        if_init = True

        ann_frame_idx = 0  # the frame index we interact with
        ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

        for bbox_coord, name in zip(input_boxes, class_names):
            bbox = np.array(bbox_coord, dtype=np.float32).reshape((2, 2))
            _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
                frame_idx=ann_frame_idx, obj_id=ann_obj_id, bbox=bbox
            )
            ann_obj_id += 1

        ##! add mask
        # mask_img_path="../notebooks/masks/aquarium/aquarium_mask.png"
        # mask = cv2.imread(mask_img_path, cv2.IMREAD_GRAYSCALE)
        # mask = mask / 255

        # _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
        #     frame_idx=ann_frame_idx, obj_id=ann_obj_id, mask=mask
        # )

    else:
        out_obj_ids, out_mask_logits = predictor.track(frame)
        # print("sam2 tracking: ", time.perf_counter() - s)
        # print(out_obj_ids)
        # print(out_mask_logits.mean())

        all_mask = np.zeros((height, width, 1), dtype=np.uint8)
        # print(all_mask.shape)
        for i in range(0, len(out_obj_ids)):
            out_mask = (out_mask_logits[i] > 0.0).permute(1, 2, 0).cpu().numpy().astype(
                np.uint8
            ) * 255

            all_mask = cv2.bitwise_or(all_mask, out_mask)
            # Convert the mask to blue color
            mask = np.zeros((height, width, 3), dtype=np.uint8)
            mask[:, :, i] = all_mask

            frame = cv2.addWeighted(frame, 1, mask, 0.75, 0)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    time_passed = (time.perf_counter() - s)
    cv2.imshow("frame", frame)
    wait_time = 1
    if time_passed < capture_time:
        wait_time = capture_time - time_passed
        wait_time = int(wait_time * 1000)
        if wait_time == 0:
            wait_time = 1

    if cv2.waitKey(wait_time) & 0xFF == ord("q"):
        break

camera.close()
cv2.destroyAllWindows()
# cap.release()
# gif = imageio.mimsave("./result.gif", frame_list, "GIF", duration=0.00085)
