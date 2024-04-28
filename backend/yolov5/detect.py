# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""

Usage - sources:
    $ python3 detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/LNwODJXcvt4'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

from utils.torch_utils import select_device, smart_inference_mode
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    strip_optimizer,
    xyxy2xywh,
)
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from models.common import DetectMultiBackend
from ultralytics.utils.plotting import Annotator, colors, save_one_box
import argparse
import csv
import os
from pymongo import MongoClient
import platform
from openai import OpenAI
import sys
from pathlib import Path
import threading
import cv2
import requests
from dotenv import load_dotenv
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import pandas as pd

import torch
load_dotenv()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
debug = os.environ.get("DEBUG")


# Retrieve environment variables
MONGO_URI = os.getenv('MONGO_URI')
CLOUDFLARE_ACCOUNT_ID = os.getenv('CLOUDFLARE_ACCOUNT_ID')
CLOUDFLARE_API_TOKEN = os.getenv('CLOUDFLARE_API_TOKEN')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Connect to the MongoDB server (adjust the connection string as needed)
mclient = MongoClient(MONGO_URI)

# Select the database and collection you want to use
db = mclient['hackdavis2024']
collection = db['python']

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def process_detections(frame_index, detections, frame_image, url_array):
    df = pd.DataFrame(columns=[
        'image', 'condition', 'authenticity', 'packaging_and_accessories', 'damage', 'returnable', 'used'])
    print(f"Frame {frame_index}:")
    string_detections = str(detections)

    if "cell phone" in string_detections.lower():
        _, buffer = cv2.imencode('.png', frame_image)
        files = {'file': ('frame_{}.png'.format(frame_index),
                          buffer.tobytes(), 'image/png')}
        headers = {'Authorization': f'Bearer {CLOUDFLARE_API_TOKEN}'}
        url = f'https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/images/v1'
        response = requests.post(url, files=files, headers=headers)

        if response.status_code == 200:
            print("Image successfully uploaded to Cloudflare.")
            # Print the full JSON response to help debug
            response_data = response.json()
            print("Full JSON Response:", response_data)
            # Extract and print the URL from the response
            variants = response_data.get('result', {}).get('variants', [])
            if variants:
                print("Retrieval URL:", variants[0])
                url_array.append(variants[0])
                length = len(url_array)
                print(
                    f'IMAGE APPENDED TO ARRAY, CURRENT ARRAY LENGTH: {length}')
            else:
                print("No variants URL found in the response.")
        else:
            print(f"Failed to upload image: {response.text}")

    if len(url_array) > 2:
        print('ARRAY REACHED LENGTH 3 PROCEEDING')
        best_image_index = getBestImage(url_array)
        print(f'Best image index retrieved: {best_image_index}')
        best_image_index = int(best_image_index)-1
        best_image_url = url_array[best_image_index]
        print(f'Best image url retrieved: {best_image_url}')
        print('getting condition')
        condition = getCondition(best_image_url)
        print(condition)
        print('getting authenticity')
        authenticity = getAuthenticity(best_image_url)
        print(authenticity)
        print('getting panda')
        packaging_and_accessories = getPandA(best_image_url)
        print(packaging_and_accessories)
        print('getting damage')
        damage = getDamage(best_image_url)
        print(damage)
        complete_content = condition + authenticity + packaging_and_accessories + damage
        returnable = getReturnFinal(complete_content)
        print(f"Return Status: {returnable}")
        df.loc[len(df)] = [best_image_url, condition,
                           authenticity, packaging_and_accessories, damage, returnable, True]
        print(f'RESULT DF:')
        print(df)
        print('Uploading to MongoDB')
        json_data = df.to_dict(orient='records')
        collection.insert_many(json_data)
        print('Upload Complete')
        url_array = []
        best_image_url = None
    return df, url_array

    # for detection in detections:
    #    if isinstance(detection, dict):
    #        print(
    #            f"  Class: {detection['class']}, Confidence: {detection['confidence']}, Box: {detection['bbox']}")
    #    else:
    #        print("Error: Detection data is not in the expected format (dict).")

    # after 3 images have been collected send all 3 to gpt4 and assess which one is the best to use. Kill the process.


def getBestImage(url_array):
    url1 = url_array[0]
    url2 = url_array[1]
    url3 = url_array[2]
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Which of the following images is the least blurry. Return only the number of the image in an integer format ie. 0. Do not return anything other than the number of the image. ",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": url1,
                        },
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": url2,
                        },
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": url3,
                        },
                    },
                ],
            }
        ],
        max_tokens=4096,
    )

    return response.choices[0].message.content

    # return the index of the image to use and then use that index to navigate to the image url
    # post image url to more gpt4 api requests answering questions. Format answers in a dictionary with keys:


def getCondition(imageUrl):
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze the attached image of a returned product. Identify any signs of wear and tear, such as scratches, stains, or alterations. Compare the current condition to the expected condition based on typical usage and provide a summary of discrepancies. Answer in 1 sentence."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": imageUrl,
                        },
                    },
                ],
            }
        ],
        max_tokens=4096,
    )

    return response.choices[0].message.content


def getAuthenticity(imageUrl):
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Examine the attached image for logos, labels, and serial numbers. Confirm the authenticity of the product by comparing these features to known genuine product characteristics. Report any inconsistencies or signs that suggest the product might be counterfeit. Answer in 1 sentence."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": imageUrl,
                        },
                    },
                ],
            }
        ],
        max_tokens=4096,
    )

    return response.choices[0].message.content


def getPandA(imageUrl):
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Assess the attached image to verify that all original packaging and accessories are present with the returned item. List any missing elements based on the standard packaging contents for this product. Answer in 1 sentence."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": imageUrl,
                        },
                    },
                ],
            }
        ],
        max_tokens=4096,
    )

    return response.choices[0].message.content


def getDamage(imageUrl):
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Inspect the attached image for any new damages not reported at the time of purchase, such as cracks or functional impairments. Describe the type and extent of any damages found and evaluate their impact on the functionality and aesthetics of the product. Answer in 1 sentence."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": imageUrl,
                        },
                    },
                ],
            }
        ],
        max_tokens=4096,
    )

    return response.choices[0].message.content


def getReturnFinal(content):
    response = client.chat.completions.create(
        model="gpt-4",  # or "gpt-4" depending on your preference and access
        messages=[
            {"role": "system", "content": "Given the following content determine if the product is suitable for a return. Return a 1 for yes and a 0 for no."},
            {"role": "user", "content": f"Content: {content}"}
        ]
    )
    return response.choices[0].message.content


'''
condition: The software can analyze images to assess wear and tear, scratches, stains, or any alterations. This helps determine if the item matches the condition described by the customer and if it adheres to the return policy’s condition requirements.
authenticity: By examining certain features such as logos, labels, and serial numbers in the images, the software can verify the authenticity of the product, ensuring it's not a counterfeit and matches the product sold.
??? comparison_with_original Product Images: The software could compare the return images with original images from the product's sale (if available) to check for discrepancies in color, size, or features, which might indicate a wrong or manipulated item being returned.
Packaging_and_Accessories: Images can show whether the original packaging and all accessories are included with the return, as required by many return policies.
damage_assessment: Detailed images can help identify new damages that were not present at the time of purchase, such as cracks or functional damages that might be visible only on close inspection.
'''

# send dictionary to mongodb via pymongo

# done.


@smart_inference_mode()
def run(
    weights=ROOT / "yolov5s.pt",  # model path or triton URL
    source=ROOT / "data/images",  # file/dir/URL/glob/screen/0(webcam)
    data=ROOT / "data/coco128.yaml",  # dataset.yaml path
    imgsz=(640, 640),  # inference size (height, width)
    conf_thres=0.25,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=1000,  # maximum detections per image
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img=False,  # show results
    save_txt=False,  # save results to *.txt
    save_csv=False,  # save results in CSV format
    save_conf=False,  # save confidences in --save-txt labels
    save_crop=False,  # save cropped prediction boxes
    nosave=False,  # do not save images/videos
    classes=None,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    augment=False,  # augmented inference
    visualize=False,  # visualize features
    update=False,  # update all models
    project=ROOT / "runs/detect",  # save results to project/name
    name="exp",  # save results to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    line_thickness=3,  # bounding box thickness (pixels)
    hide_labels=False,  # hide labels
    hide_conf=False,  # hide confidences
    half=False,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    vid_stride=1,  # video frame-rate stride
):
    source = str(source)
    save_img = not nosave and not source.endswith(
        ".txt")  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    webcam = source.isnumeric() or source.endswith(
        ".streams") or (is_url and not is_file)
    screenshot = source.lower().startswith("screen")
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name,
                              exist_ok=exist_ok)  # increment run
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True,
                                                          exist_ok=True)  # make dir

    # Load model
    url_array = []
    device = select_device(device)
    model = DetectMultiBackend(
        weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz,
                              stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(
            source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz,
                             stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(device=device), Profile(
        device=device), Profile(device=device))

    detection_data = []
    frame_count = 0

    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            if model.xml and im.shape[0] > 1:
                ims = torch.chunk(im, im.shape[0], 0)

        # Inference
        with dt[1]:
            visualize = increment_path(
                save_dir / Path(path).stem, mkdir=True) if visualize else False
            if model.xml and im.shape[0] > 1:
                pred = None
                for image in ims:
                    if pred is None:
                        pred = model(image, augment=augment,
                                     visualize=visualize).unsqueeze(0)
                    else:
                        pred = torch.cat(
                            (pred, model(image, augment=augment, visualize=visualize).unsqueeze(0)), dim=0)
                pred = [pred, None]
            else:
                pred = model(im, augment=augment, visualize=visualize)
        # NMS
        with dt[2]:
            pred = non_max_suppression(
                pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Define the path for the CSV file
        csv_path = save_dir / "predictions.csv"

        # Create or append to the CSV file
        def write_to_csv(image_name, prediction, confidence):
            """Writes prediction data for an image to a CSV file, appending if the file exists."""
            data = {"Image Name": image_name,
                    "Prediction": prediction, "Confidence": confidence}
            with open(csv_path, mode="a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=data.keys())
                if not csv_path.is_file():
                    writer.writeheader()
                writer.writerow(data)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f"{i}: "
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / "labels" / p.stem) + \
                ("" if dataset.mode == "image" else f"_{frame}")  # im.txt
            s += "%gx%g " % im.shape[2:]  # print string
            # normalization gain whwh
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(
                im0, line_width=line_thickness, example=str(names))

            if len(det):
                det[:, :4] = scale_boxes(
                    im.shape[2:], det[:, :4], im0.shape).round()

                # Collect detection information
                current_detections = []
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    label = names[c] if hide_conf else f"{names[c]} {conf:.2f}"
                    bbox = [int(x) for x in xyxy]
                    current_detections.append(
                        {'class': names[c], 'confidence': float(conf), 'bbox': bbox})

                    # Optional: annotate the image (if you still want to do this)
                    annotator.box_label(xyxy, label, color=colors(c, True))

                detection_data.extend(current_detections)

            # Process data every 10 frames
            if frame % 10 == 0:
                response, url_array = process_detections(
                    frame, detection_data, im0, url_array)
                detection_data = []  # Reset for the next batch of frames

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = names[c] if hide_conf else f"{names[c]}"
                    confidence = float(conf)
                    confidence_str = f"{confidence:.2f}"

                    if save_csv:
                        write_to_csv(p.name, label, confidence_str)

                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)
                                          ) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)
                        with open(f"{txt_path}.txt", "a") as f:
                            f.write(("%g " * len(line)).rstrip() % line + "\n")

                    if save_img or save_crop or view_img:  # Add bbox to image
                        label = None if hide_labels else (
                            names[c] if hide_conf else f"{names[c]} {conf:.2f}")
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / "crops" /
                                     names[c] / f"{p.stem}.jpg", BGR=True)

            im0 = annotator.result()
            if view_img:
                if platform.system() == "Linux" and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL |
                                    cv2.WINDOW_KEEPRATIO)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == "image":
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            # release previous video writer
                            vid_writer[i].release()
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        # force *.mp4 suffix on results videos
                        save_path = str(Path(save_path).with_suffix(".mp4"))
                        vid_writer[i] = cv2.VideoWriter(
                            save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(
            f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image
    LOGGER.info(
        f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}" % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ""
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        # update model (to fix SourceChangeWarning)
        strip_optimizer(weights[0])


def parse_opt():
    """Parses command-line arguments for YOLOv5 detection, setting inference options and model configurations."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str,
                        default=ROOT / "yolov5s.pt", help="model path or triton URL")
    parser.add_argument("--source", type=str, default=ROOT /
                        "data/images", help="file/dir/URL/glob/screen/0(webcam)")
    parser.add_argument("--data", type=str, default=ROOT /
                        "data/coco128.yaml", help="(optional) dataset.yaml path")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+",
                        type=int, default=[640], help="inference size h,w")
    parser.add_argument("--conf-thres", type=float,
                        default=0.25, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float,
                        default=0.45, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=1000,
                        help="maximum detections per image")
    parser.add_argument("--device", default="",
                        help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--view-img", action="store_true", help="show results")
    parser.add_argument("--save-txt", action="store_true",
                        help="save results to *.txt")
    parser.add_argument("--save-csv", action="store_true",
                        help="save results in CSV format")
    parser.add_argument("--save-conf", action="store_true",
                        help="save confidences in --save-txt labels")
    parser.add_argument("--save-crop", action="store_true",
                        help="save cropped prediction boxes")
    parser.add_argument("--nosave", action="store_true",
                        help="do not save images/videos")
    parser.add_argument("--classes", nargs="+", type=int,
                        help="filter by class: --classes 0, or --classes 0 2 3")
    parser.add_argument("--agnostic-nms", action="store_true",
                        help="class-agnostic NMS")
    parser.add_argument("--augment", action="store_true",
                        help="augmented inference")
    parser.add_argument("--visualize", action="store_true",
                        help="visualize features")
    parser.add_argument("--update", action="store_true",
                        help="update all models")
    parser.add_argument("--project", default=ROOT /
                        "runs/detect", help="save results to project/name")
    parser.add_argument("--name", default="exp",
                        help="save results to project/name")
    parser.add_argument("--exist-ok", action="store_true",
                        help="existing project/name ok, do not increment")
    parser.add_argument("--line-thickness", default=3,
                        type=int, help="bounding box thickness (pixels)")
    parser.add_argument("--hide-labels", default=False,
                        action="store_true", help="hide labels")
    parser.add_argument("--hide-conf", default=False,
                        action="store_true", help="hide confidences")
    parser.add_argument("--half", action="store_true",
                        help="use FP16 half-precision inference")
    parser.add_argument("--dnn", action="store_true",
                        help="use OpenCV DNN for ONNX inference")
    parser.add_argument("--vid-stride", type=int, default=1,
                        help="video frame-rate stride")
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    """Executes YOLOv5 model inference with given options, checking requirements before running the model."""
    check_requirements(ROOT / "requirements.txt",
                       exclude=("tensorboard", "thop"))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
