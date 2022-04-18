import os
import shutil
import fire
import cv2
import json
import numpy as np


# dataset parameters
json_file_name = "coco.json"
class_list = [1, 2, 3, 4, 6, 8]
class_dict = {"person": 0, "bike": 1, "car": 2, "motor": 3, "bus": 4, "truck": 5}
IMG_W = 640
IMG_H = 512
RESIZE_W = 640
RESIZE_H = 480
obj_size_thres = 0.01


def convert(coco_dataset, yolo_dataset):
    """
    convert the dataset in coco format to yolo_v5 format,
    and resize the image to 160*120, fitting our thermal configure
    :param coco_dataset:
    :type coco_dataset:
    :param yolo_dataset:
    :type yolo_dataset:
    :return:
    :rtype:
    """
    image_doc = os.path.join(yolo_dataset, "images")
    if not os.path.exists(image_doc):
        os.makedirs(image_doc)
    label_doc = os.path.join(yolo_dataset, "labels")
    if not os.path.exists(label_doc):
        os.makedirs(label_doc)

    coco_json_path = os.path.join(coco_dataset, json_file_name)
    f = open(coco_json_path,)
    data = json.load(f)
    data_annotations = data["annotations"]
    data_categories = data["categories"]
    data_images = data["images"]

    images_dict = {}
    for image in data_images:
        images_dict[image["id"]] = image["file_name"]

    categories_dict = {}
    for category in data_categories:
        categories_dict[category["id"]] = category["name"]

    image_num = 0
    check_set = set()
    for anno in data_annotations:
        image_id = anno["image_id"]
        image_name = images_dict[image_id]
        image_path = os.path.join(coco_dataset, image_name)

        # continue if the image don't exist in the dataset
        if not os.path.exists(image_path):
            continue

        category_id = anno["category_id"]
        if category_id not in class_list:   # the class is not considered
            continue
        bbox = anno["bbox"]
        x_min, y_min, w, h = bbox[0], bbox[1], bbox[2], bbox[3]

        # normalized the data
        w_norm = w / IMG_W
        h_norm = h / IMG_H
        x_cent_norm = (x_min + w/2) / IMG_W
        y_cent_norm = (y_min + h/2) / IMG_H
        # # filter out the small objects
        # if w_norm * h_norm < obj_size_thres:
        #     continue
        # get category id
        class_id = class_dict[categories_dict[category_id]]
        content = f"{class_id} {x_cent_norm} {y_cent_norm} {w_norm} {h_norm}"

        if image_id not in check_set:   # a new image
            # add to the check_set
            check_set.add(image_id)
            image_num += 1
            des_image_name = "%06d.jpg" % image_num
            des_image_path = os.path.join(image_doc, des_image_name)
            des_label_name = "%06d.txt" % image_num
            des_label_path = os.path.join(label_doc, des_label_name)

            # resize the image and write to the dest path
            image = cv2.imread(image_path, 1)   # 1 for image color, 0 for gray color
            image = cv2.resize(image, (RESIZE_W, RESIZE_H), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(des_image_path, image)

            # write the label file
            file = open(des_label_path, "w")
            file.write(content)
            file.close()

        elif image_id in check_set:     # an existing image
            # append content to label file
            des_label_name = "%06d.txt" % image_num
            des_label_path = os.path.join(label_doc, des_label_name)
            file = open(des_label_path, "a")
            file.write("\n")
            file.write(content)
            file.close()


if __name__ == "__main__":
    fire.Fire()
