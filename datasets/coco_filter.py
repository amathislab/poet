
import json
from tqdm import tqdm

path = 'data/COCO/'
mode = 'train'
keep_images_with_x_persons = False
nr_persons = 1
coco_filter = 'new'

coco_annotations_path = f"{path}annotations/person_keypoints_{mode}2017_old.json"
new_coco_annotations_path = f"{path}annotations/person_keypoints_{mode}2017_{coco_filter}.json"

with open(coco_annotations_path, "r") as f:
    data = json.load(f)

empty_images = []
empty_annotations = []
single_person_images = []
for image in tqdm(data["images"]):
    find = [item for item in data["annotations"] if item["image_id"] == image["id"]]
    # if no annotations
    if not any(find):
        empty_images.append(image["id"])
    else:
        count = 0
        for ann in find:
            # if keypoint annotations are all 0
            if not any(ann["keypoints"]):
                empty_annotations.append(ann["id"])
                count += 1
        if count == len(find):
            empty_images.append(image["id"])
        # if you want to keep only images with 1 person
        if (keep_images_with_x_persons
            and len(find) == nr_persons
            and any(find[0]["keypoints"]) ):
            single_person_images.append(image["id"])

data["annotations"][:] = [d for d in data["annotations"] if d.get("id") not in empty_annotations]
data["annotations"][:] = [d for d in data["annotations"] if d.get("image_id") not in empty_images]
data["images"][:] = [d for d in data["images"] if d.get("id") not in empty_images]

if keep_images_with_x_persons:
    data["annotations"][:] = [d for d in data["annotations"] if d.get("image_id") in single_person_images]
    data["images"][:] = [d for d in data["images"] if d.get("id") in single_person_images]

with open(new_coco_annotations_path, "w") as f:
    json.dump(data, f)
