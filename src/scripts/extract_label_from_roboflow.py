import os
import shutil
import glob
import cv2
import numpy as np

def main():

    repo_dir = os.getcwd()
    input_dir = f"{repo_dir}/parameters/input"
    output_dir = f"{repo_dir}/parameters/output"
    dataset_dir = f"{output_dir}/dataset_name"
    dataset_data_dir = f"{dataset_dir}/data"
    dataset_label_dir = f"{dataset_dir}/label"

    if not os.listdir(input_dir):
        print("Input directory is empty.")
        return

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    os.makedirs(dataset_dir)
    os.makedirs(dataset_data_dir)
    os.makedirs(dataset_label_dir)

    # Get classes from _classes.csv
    classes = {"background": {"found": False}, "lane_lines": {"found": False}, "drivable_area": {"found": False}, "cones": {"found": False}}
    class_dir = f"{input_dir}/label/_classes.csv"
    with open(class_dir, "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if i == 0:
                continue
            class_idx, class_name = line.strip().split(",")
            class_name = "".join([c for c in class_name if c.isalpha()]).lower()
            class_idx = int(class_idx)
            if "back" in class_name:
                classes["background"]["found"] = True
                classes['background']["idx"] = class_idx
            elif "lane" in class_name or "line" in class_name:
                classes["lane_lines"]["found"] = True
                classes["lane_lines"]["idx"] = class_idx
            elif "drivable" in class_name or "area" in class_name:
                classes["drivable_area"]["found"] = True
                classes["drivable_area"]["idx"] = class_idx
            elif "cone" in class_name:
                classes["cones"]["found"] = True
                classes["cones"]["idx"] = class_idx
    
    # Ensure all classes are found
    if not all([cl["found"] for cl in classes.values()]):
        print("Missing classes in _classes.csv. Need background, lane_lines, drivable_area, and cones.")
        return
    else:
        for class_name, class_dict in classes.items():
            os.makedirs(f"{dataset_label_dir}/{class_name}")
            print(f"Found class \"{class_name}\" with idx {class_dict['idx']}.")

    # Extract labels for each class
    label_dirs = sorted(glob.glob(f"{input_dir}/label/*_mask.png"))
    idxs = [os.path.basename(label_dir).split("_")[0] for label_dir in label_dirs]
    for idx, label_dir in zip(idxs, label_dirs):
        roboflow_label = cv2.imread(label_dir, cv2.IMREAD_GRAYSCALE)
        for class_name, class_dict in classes.items():
            label = np.zeros_like(roboflow_label)
            label[roboflow_label == class_dict['idx']] = 255
            cv2.imwrite(f"{dataset_label_dir}/{class_name}/{idx}.jpg", label)
    
    # Extract data images
    data_dirs = sorted(glob.glob(f"{input_dir}/label/*.jpg"))
    idxs = [os.path.basename(data_dir).split("_")[0] for data_dir in data_dirs]
    for idx, data_dir in zip(idxs, data_dirs):
        data = cv2.imread(data_dir, cv2.IMREAD_COLOR)
        cv2.imwrite(f"{dataset_data_dir}/{idx}.jpg", data)

if __name__ == "__main__":
    main()