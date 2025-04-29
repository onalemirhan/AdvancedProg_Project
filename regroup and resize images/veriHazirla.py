import os
import shutil

original_path = r"D:\FMD_DATASET"
final_path = r"D:\final_dataset"

class_map = {
    "with_mask": "with_mask",
    "without_mask": "without_mask",
    "incorrect_mask": "incorrect_mask"
}

for original_class, target_class in class_map.items():
    src_root = os.path.join(original_path, original_class)
    dst_root = os.path.join(final_path, target_class)
    os.makedirs(dst_root, exist_ok=True)

    for sub in os.listdir(src_root):  
        sub_path = os.path.join(src_root, sub)
        if os.path.isdir(sub_path):
            for file in os.listdir(sub_path):
                src_file = os.path.join(sub_path, file)
                dst_file = os.path.join(dst_root, file)
                shutil.copy2(src_file, dst_file)