import os
from glob import glob



json_files = glob(os.path.join("*.json"))
print(json_files)
for json_file in json_files:
    os.system(f"labelme_json_to_dataset {json_file} -o {''.join(json_file.split('.')[:-1])}")
