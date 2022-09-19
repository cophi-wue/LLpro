import json
import os
from pathlib import Path
from code.utils.preprocess import test_file


def read(gold_dir: Path, ):
    for gold_file in gold_dir.iterdir():
        if ".pred" in gold_file.name: continue
        if test_file not in gold_file.name: continue
        print("--" * 10, gold_file.name)
        with open(str(gold_file)) as f:

            data = json.load(f)
        text = data["text"]
        for scene in data["scenes"]:
            print("*" * 10, scene["type"])
            print(text[scene["begin"]: scene["end"]])
            print()


if __name__ == '__main__':
    json_path = "/home/murathan/Desktop/scene-segmentation/json" if "home/" in os.getcwd() else "/cephyr/users/murathan/Alvis/scene-segmentation/json"
    #json_path = "33stss_20_30_predictions"
    json_dir = Path(json_path)
    read(json_dir)
