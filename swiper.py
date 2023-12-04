import subprocess
from infer import AttractionClassifier
import random
import logging
from time import sleep
import os
from PIL import Image
from uuid import uuid4
import argparse
from termcolor import colored


logging.basicConfig(filename="logfile.log", level=logging.INFO)

def focus_brave():
    logging.info("Focusing Brave")
    sleep(0.1)
    subprocess.run(["xdotool", "search", "--onlyvisible", "--class", "Brave-browser", "windowactivate"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--posDir", type=str, required=True)
    parser.add_argument("--negDir", type=str, required=True)
    args = parser.parse_args()
    posDir = args.posDir
    negDir = args.negDir

    attraction = AttractionClassifier()

    while True:
        sleep_time = random.randint(3, 8)
        logging.info(f"waiting {sleep_time} seconds before proceeding to hopefully prevent rate limiting")
        sleep(sleep_time)
        focus_brave()
        logging.info("taking a screenshot")
        subprocess.run(["gnome-screenshot", "-w", "-f", "temp_screenshot.png"])
        pwd = os.getcwd()
        imgPath = os.path.join(pwd, "temp_screenshot.png")
        logging.info(f'Classifying image at {imgPath}')
        result = attraction.classify_image(imgPath, bCentralCrop=True)
        if result is None:
            logging.info(colored("no face, no swipe right. GET OUTTA HERE!", "red"))
            subprocess.run(["xdotool", "key", "Left"])
            continue
        logging.info(f'Classification result: {result[0]["label"]}')
        print(result[0])
        if result[0]["label"] == "pos" and result[0]["score"] >= 0.9:
            logging.info(colored("pos and confident enough, swiping right", "green"))
            subprocess.run("pw-play", "right.wav")
            subprocess.run(["xdotool", "key", "Right"])
            newImgPath = os.path.join(posDir, f"{uuid4()}.png")
            while os.path.exists(newImgPath):
                newImgPath = os.path.join(posDir, f"{uuid4()}.png")
            os.rename(imgPath, newImgPath)
        else:
            subprocess.run(["xdotool", "key", "Left"])
            logging.info("not pos or pos but not confident enough, swiping left")
            newImgPath = os.path.join(negDir, f"{uuid4()}.png")
            while os.path.exists(newImgPath):
                newImgPath = os.path.join(negDir, f"{uuid4()}.png")
            os.rename(imgPath, newImgPath)


if __name__ == "__main__":
    main()