import subprocess
import random
import logging
from time import sleep
import os
from PIL import Image
from uuid import uuid4
import argparse
from termcolor import colored
import requests
from face_grab import FaceGrabber
from dotenv import load_dotenv
import numpy as np

load_dotenv()


logging.basicConfig(level=logging.INFO) #filename="logfile.log")

# constants
APIURL = "https://api-inference.huggingface.co/models/ongkn/attraction-classifier"
APIKEY = os.getenv("APIKEY") # an HF API key (user or org)
headers = {"Authorization": f"Bearer {APIKEY}"}

def focus_ffox():
    logging.info("Focusing the Fox")
    sleep(0.1)
    subprocess.run(["xdotool", "search", "--onlyvisible", "--class", "librewolf", "windowactivate"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--posDir", type=str, required=True)
    parser.add_argument("--negDir", type=str, required=True)
    args = parser.parse_args()
    posDir = args.posDir
    negDir = args.negDir
    faceGrabber = FaceGrabber()

    while True:
        sleepTime = random.randint(2, 4)
        logging.info(f"waiting {sleepTime} seconds before proceeding to hopefully prevent rate limiting")
        sleep(sleepTime)
        focus_ffox()
        logging.info("taking a screenshot")
        subprocess.run(["gnome-screenshot", "-w", "-f", "temp_screenshot.png"])
        pwd = os.getcwd()
        imgPath = os.path.join(pwd, "temp_screenshot.png")
        logging.info("grabbing face")
        image = Image.open(imgPath).convert("RGB")
        face = faceGrabber.grab_faces(np.array(image))
        if face is None:
            logging.info(colored("no face, no swipe right. GET OUTTA HERE!", "red"))
            focus_ffox()
            subprocess.run(["xdotool", "key", "Left"])
            continue
        face = Image.fromarray(face)
        face.save("face.png")
        face.show()
        with open("face.png", "rb") as f:
            face = f.read()
        logging.info(f'Classifying image at {imgPath}')
        result = requests.post(APIURL, headers=headers, data=face).json()
        if result[0]["label"] == "pos" and result[0]["score"] < 0.9:
            logging.info(colored(f'Classification result: {result[0]["label"]} with score {result[0]["score"]}', "green"))
        elif result[0]["label"] == "pos" and result[0]["score"] >= 0.9:
            logging.info(colored(f'Classification result: {result[0]["label"]} with score {result[0]["score"]}', "blue"))
        else:
            logging.info(colored(f'Classification result: {result[0]["label"]} with score {result[0]["score"]}', "red"))
        if result[0]["label"] == "pos" and result[0]["score"] >= 0.9:
            logging.info(colored("pos and confident enough, swiping right", "green"))
            subprocess.run("pw-play", "right.wav")
            focus_ffox()
            subprocess.run(["xdotool", "key", "Right"])
            newImgPath = os.path.join(posDir, f"{uuid4()}.png")
            while os.path.exists(newImgPath):
                newImgPath = os.path.join(posDir, f"{uuid4()}.png")
            os.rename(imgPath, newImgPath)
        else:
            focus_ffox()
            subprocess.run(["xdotool", "key", "Left"])
            logging.info("not pos or pos but not confident enough, swiping left")
            newImgPath = os.path.join(negDir, f"{uuid4()}.png")
            while os.path.exists(newImgPath):
                newImgPath = os.path.join(negDir, f"{uuid4()}.png")
            os.rename(imgPath, newImgPath)


if __name__ == "__main__":
    main()