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
import numpy as np


logging.basicConfig(filename="logfile.log", level=logging.INFO)

def focus_brave():
    sleep(0.1)
    subprocess.run(["xdotool", "search", "--onlyvisible", "--class", "Brave-browser", "windowactivate"])

# def focus_tinder(windowID: str):
#     logging.info("Focusing Tinder")
#     sleep(0.1)
#     subprocess.run(["xdotool", "windowactivate", windowID])

# def swipe_left(window_id: str):
#     logging.info("Swiping left")
#     subprocess.run(f'xdotool mousemove --window {window_id} 300 400 mousedown 1 && xdotool mousemove --window {window_id} 50 400 mouseup 1', shell=True)    # subprocess.run(["xdotool", "click", "1"])

# def swipe_right(window_id: str):
#     logging.info("Swiping right")
#     subprocess.run(["xdotool", "mousemove", "--window", window_id, "300", "400", "mousedown", "1", "mousemove", "--window", window_id, "550", "400", "mouseup", "1"])
#     # subprocess.run(["xdotool", "click", "1"])


parser = argparse.ArgumentParser()
parser.add_argument("--posDir", type=str, required=True)
parser.add_argument("--negDir", type=str, required=True)
# parser.add_argument("--windowID", type=str, required=True)
args = parser.parse_args()
posDir = args.posDir
negDir = args.negDir
# windowID = args.windowID


def main() -> None:
    attraction = AttractionClassifier()

    while True:
        scores = []
        labels = []
        prevFace = None
        sleepTime = random.randint(2, 4)
        logging.info(f"waiting {sleepTime} seconds before proceeding to hopefully prevent rate limiting")
        sleep(sleepTime)


        while True:
            focus_brave()
            subprocess.run(["gnome-screenshot", "-w", "-f", "temp_screenshot.png"], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            pwd = os.getcwd()
            imgPath = os.path.join(pwd, "temp_screenshot.png")
            ret = attraction.classify_image(imgPath, bCentralCrop=False)

            if ret is None:
                logging.info(colored("no face, no swipe right. GET OUTTA HERE!", "red"))
                focus_brave()
                swipe_left_if_needed(scores, labels)
                break

            result, face = ret
            log_res(result, scores)
            scores, labels = update_scores_and_labels(result, scores, labels)

            if face_not_changed(face, prevFace) or confident_enough_to_swipe_right(scores, labels):
                handle_swipe_right_or_left(scores, labels, imgPath, result)
                break

            prevFace = face

def swipe_left_if_needed(scores, labels):
    if scores and labels:
        if np.mean(scores) >= 0.8 and labels.count("pos") > labels.count("neg"):
            logging.info(colored("swiping right", "green"))
            swipe_right()
        else:
            swipe_left()
    else:
        swipe_left()

def log_res(result, scores):
    logging.info(f"Classification result: {result[0]['label']}")
    logging.info(f"Mean scores: {np.mean(scores)}")

def update_scores_and_labels(result, scores, labels):
    if result[0]["label"] == "pos":
        scores.append(result[0]["score"])
    labels.append(result[0]["label"])
    return scores, labels

def face_not_changed(face, prevFace):
    return np.array_equal(face, prevFace)

def confident_enough_to_swipe_right(scores, labels):
    return np.mean(scores) >= 0.8 and labels.count("pos") > labels.count("neg")

def handle_swipe_right_or_left(scores, labels, imgPath, result):
    if confident_enough_to_swipe_right(scores, labels):
        log_result_and_swipe("right", result[0])
        move_img_to_dir(imgPath, posDir)
    else:
        log_result_and_swipe("left", result[0])
        move_img_to_dir(imgPath, negDir)

def swipe_right():
    focus_brave()
    subprocess.run(["xdotool", "key", "Right"])

def swipe_left():
    focus_brave()
    subprocess.run(["xdotool", "key", "Left"])

def log_result_and_swipe(direction, result):
    if direction == "right":
        logging.info(colored("swiping right", "green"))
        if not scores.empty():
            if np.mean(scores) >= 0.9:
                logging.info(colored(f"result: {result}", "blue"))
            else:
                logging.info(colored(f"result: {result}", "green"))
    else:
        logging.info(colored("swiping left", "red"))
        logging.info(colored(f"result: {result}", "red"))

def move_img_to_dir(imgPath, dir):
    newPath = os.path.join(dir, f"{uuid4()}.png")
    while os.path.exists(newPath):
        newPath = os.path.join(dir, f"{uuid4()}.png")
    # os.rename(imgPath, newPath)

if __name__ == "__main__":
    main()