import cv2
import numpy as np

img = cv2.imread('train/pepega.png')

def predict(img: np.ndarray, model_path: str) -> np.ndarray:
    mean, std = np.load(model_path)
    img_pred = (img-mean)/std
    return img_pred

def train(img: np.ndarray, save_model_path: str) -> None:
    mean  = np.mean(img)
    std   = np.std(img)

    np.save(save_model_path, np.array([mean,std]))