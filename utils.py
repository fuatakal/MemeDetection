import cv2
from PIL import Image
import numpy as np
import json
import os

def get_configs(path):
  with open(path, 'r') as file:
      json_data = file.read()
      configs = json.loads(json_data)
  return configs

def get_display_image(path, config_path):
  configs = get_configs(config_path)
  img = Image.open(path)
  img_arr = np.array(img)
  imshow = cv2.resize(img_arr, dsize=(configs['img_size'], configs['img_size']), interpolation=cv2.INTER_CUBIC)
  return imshow

def get_inference_image(path, config_path):
  configs = get_configs(config_path)
  img = Image.open(path)
  img_arr = np.array(img)
  imgpred = cv2.resize(
        (img_arr / 255 - np.array([0.513, 0.485, 0.461]))
        / np.sqrt(np.array([0.107, 0.103, 0.108])),
        dsize=(configs['img_size'], configs['img_size']),
        interpolation=cv2.INTER_CUBIC,
    )
  return imgpred

def category_decode(prediction):
  prediction = round(prediction[0][0])
  return 'meme' if prediction else 'regular'

def iterative_scan_predict(img_paths, config_path, model):
  images = os.listdir(img_paths)
  preds = {}

  for each_image in images:
    exact_path = os.path.join(img_paths, each_image)
    impred = get_inference_image(exact_path, config_path)
    impred_resized = np.expand_dims(impred, axis=0)
    output = model.predict(impred_resized)
    prediction = category_decode(output)
    preds[exact_path] = prediction

  with open('./predictions/model_predictions.json', 'w') as json_file:
      json.dump(preds, json_file, indent=4)