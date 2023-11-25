from tensorflow.keras.models import load_model
from tensorflow_addons.optimizers import AdamW
from patches import Patches, PatchEncoder
from positional_encoding import PositionalEmbedding
from utils import get_inference_image, category_decode, iterative_scan_predict
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# const vars
MODEL_PATH = './model/ConcepCap 0.67 TXT 1.00 ViTa.h5'
CONFIG_PATH = './configs/config.json'
IMG_PATH = './dataset'

if __name__ == '__main__':

  # load the model
  model = load_model(MODEL_PATH,
                    custom_objects = {
                          "Patches": Patches,
                          "PatchEncoder": PatchEncoder,
                          "PositionalEmbedding": PositionalEmbedding,
                          "adamw": AdamW
                    })

  # inference
  iterative_scan_predict(IMG_PATH, CONFIG_PATH, model)