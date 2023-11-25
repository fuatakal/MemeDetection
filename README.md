# Meme Detection Model

* This code is adapted from the "MemeTector: Enforcing deep focus for meme detection" paper's github repo.

      ** url: https://arxiv.org/abs/2205.13268
  
      ** github: https://github.com/mever-team/memetector

* First, you need to install libraries. To do this run the following command:

    `python -m pip install -r requirements.txt`

* Put the test images in the `dataset` folder.

* Download the pretrained model file from `https://github.com/mever-team/memetector/blob/main/ckpt/ConcepCap%200.67%20TXT%201.00%20ViTa.h5` and put it into the `model` folder.

* Run the inference.py using `python inference.py` command.

* The results will be stored in the `predictions` folder.

