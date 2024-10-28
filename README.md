# Amnesia as a Catalyst for Enhancing Black Box Pixel Attacks in Image Classification and Object Detection

This repository contains the code of Remember and Forget Pixel Attack using RL ([RFPAR](https://nips.cc/virtual/2024/poster/95435)) for image classification tasks using ResNext and object detection tasks using YOLO.

If you use the code or find this project helpful, please consider citing our paper.
```bash
@article{Song2024RFPAR,
  title={Amnesia as a Catalyst for Enhancing Black Box Pixel Attacks in Image Classification and Object Detection},
  author={Song, S., Ko, D., and Jung, J., H.},
  booktitle={Advances in Neural Information Processing Systems, NeurIPS},
  year={2024}
}
```

## Project Structure

- `Adversarial_RL_simple.py`: Contains the implementation of the adversarial reinforcement learning algorithm.
- `Argoverse_Attack_Video.pptx`: Presentation file with attack results on the Argoverse dataset. Run the slide show to view the data sequence as a video.
- `COCO/`: Directory containing COCO dataset images.
- `ImageNet/`: Directory containing ImageNet dataset images.
- `config.py`: Configuration file for setting up various parameters.
- `Environment.py`: Defines the environment for reinforcement learning.
- `main_cls.py`: Main script to run the attack for image classification.
- `main_od.py`: Main script to run the attack for object detection.
- `requirements.txt`: List of Python packages required to run the project.
- `results/`: Directory to store results of the adversarial attacks.
- `utils.py`: Utility functions used throughout the project.
- `yolov8n.pt`: Pre-trained YOLO model weights.




## Usage

### 0. Unzip Datasets(ImageNet and COCO samples)

### 1. Install the requirements.txt
To install the required Python packages, run:

```bash
pip install -r requirements.txt
```

### 2. Running the attack
To run the adversarial attack, execute the main script:

In classification task

```bash
python main_cls.py
```

In object detection task

```bash
python main_od.py
```


## Results

The experimental results are stored in the `results` directory according to the task. Results for classification tasks are stored in the `ImageNet` folder, while results for object detection tasks are stored in the `COCO` folder.

- `adv_images`: Stores images successfully attacked.
- `adv_result`: Stores the altered results of object detection.
- `delta_images`: Stores the perturbations that create adversarial images.
- `original_result`: Stores the results of object detection on clean images.
