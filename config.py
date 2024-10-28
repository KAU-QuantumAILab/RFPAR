# Hyperparameters
config = {
    #  Fix config
    "batch_size": 50,  # batch_size refers to the size of the mini-batch used during RL training. Here, it is fixed at 50.
    "RL_learning_rate": 0.0001,  # RL_learning_rate refers to the learning rate of the RL in the Remember process. Here, it is fixed at 0.0001.
    "RGB": 3,  # RGB refers to the channels of the image. Here, it is fixed to 3 dimensions.
    "shape_unity" : True,  # Shape_unity checks whether the width and height of the image are unified across the entire dataset.
    "subtract": True,   
    "std": "learn",  #  We use learnable standard deviation

    #  Variate config
    "classifier" : "ResNext",  # This refers to the victim model being deceived. In this code, experiments are conducted on resnext50_32x4d and yolo.
    "attack_pixel" : 0.01,  # This is the hyperparameter α that selects the number of pixels for the image size. Different values are set for each task: 0.01 for image classification and 0.05 for object detection.
    "dataset": "ImageNet",  # ImageNet, COCO
    "bound" : 100,  # the maximum number of iteration for Remember and Forget process. We use 100.
    "patient" : 3,  # Duration of condition, denoted as T. We use 3 in image classification.
    "limit" : 5e-2  # Bound threshold, denoted as η in the paper. We use 0.05

}






if config["dataset"] == "ImageNet":
    config["img_size_x"] = 224
    config["img_size_y"] = 224
    config["RGB"] = 3
    config["patient"] = 3
    config["bound"] = 100
    config["limit"] = 5e-2

elif config["dataset"] == "COCO":
    config["img_size_x"] = 224
    config["img_size_y"] = 224
    config["RGB"] = 3
    config["patient"] = 20
    config["bound"] = 100
    config["limit"] = 5e-2
