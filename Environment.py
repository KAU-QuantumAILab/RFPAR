import torch
import torch.nn as nn
import numpy as np
from utils import normalize
import torch.nn.functional as F
from PIL import Image



class Env:
    def __init__(self, classification_model,  config):
        super().__init__()
        self.classification_model = classification_model
        self.pixel = config["attack_pixel"]
        self.config = config
        self.ori_prob = []
        self.ori_cls = []
        self.ori_box_num = []
        self.s= None
        self.init = False



    def make_transformed_images(self, original_images, actions):
        # Check if the images are grayscale or RGB
        RGB = original_images.shape[1]
        x_bound = original_images.shape[2]
        y_bound = original_images.shape[3]
        
        # Apply sigmoid to actions
        actions = torch.sigmoid(actions)
        arr = []

        # Scale the actions to the image dimensions
        x = (actions[:, 0]*x_bound - 1).long()  # scale to 0,1,...,26
        y = (actions[:, 1]*y_bound - 1).long()



        if RGB == 1:    # Grayscale
            brightness = ((actions[:, 2] * 255).int()).float() / 255
            for i in range(self.config["batch_size"]):
                changed_image = (
                    original_images[i].squeeze().squeeze().detach().cpu().numpy()
                )
                changed_image[x[i], y[i]] = brightness[i]
                arr.append(changed_image)
        elif RGB == 3:   # RGB
            if self.config["classifier"] == "yolo" or self.config["classifier"] == "ddq":       
                r= (actions[:,2]>0.5).float()*255
                g= (actions[:,3]>0.5).float()*255
                b= (actions[:,4]>0.5).float()*255
            else:                                                                               
                r= (actions[:,2]>0.5).float()
                g= (actions[:,3]>0.5).float()
                b= (actions[:,4]>0.5).float()


            batch = original_images.shape[0]
            

            for i in range(batch):
                changed_image =original_images[i].clone()

                if self.pixel == 1:         #   The number of attacked pixels = 1
                    changed_image[0, x[i], y[i]] = r[i]
                    changed_image[1, x[i], y[i]] = g[i]
                    changed_image[2, x[i], y[i]] = b[i]
                else:
                    idx = torch.tensor([j for j in range(self.pixel)])*batch + i
                    if self.config["classifier"] == "yolo" or self.config["classifier"] == "ddq":
                        changed_image[0, x[idx], y[idx]] = r[idx].type(torch.uint8)
                        changed_image[1, x[idx], y[idx]] = g[idx].type(torch.uint8)
                        changed_image[2, x[idx], y[idx]] = b[idx].type(torch.uint8)
                    else:
                        changed_image[0, x[idx], y[idx]] = r[idx].type(torch.float32)
                        changed_image[1, x[idx], y[idx]] = g[idx].type(torch.float32)
                        changed_image[2, x[idx], y[idx]] = b[idx].type(torch.float32)
                arr.append(changed_image.unsqueeze(0))

                
        
        changed_images = torch.cat(arr,0)
        return changed_images

    def step(self, original_images, actions, labels, ori_prob=None):
        # Transform the images based on the actions
        # This attack is used in the paper
        # We use this attack in image classification

        changed_images = self.make_transformed_images(original_images, actions)    
        labels = labels.to(self.config["device"])

        with torch.no_grad():
            if self.init == True:
                # Check the confidence score in the initial images for reward generation
                original_outputs = torch.softmax(self.classification_model(
                    normalize(original_images.to(self.config["device"]),self.config)), dim=1
                )
                self.ori_prob = original_outputs[np.arange(labels.shape[0]),labels]
                ori_prob = self.ori_prob.clone()
            changed_outputs = torch.softmax(self.classification_model(normalize(changed_images.to(self.config["device"]),self.config)), dim=1)
            changed_preds, changed_preds_idx = torch.max(changed_outputs, dim=1)
            

            # Check if the predictions changed

            change_list = (labels != changed_preds_idx).to('cuda')

            
            # Calculate

            rewards = ori_prob-changed_outputs[np.arange(labels.shape[0]),labels]


            
        return rewards, change_list.to('cuda'), changed_images.to('cuda'), changed_preds_idx


    def yolo_step(self, original_images, actions, bt,labels=None, probs=None):
        # Use fixed the number of objects to be removed and attack images with the same shape to the object detector
        # This attack is not used in the paper

        changed_images = self.make_transformed_images(original_images.to(self.config["device"]), actions)    
        with torch.no_grad():
            if self.init == True:
                # Check the confidence score and number of detected objects in the initial images for reward generation
                labels = []
                probs = []
                for img in original_images:
                    result = self.classification_model(img.detach().cpu().numpy().transpose(1,2,0),imgsz=640,conf=self.config["yolo_conf"])
                    probs.append(result[0].boxes.conf)
                    labels.append(result[0].boxes.cls)

                self.ori_prob = self.ori_prob + probs
                self.ori_cls = self.ori_cls + labels

            changed_images = changed_images.detach().cpu().numpy().transpose(0,2,3,1)
            prob_list = []
            cls_list = []
            temp_list =[]

            for n,i in enumerate(changed_images):
                temp_list.append(i)
                results = self.classification_model(i,conf=self.config["yolo_conf"])
                prob_list.append(results[0].boxes.conf)
                cls_list.append(results[0].boxes.cls)

            rewards = []
            dif_list = []


            for i in range(len(cls_list)):
                size = max(torch.bincount(self.eval_cls[self.config["batch_size"]*bt+i].long()).shape[0], torch.bincount(cls_list[i].long()).shape[0])
                temp = torch.bincount(self.eval_cls[self.config["batch_size"]*bt+i].long(), minlength=size) - torch.bincount(cls_list[i].long(), minlength=size)
                people_count = temp[0:1].sum()
                vehicle_count = temp[1:].sum()
                
                dif = people_count+vehicle_count
                if dif == torch.bincount(self.eval_cls[self.config["batch_size"]*bt+i].long(), minlength=size).sum():
                    dif = self.config["attack_level"]
                if dif>=self.config["attack_level"]:
                    print(torch.bincount(self.eval_cls[self.config["batch_size"]*bt+i].long(), minlength=size),torch.bincount(cls_list[i].long(), minlength=size))
                dif_list.append(dif)

                probs[i] = torch.nn.functional.pad(probs[i], (0,size - len(probs[i])), mode='constant', value=0)
                prob_list[i] = torch.nn.functional.pad(prob_list[i], (0,size - len(prob_list[i])), mode='constant', value=0)
                reward = (probs[i] - prob_list[i]).sum()
                rewards.append(reward)


        return torch.tensor(rewards).to(self.config["device"]), torch.tensor(dif_list), changed_images


    def yolo_step_disunity(self, original_images, actions, bt, hws,labels=None, probs=None):
        # Use fixed the number of objects to be removed and attack images with the different shape to the object detector
        # This attack is not used in the paper

        changed_images = self.make_transformed_images(original_images.to(self.config["device"]), actions)    
        with torch.no_grad():
            if self.init == True:
                # Check the confidence score and number of detected objects in the initial images for reward generation
                labels = []
                probs = []
                # each
                for n,img in enumerate(original_images):
                    result = self.classification_model(img.detach().cpu().numpy().transpose(1,2,0)[:hws[n,0],:hws[n,1]],imgsz=640,conf=self.config["yolo_conf"])
                    probs.append(result[0].boxes.conf)
                    labels.append(result[0].boxes.cls)


                self.ori_prob = self.ori_prob + probs
                self.ori_cls = self.ori_cls + labels


            changed_images = changed_images.detach().cpu().numpy().transpose(0,2,3,1)
            prob_list = []
            cls_list = []
            temp_list =[]

            # each type
            for n,i in enumerate(changed_images):
                temp_list.append(i[:hws[n,0],:hws[n,1]])
                results = self.classification_model(i[:hws[n,0],:hws[n,1]],imgsz=640,conf=self.config["yolo_conf"])
                prob_list.append(results[0].boxes.conf)
                cls_list.append(results[0].boxes.cls)

            rewards = []
            dif_list = []

            for i in range(len(cls_list)):

                size = max(torch.bincount(self.eval_cls[self.config["batch_size"]*bt+i].long()).shape[0], torch.bincount(cls_list[i].long()).shape[0])
                temp = torch.bincount(self.eval_cls[self.config["batch_size"]*bt+i].long(), minlength=size) - torch.bincount(cls_list[i].long(), minlength=size)
                people_count = temp[0:1].sum()
                vehicle_count = temp[1:].sum()
                
                dif = people_count+vehicle_count
                if dif == torch.bincount(self.eval_cls[self.config["batch_size"]*bt+i].long(), minlength=size).sum():
                    dif = self.config["attack_level"]
                if dif>=self.config["attack_level"]:
                    print(torch.bincount(self.eval_cls[self.config["batch_size"]*bt+i].long(), minlength=size),torch.bincount(cls_list[i].long(), minlength=size))
                    
                dif_list.append(dif)

                probs[i] = torch.nn.functional.pad(probs[i], (0,size - len(probs[i])), mode='constant', value=0)
                prob_list[i] = torch.nn.functional.pad(prob_list[i], (0,size - len(prob_list[i])), mode='constant', value=0)
                reward = (probs[i] - prob_list[i]).sum()
                rewards.append(reward)



        return torch.tensor(rewards).to(self.config["device"]), torch.tensor(dif_list), changed_images


    def yolo_step_not_sub(self, original_images, actions, bt,labels=None, probs=None):
        # Attack images with the same shape without determining the number of objects to remove in YOLO
        # This attack is used in the paper
        changed_images = self.make_transformed_images(original_images.to(self.config["device"]), actions)    
        

        with torch.no_grad():
            if self.init == True:
                # Check the confidence score and number of detected objects in the initial images for reward generation
                labels = []
                probs = []
                for img in original_images:
                    result = self.classification_model(img.detach().cpu().numpy().transpose(1,2,0),imgsz=640,conf=self.config["yolo_conf"])
                    probs.append(result[0].boxes.conf)
                    labels.append(result[0].boxes.cls)

                self.ori_prob = self.ori_prob + probs
                self.ori_cls = self.ori_cls + labels

            changed_images = changed_images.detach().cpu().numpy().transpose(0,2,3,1)
            prob_list = []
            cls_list = []
            temp_list =[]

            for n,i in enumerate(changed_images):
                temp_list.append(i)
                results = self.classification_model(i,conf=self.config["yolo_conf"])
                prob_list.append(results[0].boxes.conf)
                cls_list.append(results[0].boxes.cls)


            rewards = []
            dif_list = []


            for i in range(len(cls_list)):
                # Check the maximum label
                size = max(torch.bincount(labels[i].long()).shape[0], torch.bincount(cls_list[i].long()).shape[0])
                temp = torch.bincount(labels[i].long(), minlength=size) - torch.bincount(cls_list[i].long(), minlength=size)
                dif = temp.sum() 
                dif_list.append(dif)
                # Check labels of removed boxes and generate rewards
                if (temp!=0).any():
                   
                    indices = list(filter(lambda x: temp[x] > 0, range(len(temp))))
                    values = list(filter(lambda x: x > 0, temp))

                    for indice, value in zip(indices, values):

                        add_cls = torch.LongTensor([indice for _ in range(value)]).to(self.config["device"])
                        add_prob = torch.zeros(value).float().to(self.config["device"])
                        cls_list[i] = torch.cat((cls_list[i].long(),add_cls), dim=0)
                        prob_list[i] = torch.cat((prob_list[i],add_prob),dim=0)

                    
                    indices = list(filter(lambda x: temp[x] < 0, range(len(temp))))
                    values = list(filter(lambda x: x < 0, temp))
                    for indice, value in zip(indices, values):

                        add_cls = torch.LongTensor([indice for _ in range(abs(value))]).to(self.config["device"])
                        add_prob = torch.zeros(abs(value)).float().to(self.config["device"])
                        labels[i] = torch.cat((labels[i].long(),add_cls),dim=0)
                        probs[i] = torch.cat((probs[i],add_prob),dim=0) 
              
                reward = (probs[i][labels[i].sort()[1]].to('cpu')-prob_list[i][cls_list[i].sort()[1]].to('cpu')).sum()+dif
                rewards.append(reward)




        return torch.tensor(rewards).to(self.config["device"]), torch.tensor(dif_list), changed_images


    def yolo_step_disunity_not_sub(self, original_images, actions, bt, hws,labels=None, probs=None):
        # Attack images with different shapes without determining the number of objects to remove in YOLO
        # This attack is used in the paper
        changed_images = self.make_transformed_images(original_images.to(self.config["device"]), actions)    
        with torch.no_grad():
            if self.init == True:
                # Check the confidence score and number of detected objects in the initial images for reward generation
                labels = []
                probs = []
                
                for n,img in enumerate(original_images):
                    result = self.classification_model(img.detach().cpu().numpy().transpose(1,2,0)[:hws[n,0],:hws[n,1]],imgsz=640,conf=self.config["yolo_conf"],verbose=False)
                    probs.append(result[0].boxes.conf)
                    labels.append(result[0].boxes.cls)

                self.ori_prob = self.ori_prob + probs
                self.ori_cls = self.ori_cls + labels

            changed_images = changed_images.detach().cpu().numpy().transpose(0,2,3,1)
            prob_list = []
            cls_list = []
            temp_list =[]

            for n,i in enumerate(changed_images):
                
                temp_list.append(i[:hws[n,0],:hws[n,1]])
                results = self.classification_model(i[:hws[n,0],:hws[n,1]],conf=self.config["yolo_conf"],verbose=False)
                prob_list.append(results[0].boxes.conf)
                cls_list.append(results[0].boxes.cls)


            rewards = []
            dif_list = []


            for i in range(len(cls_list)):
                # Check the maximum label
                size = max(torch.bincount(labels[i].long()).shape[0], torch.bincount(cls_list[i].long()).shape[0])
                temp = torch.bincount(labels[i].long(), minlength=size) - torch.bincount(cls_list[i].long(), minlength=size)
                dif = temp.sum()   
                dif_list.append(dif)
                # Check labels of removed boxes and generate rewards
                if (temp!=0).any():
                   
                    indices = list(filter(lambda x: temp[x] > 0, range(len(temp))))
                    values = list(filter(lambda x: x > 0, temp))
                    for indice, value in zip(indices, values):

                        add_cls = torch.LongTensor([indice for _ in range(value)]).to(self.config["device"])
                        add_prob = torch.zeros(value).float().to(self.config["device"])
                        cls_list[i] = torch.cat((cls_list[i].long(),add_cls), dim=0)
                        prob_list[i] = torch.cat((prob_list[i],add_prob),dim=0)

                    indices = list(filter(lambda x: temp[x] < 0, range(len(temp))))
                    values = list(filter(lambda x: x < 0, temp))
                    for indice, value in zip(indices, values):

                        add_cls = torch.LongTensor([indice for _ in range(abs(value))]).to(self.config["device"])
                        add_prob = torch.zeros(abs(value)).float().to(self.config["device"])
                        labels[i] = torch.cat((labels[i].long(),add_cls),dim=0)
                        probs[i] = torch.cat((probs[i],add_prob),dim=0) 
              
                reward = (probs[i][labels[i].sort()[1]].to('cpu')-prob_list[i][cls_list[i].sort()[1]].to('cpu')).sum()+dif
                rewards.append(reward)

        return torch.tensor(rewards).to(self.config["device"]), torch.tensor(dif_list), changed_images


    def ddq_step_disunity_not_sub_batch(self, original_images, actions, bt, hws,labels=None, probs=None):
        # Attack images with different shapes on the DDQ detector without determining the number of objects to remove
        # This attack is used in the paper
        changed_images = self.make_transformed_images(original_images.to(self.config["device"]), actions)    
        
        with torch.no_grad():
            if self.init == True:
                # Check the confidence score and number of detected objects in the initial images for reward generation
                imgs = []
                labels = []
                probs = []
                for n,img in enumerate(original_images):
                    imgs.append(img.detach().cpu().numpy().transpose(1,2,0)[:hws[n,0],:hws[n,1]])

                results = self.classification_model(imgs)
                for n in range(len(original_images)):
                    filtered_predictions =  [(index, pred) for index, pred in enumerate(results['predictions'][n]['scores']) if pred > self.config["yolo_conf"]]
                    if filtered_predictions:
                        indices, predictions = zip(*filtered_predictions)
                    else:
                        indices, predictions = [], []
                    probs.append(torch.tensor([predictions[_] for _ in indices]).to(self.config['device']))
                    labels.append(torch.tensor([results['predictions'][n]['labels'][_] for _ in indices]).to(self.config['device']))

                self.ori_prob = self.ori_prob + probs
                self.ori_cls = self.ori_cls + labels

            changed_images = changed_images.detach().cpu().numpy().transpose(0,2,3,1)
            prob_list = []
            cls_list = []
            imgs = []


            for n,i in enumerate(changed_images):
                imgs.append(i[:hws[n,0],:hws[n,1]])

            results = self.classification_model(imgs)
            for n in range(len(original_images)):
                filtered_predictions =  [(index, pred) for index, pred in enumerate(results['predictions'][n]['scores']) if pred > self.config["yolo_conf"]]
                if filtered_predictions:
                    indices, predictions = zip(*filtered_predictions)
                else:
                    indices, predictions = [], []
                prob_list.append(torch.tensor([predictions[_] for _ in indices]).to(self.config['device']))
                cls_list.append(torch.tensor([results['predictions'][n]['labels'][_] for _ in indices]).to(self.config['device']))


            rewards = []
            dif_list = []

            for i in range(len(cls_list)):
                # Check the maximum label
                size = max(torch.bincount(labels[i].long()).shape[0], torch.bincount(cls_list[i].long()).shape[0])
                temp = torch.bincount(labels[i].long(), minlength=size) - torch.bincount(cls_list[i].long(), minlength=size)
                dif = temp.sum()    
                dif_list.append(dif)
                # Check labels of removed boxes and generate rewards
                if (temp!=0).any():
                   
                    indices = list(filter(lambda x: temp[x] > 0, range(len(temp))))
                    values = list(filter(lambda x: x > 0, temp))

                    for indice, value in zip(indices, values):

                        add_cls = torch.LongTensor([indice for _ in range(value)]).to(self.config["device"])
                        add_prob = torch.zeros(value).float().to(self.config["device"])
                        cls_list[i] = torch.cat((cls_list[i].long(),add_cls), dim=0)
                        prob_list[i] = torch.cat((prob_list[i],add_prob),dim=0)
    
                    indices = list(filter(lambda x: temp[x] < 0, range(len(temp))))
                    values = list(filter(lambda x: x < 0, temp))
                    for indice, value in zip(indices, values):

                        add_cls = torch.LongTensor([indice for _ in range(abs(value))]).to(self.config["device"])
                        add_prob = torch.zeros(abs(value)).float().to(self.config["device"])
                        labels[i] = torch.cat((labels[i].long(),add_cls),dim=0)
                        probs[i] = torch.cat((probs[i],add_prob),dim=0) 
              
                reward = (probs[i][labels[i].sort()[1]].to('cpu')-prob_list[i][cls_list[i].sort()[1]].to('cpu')).sum()+dif
                rewards.append(reward)



        return torch.tensor(rewards).to(self.config["device"]), torch.tensor(dif_list), changed_images