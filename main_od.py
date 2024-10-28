from ultralytics import YOLO
from PIL import  Image, ImageDraw, ImageFont
import glob
import numpy as np
from config import config
import torchvision.transforms as transforms
import Environment
import Adversarial_RL_simple as Adversarial_RL_simple
import torch
from utils import seed_all, sample_action, early_stopping,MyBaseDataset, L0_norm,L2_norm
import math
import copy
import os
import math



def attack(model, train_data, config ,hw_array):


    env = Environment.Env(model,config=config)

    #  Check if image shapes are unified
    h_max, w_max = hw_array.max(axis=0)[0], hw_array.max(axis=0)[1]
    if ((hw_array[:,0] != h_max).sum()!=0 or (hw_array[:,1] != w_max).sum()!=0 ) == True:
        config["shape_unity"] = False
        for _ in range(len(train_data)):
            train_data[_] = np.pad(train_data[_],((0,h_max-hw_array[_,0]),(0,w_max-hw_array[_,1]),(0,0)),'constant',constant_values=0)
        env.hw_array = torch.tensor(hw_array).long()
    else:
        config["shape_unity"] = True
    print(f'shape_unity : {config["shape_unity"]}')


    
    #  Object detection results for clean images
    i = 1
    for n,_ in enumerate(train_data):
        if config["shape_unity"] == True:
            results = model(_,conf=config["yolo_conf"],verbose=False)
        elif config["shape_unity"] == False:
            results = model(_[:hw_array[n,0],:hw_array[n,1]],conf=config["yolo_conf"],verbose=False)
        env.ori_prob.append(results[0].boxes.conf)
        env.ori_cls.append(results[0].boxes.cls)
        env.ori_box_num.append(results[0].boxes.shape[0])

        im_array = results[0].plot() 
        im = Image.fromarray(im_array)
        im.save(f'{result_path}ori_'+'{0:04}'.format(i)+'.jpg') 
        i+=1
    
    env.ori_box_num = torch.tensor(env.ori_box_num).long()
    print("The Average number of Detected Object :",env.ori_box_num.float().mean().item())

    

    # Setting
    # Image part of Memory  
    update_images = torch.tensor(np.array(train_data)).clone().to(config["device"])
    metric_images = torch.tensor(np.array(train_data)).clone()
    trick_element = torch.zeros(len(train_data))
    yolo_list = torch.tensor(np.array(train_data)).clone()
    cls_list = []    

    torchvision_transform = transforms.Compose([transforms.Resize((256, 256)), transforms.CenterCrop(224)])
    train_data = MyBaseDataset(train_data, trick_element, transform=True)

    it=0
    total = 0
    iteration =  torch.zeros(update_images.shape[0]).to(config["device"])
    change_idx = []
    L0 = []
    L2 = []
    box_count = torch.zeros(update_images.shape[0]).to(config["device"])

    
    


    for p in range(config["bound"]):
        # This is the Forget Process. We repeat this process until T.
        # Load agent & Forget parameters to initialize
        agent = Adversarial_RL_simple.REINFORCE(config).to(config["device"])

        
        #  Setting
        flag = False
        stop_count = 0
        prev_change_list = torch.zeros(update_images.shape[0]).to(config["device"])
        update_rewards = torch.zeros(update_images.shape[0]).to(config["device"])
        temp = torch.zeros(update_images.shape[0]).to(config["device"])
        

        while 1:
            
            bts = math.ceil(train_data.x_data.shape[0]/config["batch_size"])

            train_x = []
            change_train_x = []
            it += 1
            total_rewards_list = []
            total_change_list = torch.tensor([]).to(config["device"])

            for bt in range(bts):


                #  Batch data
                if bt != (bts-1):
                    s = train_data.x_data[bt*config["batch_size"]:bt*config["batch_size"]+config["batch_size"]]
                    if env.init == False:
                        labels = env.ori_cls[bt*config["batch_size"]:bt*config["batch_size"]+config["batch_size"]]
                        probs = env.ori_prob[bt*config["batch_size"]:bt*config["batch_size"]+config["batch_size"]]
                    if config["shape_unity"] == False:
                        hws = env.hw_array[bt*config["batch_size"]:bt*config["batch_size"]+config["batch_size"]]
                else: 
                    s = train_data.x_data[bt*config["batch_size"]:]
                    if env.init == False:
                        probs = env.ori_prob[bt*config["batch_size"]:]
                        labels = env.ori_cls[bt*config["batch_size"]:]
                        
                    if config["shape_unity"] == False:
                        hws = env.hw_array[bt*config["batch_size"]:]

                #  Sample action
                s = s.permute(0,3,1,2)
                action_means, action_stds = agent(torchvision_transform(s.to(config["device"]))/255)
                action_stds = torch.clamp(action_stds, 0.1, 10)  # std가 0이 되는 것을 방지
                actions, actions_logprob = sample_action(action_means, action_stds, config)

                if config["attack_pixel"] != 1:
                    actions = actions.view(-1,5)
                    actions_logprob = actions_logprob.sum(axis=0)

                #  Generate reward
                if config["shape_unity"] == True:
                    rewards, dif_list, changed_images = env.yolo_step_not_sub(s, actions, bt,labels, probs)
                elif config["shape_unity"] == False:
                    rewards, dif_list, changed_images = env.yolo_step_disunity_not_sub(s, actions, bt, hws, labels, probs)
                
                change_list = dif_list
                
    
                s = s.permute(0,2,3,1)

                
                #  Setting for updating Memory
                for _ in range(len(changed_images)):
                    change_train_x.append(changed_images[_])
                total_change_list = torch.cat((total_change_list,change_list.to(config["device"])),dim=0).long()
                total_rewards_list += rewards.tolist()

                
                #  Training RL
                agent.r = rewards 
                agent.prob = actions_logprob
                agent.train_net()
 


            # Memory update

            standard = update_rewards.mean()
            total_rewards_list = torch.tensor(total_rewards_list).to(config["device"])

            update_rewards = torch.cat((update_rewards.unsqueeze(dim=0), total_rewards_list.unsqueeze(dim=0)),dim=0)
            update_rewards_indices = torch.max(update_rewards,axis=0).indices
            update_rewards = update_rewards[update_rewards_indices,np.arange(total_rewards_list.shape[0])]
            
            update_images = torch.cat((update_images.unsqueeze(dim=0),torch.tensor(np.array(change_train_x)).to(config["device"]).unsqueeze(dim=0)),dim=0)
            update_images = update_images[update_rewards_indices,np.arange(total_rewards_list.shape[0]),:,:]

                
            temp = torch.max(torch.cat((prev_change_list.unsqueeze(dim=0) ,total_change_list.unsqueeze(dim=0)),dim=0),axis=0).values
            delta_box =  temp - prev_change_list
            prev_change_list = temp.clone()
            
            update_rewards_sum = update_rewards.mean() 
            

            if delta_box.sum()>0:
                change_idx = list(filter(lambda x: delta_box[x] > 0, range(len(delta_box))))
                for _ in change_idx:
                    yolo_list[_] = torch.tensor(change_train_x[_]).clone()
                    iteration[_] = it

            #  1 iter count
            box_count += delta_box



            #  Convergence condition
            stop_count, flag = early_stopping(((update_rewards_sum-standard)/standard) + delta_box.sum(), stop_count, limit=config["limit"],patient=config["patient"])

            if flag == True:
                env.init = True
                train_x=update_images
                train_data = MyBaseDataset(train_x, cls_list)
                env.ori_cls = []
                env.ori_prob = []
                it += 1
                break
            elif flag == False:
                env.init = False
            

        #  total count
        total = box_count.mean()
   

        print(f'Forget:{p}, eliminated box: {total}')
        

        
        if env.ori_box_num.float().mean()-total == 0:
            print("all Images are deceived")
            break

    #  Metric
    total = len(prev_change_list)
    
    if config["shape_unity"] == True:
        L0 = L0_norm(metric_images,yolo_list)
        L2 = L2_norm(metric_images.float(),yolo_list)

    elif config["shape_unity"] == False:
        for _ in range(total):
            L0.extend(L0_norm(metric_images[_,:hw_array[n,0],:hw_array[n,1]],yolo_list[_,:hw_array[n,0],:hw_array[n,1]]))
            L2.extend(L2_norm(metric_images.float()[_,:hw_array[n,0],:hw_array[n,1]],yolo_list[_,:hw_array[n,0],:hw_array[n,1]]))


    
    #  Save results
    for _ in range(len(yolo_list)):

        if config["shape_unity"] == True:
            img = Image.fromarray(yolo_list[_].numpy())
            delta_img = Image.fromarray(torch.abs(metric_images[_]-yolo_list[_]).numpy())
            result = model(yolo_list[_].numpy(), conf=config["yolo_conf"], verbose=False)
        elif config["shape_unity"] == False:
            result = model(yolo_list[_,:hw_array[_,0],:hw_array[_,1]].numpy(),conf=config["yolo_conf"],verbose=False)
            delta_img = Image.fromarray(255 - torch.abs(metric_images[_,:hw_array[_,0],:hw_array[_,1]] - yolo_list[_,:hw_array[_,0],:hw_array[_,1]]).numpy())
            img = Image.fromarray(yolo_list[_,:hw_array[_,0],:hw_array[_,1]].numpy())


        img.save(adv_path+f'adv_'+'{0:04}'.format(_+1)+'.png','PNG')
        delta_img.save(delta_path+f'delta_'+'{0:04}'.format(_+1)+'.png','PNG')
        im_array = result[0].plot() 
        im = Image.fromarray(im_array)  
        im.save(f'{adv_result_path}adv_'+'{0:04}'.format(_+1)+'.png','PNG')  


if __name__ == '__main__':


    seed = 2
    seed_all(seed)

    config["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config["bound"] = 100  # Tthe maximum number of iteration for Remember and Forget process. We use 100.
    config["limit"] = 5e-2  # Bound threshold, denoted as η in the paper. We use 0.05
    config["attack_pixel"] = 0.05  # The attack dimension in the Remember process, denoted as α. We use 0.05.
    config["yolo_conf"] = 0.50  # Confidence threshold for detecting objects
    config["patient"] = 20  # Duration of condition, denoted as T. We use 20 in object detection.
    config["classifier"] = "yolo"   # Detector to attack, "yolo" or "ddq"
    config["dataset"] = "COCO"


    #  Set fixed path
    fix_path = os.path.abspath(os.getcwd())

    
    model = YOLO('yolov8n.pt')



    #  Dataset path
    file_path = fix_path+f"/COCO/images/val/"

    #  Result path
    result_path = fix_path + f"/results/COCO/original_result"
    adv_path = fix_path + f"/results/COCO/adv_images"
    adv_result_path = fix_path + f"/results/COCO/adv_result"
    delta_path = fix_path + f"/results/COCO/delta_images"

    

    os.makedirs(result_path, exist_ok=True)
    os.makedirs(adv_path, exist_ok=True)
    os.makedirs(adv_result_path, exist_ok=True)
    os.makedirs(delta_path, exist_ok=True)



    result_path = result_path + "/"
    adv_path = adv_path + "/"
    adv_result_path = adv_result_path + "/"
    delta_path = delta_path+"/"


    list_images= glob.glob(file_path + '*.jpg')
    list_images = sorted(list_images)

    img_list = []
    img_hw_list = []
    for image in list_images:
        img = np.array(Image.open(image))
        img_hw_list.append([img.shape[0],img.shape[1]])
        img_list.append(img)
    img_hw_list= np.array(img_hw_list)
    config["ni"] = len(img_list)
    img_array = img_list
    config["attack_pixel"] = int((img_hw_list.max(axis=0)[0]+img_hw_list.max(axis=1)[0])/2*config["attack_pixel"])
    
    #  RFPAR
    attack(model, img_array, config,img_hw_list)








