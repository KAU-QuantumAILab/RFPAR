import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import Adversarial_RL_simple as Adversarial_RL_simple
import Environment
import os
import glob
from PIL import  Image
from config import config
from utils import seed_all, sample_action, early_stopping, MyBaseDataset, L0_norm,L2_norm




def attack(model, train_data, config):


    env = Environment.Env(model,config=config)
    env.init=True
    
    #  Setting
    #  Image part of Memory
    update_images = train_data.x_data.clone().to(config["device"])
    metric_images = train_data.x_data.clone().to(config["device"])
    idx_list = (torch.LongTensor([_ for _ in range(len(metric_images))])).to(config["device"])
    init_ori_prob = []
    ori_prob = None
     

    i=0
    total = 0
    it = []
    L0 = []
    L2 = []
    save_labels = torch.zeros(update_images.shape[0])    

    for p in range(config["bound"]):    
        #  This is the Forget Process. We repeat this process until T.
        #  Load agent & Forget parameters to initialize
        agent = Adversarial_RL_simple.REINFORCE(config).to(config["device"])



        if p != 0:
            init_ori_prob = env.ori_prob - update_rewards
        
        env.ori_prob = init_ori_prob

        flag = False
        stop_count = 0
        deceived_count = 0
        succes_list =[]
        succes_list_idx = []
        succes_labels = []
        delta_list = []
        #  Reward part of Memory
        update_rewards = torch.zeros(update_images.shape[0]).to(config["device"])
        

        while 1:
            #  This is the Remember Process. We train RL until convergence.
            #  Prepare Loader
            train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=config["batch_size"], shuffle=False)
            i += 1
            
            train_x = []
            train_y = []
            change_train_x = []
            DIP = 0
            total_rewards_list = []
            
            total_change_list = torch.tensor([]).to(config["device"])


            for bt, (s,labels) in enumerate(train_loader):
                #  Move data to the device
                s = s.to(config["device"])
                labels = labels.to(config["device"])

                #  Sample action
                action_means, action_stds = agent(s)
                action_stds = torch.clamp(action_stds, 0.1, 10)  # std가 0이 되는 것을 방지
                actions, actions_logprob = sample_action(action_means, action_stds, config)


                if config["attack_pixel"] != 1:
                    actions = actions.view(-1,5)
                    actions_logprob = actions_logprob.sum(axis=0)

                if env.init == False:
                    ori_prob = env.ori_prob[bt*config["batch_size"]:bt*config["batch_size"]+len(labels)]

                #  Generate reward
                rewards, change_list, changed_images, change_labels = env.step(s.type(torch.float32), actions, labels,ori_prob)

                
                if env.init == True:
                    init_ori_prob.append(env.ori_prob)
                

                #  Collecting fail images
                train_x.append(s[change_list==0])
                train_y.append(labels[change_list==0])
                #  Collecting adversarial images
                succes_list.append(changed_images[change_list==1])
                succes_list_idx.append(idx_list[config["batch_size"]*bt:config["batch_size"]*bt+config["batch_size"]][change_list==1])
                succes_labels.append(change_labels[change_list==1])
                #  Update iteration
                it = it + [i for _ in range(change_list.sum())]

                #  Setting for updating Memory
                change_train_x.append(changed_images)
                total_change_list = torch.cat((total_change_list,change_list),dim=0).long()
                total_rewards_list += rewards.tolist()

                #  Accumulate the number of adversarial images in 1 training
                DIP += change_list.sum()

                

                
                #  Training RL
                agent.r = rewards + change_list
                agent.prob = actions_logprob

                agent.train_net()
 
            if env.init == True:
                env.ori_prob = torch.cat(init_ori_prob,0)


            if not train_x: break

            change_train_x = torch.cat(change_train_x,0)

            train_y = torch.cat(train_y,0)
            train_x = torch.cat(train_x,0)

            #  Count adversarial images in Remember process
            deceived_count += DIP

            #  saving metric
            if total_change_list.sum()>0:
                for idx, value in enumerate(total_change_list):
                    if value == 1:
                        L0 = L0+L0_norm(metric_images[idx],change_train_x[idx])
                        L2 = L2+L2_norm(metric_images[idx],change_train_x[idx])
                        delta_list.append(abs(metric_images[idx] - change_train_x[idx]))
            metric_images = metric_images[total_change_list==0]
            change_train_x = change_train_x[total_change_list==0]
            idx_list = idx_list[total_change_list==0]
            env.ori_prob = env.ori_prob[total_change_list==0]


            #  Memory update
            update_rewards = update_rewards[total_change_list==0]
            update_images = update_images[total_change_list==0]
            standard = update_rewards.mean()

            total_rewards_list = torch.tensor(total_rewards_list).to(config["device"])[total_change_list==0]
            update_rewards = torch.cat((update_rewards.unsqueeze(dim=0), total_rewards_list.unsqueeze(dim=0)),dim=0)

            update_rewards_indices = torch.max(update_rewards,axis=0).indices
            update_rewards = update_rewards[update_rewards_indices,np.arange(total_rewards_list.shape[0])]

            update_rewards_sum = update_rewards.mean() + total_change_list.sum()
            
            update_images = torch.cat((update_images.unsqueeze(dim=0),change_train_x.unsqueeze(dim=0)),dim=0)
            update_images = update_images[update_rewards_indices,np.arange(total_rewards_list.shape[0]),:,:]

                
            #  Convergence condition
            stop_count, flag = early_stopping(((update_rewards_sum-standard)/standard), stop_count, limit=config["limit"],patient=config["patient"])



            if flag == True:
                train_x=update_images
                train_data = MyBaseDataset(train_x, train_y)
                break
            elif flag == False:
                train_data = MyBaseDataset(train_x, train_y)
                env.init=False


        #  total count
        total += deceived_count
        
        if succes_list:

            succes_list = torch.cat(succes_list,0)
            succes_list_idx = torch.cat(succes_list_idx,0)
            succes_labels = torch.cat(succes_labels,0)
            
            for n, changed_image in enumerate(succes_list):
                img = Image.fromarray((changed_image.permute(1,2,0).detach().to('cpu').numpy()*255).astype('uint8'))
                delta = Image.fromarray(255 - (delta_list[n].permute(1,2,0).detach().to('cpu').numpy()*255).astype('uint8'))
                img.save(adv_path+f'adv_'+'{0:04}'.format(int(succes_list_idx[n]))+'.png','PNG')
                delta.save(delta_path+f'delta_'+'{0:04}'.format(int(succes_list_idx[n]))+'.png','PNG')
                #  label change
                save_labels[succes_list_idx[n]] = succes_labels[n]



        print(f'Forget:{p+1}, Deceived Image : {deceived_count},total Deceived : {total}, rest : {len(train_data)}')
        

        
        if len(train_data) == 0:
            print("all Images are deceived")
            break
    torch.save(save_labels,label_path+f'adv_labels.pt')




if __name__ == '__main__':

    seed = 2
    seed_all(seed)

    config["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config["bound"] = 100  # the maximum number of iteration for Remember and Forget process. We use 100.
    config["limit"] = 5e-2  # Bound threshold, denoted as η in the paper. We use 0.05
    config["attack_pixel"] = 0.01  # The attack dimension in the Remember process, denoted as α. We use 0.01.
    config["patient"] = 3  # Duration of condition, denoted as T. We use 3 in image classification.
    config["classifier"] = "resnext50_32x4d"  # Classifier to attack, "resnext50_32x4d"
    config["dataset"] = "ImageNet"

    print("Device name: {}".format(torch.cuda.get_device_name(0)))
    print("Using Device: {}".format(torch.cuda.is_available()))
    print("Seed: {}".format(seed))
    
    #  Set fixed path
    fix_path = os.path.abspath(os.getcwd())

    #  Set result path
    adv_path = fix_path + f'/results/{config["dataset"]}/adv_images'
    delta_path = fix_path + f'/results/{config["dataset"]}/delta_images'
    label_path = fix_path + f'/results/{config["dataset"]}/'

    os.makedirs(adv_path, exist_ok=True)
    os.makedirs(delta_path, exist_ok=True)


    adv_path = adv_path + "/"
    delta_path = delta_path+"/"



    

    


            

    
    #  Prepare model
    model = models.resnext50_32x4d(pretrained=True).to(config["device"])
    model = torch.nn.DataParallel(model)
    model.eval()                                               
    

    #  Prepare dataset
    file_path = fix_path+f'/ImageNet/'
    list_images= sorted(glob.glob(file_path + 'images/*.png'))
    labels = torch.load(file_path+f'label.pt').to(config["device"])

    img_list = []
    for name in list_images:
        img = np.array(Image.open(name))
        img_list.append(img)
    img_list = (torch.tensor(np.array(img_list)).permute(0,3,1,2)/255).to(config["device"])
    train_data = MyBaseDataset(img_list,labels)

    

    #  Set attack pixel
    config["attack_pixel"] = int((img_list.shape[2]+img_list.shape[3])/2*config["attack_pixel"])



    
    print(f'model : {config["classifier"]}, attack pixel : {config["attack_pixel"]}, patient : {config["patient"]}')

    attack(model, train_data,config)


