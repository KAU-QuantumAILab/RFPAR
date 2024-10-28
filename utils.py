import random
import numpy as np
import torch
import torch.distributions as dist
from torch.utils.data import Dataset




class MyBaseDataset(Dataset):
    #  Dataset for attack
    def __init__(self, x_data, y_data, transform=None):
        if transform != None:
            self.x_data = torch.tensor(np.array(x_data))
        else:
            self.x_data = x_data
        self.y_data = y_data
        self.transform = transform
        
    def __getitem__(self, index): 
        
        return self.x_data[index], self.y_data[index]
        
    def __len__(self): 
        return self.x_data.shape[0]


def seed_all(seed):
    #  Set fixed seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def normalize(train_data,config):
    #  Normalization for ImageNet
    if config["dataset"]=="ImageNet":
        if len(train_data.shape) == 4:
            mean=torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(config["device"])
            std=torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(config["device"])
            train_data = (train_data-mean)/std
        else:
            mean=torch.tensor([0.485, 0.456, 0.406]).view(3,1,1).to(config["device"])
            std=torch.tensor([0.229, 0.224, 0.225]).view(3,1,1).to(config["device"])
            train_data = (train_data-mean)/std
    return train_data



def organization(model,train_data,config):
    #  Prepare dataset
    train_set = []
    train_label = []
    
    label_list = torch.zeros(config["num_label"]).to(config["device"])
    if config["dataset"] == "CIFAR10":
        loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=1, shuffle=False)
    elif config["dataset"] == "ImageNet":
        loader = train_data

    model.eval()
    for idx,(s,label) in enumerate(loader):
        if label < config["num_label"]:
            if label_list[label] < config["num_data"] :

                    # classify the dataset and collection data

                s = s.to(config["device"])
                label = label.to(config["device"])
                
                with torch.no_grad():
                    y = torch.softmax(model(normalize(s,config)),dim=1)
                    y = torch.max(y,1).indices 

                    if (y == label) == True:
                        train_set.append(s)
                        train_label.append(label)
                        label_list[label] += 1

            if label_list.sum().item() == config["num_data"]*config["num_label"] : 
                print(label_list.max(),label_list.min())
                break
        # print(config["num_data"],config["num_label"])

    train_set = torch.cat(train_set,0)
    train_label = torch.cat(train_label,0)
    train_data = MyBaseDataset(train_set,train_label)
    print(train_set.shape)
    # torch.save(train_data,'resnext50_32x4d.pt')

    return train_data



def sample_action(actions_mean, action_std, config):
    #  Function to sample actions
    if not isinstance(action_std, torch.Tensor):
        cov_mat = torch.eye(config["action_dim"]).to(config["device"]) * action_std**2
    else:
        cov_mat = torch.diag_embed(action_std.pow(2)).to(config["device"])
    distribution = dist.MultivariateNormal(actions_mean, cov_mat)
    if config["attack_pixel"] == 1:
        actions = distribution.sample()
    else:
        actions = distribution.sample_n(config["attack_pixel"])
    

    actions_logprob = distribution.log_prob(actions)


    return actions, actions_logprob







def early_stopping(count, stop_count,limit=10, patient=5):
    #  Function to check convergence condition and duration. Outputs flag = True if duration exceeds the limit.

    flag=False

    if count <= limit:
        stop_count += 1

        if stop_count >= patient:
            flag = True

    else:
        stop_count = 0
    

    return stop_count, flag


#  Metric
#  Function to calculate L_0 and L_2 norms.
def step_function(last_data, new_data):
    return (new_data-last_data)>0

def L0_norm(original_img, change_img):
    norm = torch.count_nonzero(original_img-change_img).to('cpu').tolist()
    return [norm]

def L2_norm(original_img, change_img):
    norm = torch.norm((original_img-change_img), p=2).to('cpu').tolist()
    return [norm]
