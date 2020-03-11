import os 
from glob import glob
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset



class ModelNet40Views(Dataset):
    def __init__(self, data_root,  base_model, mode='train'):
        super(ModelNet40Views, self).__init__()

        self.data_root = data_root
        self.mode = mode
        self.image_dirs = []
        self.labels = []
        self.image_dict = {}
        
        if base_model in ('ALEXNET', 'VGG13', 'VGG13BN', 'VGG11BN', 'RESNET50', 'GOOGLENET'):
            self.img_size = 224
        elif base_model in ('RESNET101'):
            self.img_size = 227
        elif base_model in ('INCEPTION_V3'):
            self.img_size = 299
        else:
            raise NotImplementedError

        self.transform = transforms.Compose([
                    transforms.Resize(self.img_size),
                    transforms.ToTensor()
                ])

        class_list = os.listdir(self.data_root)
        if self.mode == 'train':
            for oneclass in class_list:
                self.image_dict[oneclass] = glob(os.path.join(data_root, oneclass, 'train', '*.jpg'))
        elif self.mode == 'val':
            for oneclass in class_list:
                self.image_dict[oneclass] = glob(os.path.join(data_root, oneclass, 'test', '*.jpg'))          
        else: 
            raise NotImplementedError
        
        for class_key in self.image_dict:
            name_dict = {}
            for image_dir in self.image_dict[class_key]:
                image_class = '_'.join(os.path.split(image_dir)[1].split('.')[0].split('_')[:-1])
                if image_class in name_dict:
                    name_dict[image_class].append(image_dir)
                else:
                    name_dict[image_class] = [image_dir]

            for image_class, dirs in name_dict.items():
                self.image_dirs.append(dirs)
                self.labels.append(class_list.index(class_key))
       
        self.image_num = len(self.image_dirs) if len(self.image_dirs)==len(self.labels) else print("labels don't match")
        

    def __getitem__(self, idx):
        images = [self.transform(Image.open(image)) for image in self.image_dirs[idx]] 
        return torch.stack(images).float(), self.labels[idx]
    
    def __len__(self):
        return self.image_num


if __name__ == '__main__':
    train = ModelNet40Views(data_root='', base_model='ALEXNET', mode="train")
    # test = ModelNet40(config=args, partition='test')
    # for data, label in train:
    #     print(data.shape)
    #     print(label)
    data, label = train[514]
    print('data: {}'.format(data.shape))
    print('label: {}'.format(label))