from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import numpy as np
from skimage import io

BATCH_SIZE = 8

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x : 0.299 * x[0] + 0.587 * x[1] + 0.114 * x[2]),
    transforms.Lambda(lambda x : x/x.max())
])

class custom_dataset(Dataset):

    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        self.folders = os.listdir(root)
        self.files = []
        self.labels = []
        self.map = dict(zip([i for i in range(len(self.folders))],
                            [s for s in self.folders]))
        self.inv_map = dict(zip([s for s in self.folders],
                                [i for i in range(len(self.folders))]))
        
        self.load_labels_and_files()

    def load_labels_and_files(self):
        for folder in self.folders:
            curr_files = os.listdir(os.path.join(self.root, folder))
            self.files.extend(curr_files)
            self.labels.extend([self.inv_map[folder]] * len(curr_files))
        
        self.dataset_length = len(self.files)

    def __getitem__(self, index):
         
        y = self.labels[index]
        path = self.root+self.map[y]+'//'+self.files[index]
        X = io.imread(path, as_gray = True)
        if(len(X.shape)==2): X = np.stack([X,X,X],axis=2)
        elif(X.shape[2]==4) : X = X[:,:,:3]
        y = self.labels[index]
        X = self.transforms(X)

        return X,y

    def __len__(self):
        return self.dataset_length
    

root_dir = "dataset\\mnist\\"
root_train = root_dir+"train\\"
root_val = root_dir+"val\\"

train_dataset = custom_dataset(root_train, transform)
val_dataset = custom_dataset(root_val, transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, 
                        shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, 
                        shuffle=False)