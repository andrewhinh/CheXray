import json
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from modules.utils.dicom import PILDicom2

train_transform = transforms.Compose([transforms.Resize(256), transforms.RandomCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor()])
valtest_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
normalize = transforms.Compose([transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

class RepGenDataset(Dataset):
    def __init__(self, data, isvaltest, viewtype, ispred, classes):
        self.data = data
        self.isvaltest = isvaltest
        self.viewtype = viewtype
        self.ispred = ispred
        self.transform = valtest_transform if isvaltest else train_transform
        self.normalize = normalize
        self.classes = classes
        self._DataLoader__idxs = list(data.index)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        row = self.data.loc[idx]
        #Images
        image = PILDicom2.create(self.data.loc[idx, self.viewtype])
        image = self.transform(image)   
        image = image.type(torch.float64)
        image = torch.cat((image, image, image), 0)
        #Normalize to [0, 1] before paper norm
        image-=image.min()
        image/=image.max()
        #Paper Norm
        image = self.normalize(image) 
        image = image.permute(1,2,0)
        #Rest
        if self.ispred: 
            report_ids=[0]
            report_masks=[1]
            seq_length=1
            onehot_lbls=torch.tensor([0.0]*len(self.classes), dtype=torch.float64)
        else:
            report_ids = json.loads(row['idx_reports'])
            report_masks = json.loads(row['mask_reports'])
            seq_length = row['tok_reports_length']
            onehot_lbls = torch.tensor(list(row[self.classes]))
        sample = (image, report_ids, report_masks, seq_length, onehot_lbls)
        return sample