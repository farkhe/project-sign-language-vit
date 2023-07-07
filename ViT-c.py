
import pandas as pd
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
from transformers import ViTModel
from PIL import Image
from tqdm.notebook import tqdm
from sklearn import model_selection
from tqdm import tqdm
from transformers import ViTModel
from torch.optim.lr_scheduler import StepLR
from transformers import ViTForImageClassification
from statistics import mean

plt.style.use('fivethirtyeight')

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

BASE_PATH = 'alpha2'


IMG_SIZE = 224
BATCH_SIZE = 16
LR = 1e-4
EPOCHS = 0
num_classes = 28


train = pd.read_csv('alpha2/classification.csv')
train.head()
print(train.shape)

plt.figure(figsize=(6, 3))

train['image_class'].value_counts().plot(
    kind='bar',
    color='#558364',
    width=0.7
)

plt.xlabel('Label', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Distribution of Labels', fontsize=15)
plt.xticks(rotation=360)
#plt.show()

train_df, val_df = model_selection.train_test_split(
    train, test_size=0.12, random_state=42, stratify=train['image_class'].values
)


class signDataset(torch.utils.data.Dataset):
    
    def __init__(self, df, data_path=BASE_PATH, mode='train', transforms=None):
        super().__init__()
        self.df_data = df.values
        self.data_path = data_path
        self.transforms = transforms 
        self.mode = mode
        self.data_dir = 'train_images' if mode == 'train' else 'test_images'
    
    def __len__(self):
        return len(self.df_data)
    
    def __getitem__(self, index):
        img_name, label, x1_pixel, y1_pixel, x2_pixel, y2_pixel = self.df_data[index]
        img_path = os.path.join(self.data_path, self.data_dir, img_name)
        img = Image.open(img_path).convert('RGB')

        # Convert normalized coordinates to pixel values
        img_width, img_height = img.size
        x1 = int(x1_pixel * img_width)
        y1 = int(y1_pixel * img_height)
        x2 = int(x2_pixel * img_width)
        y2 = int(y2_pixel * img_height)
            
        if self.transforms is not None:
            img = self.transforms(img)
        
        return img, label, x1, y1, x2, y2  # Return image, label, and coordinates




transforms_train = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.2),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomResizedCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

transforms_val = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

train_dataset = signDataset(df=train_df, data_path=BASE_PATH, mode='train', transforms=transforms_train)
val_dataset = signDataset(df=val_df, data_path=BASE_PATH, mode='train', transforms=transforms_val)

print(len(train_dataset), len(val_dataset))

model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
model.classifier = nn.Linear(in_features=768, out_features=num_classes, bias=True)


# add the coordination to the model
class ViTWithCoordinates(nn.Module):
    def __init__(self, num_classes):
        super(ViTWithCoordinates, self).__init__()
        self.vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224")
        self.coord_branch = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 64)
        )
        self.classifier = nn.Linear(self.vit_model.config.hidden_size + 64, num_classes)

    def forward(self, img, x1, y1, x2, y2):
        vit_output = self.vit_model(img)
        vit_features = vit_output.last_hidden_state[:, 0, :]
        coords = torch.stack([x1, y1, x2, y2], dim=1).to(img.dtype)
        coord_features = self.coord_branch(coords)
        combined_features = torch.cat([vit_features, coord_features], dim=1)
        output = self.classifier(combined_features)
        return output

model = ViTWithCoordinates(num_classes)
model.to(device)


train_losses = []
train_accs = []
val_losses = []
val_accs = []

def train_model(model, train_dataset, val_dataset, learning_rate, epochs):

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.9)
    
    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    for epoch_num in range(epochs):

            total_acc_train = 0
            total_loss_train = 0
            #add the coordination to the training and validation loop
            for train_images, train_labels, x1, y1, x2, y2 in tqdm(train_dataloader):

                train_images = train_images.to(device)
                train_labels = train_labels.to(device)
                x1 = x1.to(device)
                y1 = y1.to(device)
                x2 = x2.to(device)
                y2 = y2.to(device)
                
                optimizer.zero_grad()
            
                # Pass image and coordinates through the model
                output = model(train_images, x1, y1, x2, y2)
            
                batch_loss = criterion(output, train_labels.long()) 
                total_loss_train += batch_loss.item()
                
                _, predicted = torch.max(output, 1)
                acc = (predicted == train_labels).sum().item()
                total_acc_train += acc
            
                batch_loss.backward()
                optimizer.step()
                
            scheduler.step()
            
            total_acc_val = 0
            total_loss_val = 0

            with torch.no_grad():

                for val_images, val_labels, x1, y1, x2, y2  in val_dataloader:
                    
                    val_images = val_images.to(device)
                    val_labels = val_labels.to(device)
                    x1 = x1.to(device)
                    y1 = y1.to(device)
                    x2 = x2.to(device)
                    y2 = y2.to(device)
                    
                    output = model(val_images, x1, y1, x2, y2)
            
                    batch_loss = criterion(output, val_labels.long()) 
                    total_loss_val += batch_loss.item()
                    
                    _, predicted = torch.max(output, 1)
                    acc = (predicted == val_labels).sum().item()
                    total_acc_val += acc
            
            print(f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_dataset): .3f} \
            | Train Accuracy: {total_acc_train / len(train_dataset): .3f} \
            | Val Loss: {total_loss_val / len(val_dataset): .3f} \
            | Val Accuracy: {total_acc_val / len(val_dataset): .3f}')
            
            train_losses.append(total_loss_train / len(train_dataset))
            train_accs.append(total_acc_train / len(train_dataset))
            val_losses.append(total_loss_val / len(val_dataset))
            val_accs.append(total_acc_val / len(val_dataset))


train_model(model, train_dataset, val_dataset, LR, EPOCHS)

plt.figure(figsize=(18, 6))

plt.plot(
    train_losses, 
    label='Train_Losses', 
    color='red', 
    linewidth=1.5
)
plt.plot(
    val_losses, 
    label='Val_Losses', 
    color='blue', 
    linewidth=1.5
)

plt.plot(
    train_accs, 
    label='Train_Accuracy', 
    color='green', 
    linewidth=1.5
)
plt.plot(
    val_accs, 
    label='Val_Accuracy', 
    color='pink', 
    linewidth=1.5
)

plt.xlabel('Epoch')
plt.ylabel('Loss / Accuracy')
plt.title('Loss / Accuracy on train / validation')
plt.legend()
plt.show()


sub_df = pd.read_csv('alpha2/classification_fin.csv')
sub_df.head()
preds = []
test_dataset = signDataset(df=sub_df, data_path=BASE_PATH, mode='test', transforms=transforms_val)
mapping = {
    "0": "ALIF",
    "1": "BAA",
    "2": "TA",
    "3": "THA",
    "4": "JEEM",
    "5": "HAA",
    "6": "KHAA",
    "7": "DELL",
    "8": "DHELL",
    "9": "RAA",
    "10": "ZAY",
    "11": "SEEN",
    "12": "SHEEN",
    "13": "SAD",
    "14": "DAD",
    "15": "TAA",
    "16": "DHAA",
    "17": "AYN",
    "18": "GHAYN",
    "19": "FAA",
    "20": "QAAF",
    "21": "KAAF",
    "22": "LAAM",
    "23": "MEEM",
    "24": "NOON",
    "25": "HA",
    "26": "WAW",
    "27": "YA",
}
def predict(model, test_dataset):
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)
    with torch.no_grad():
        for test_images, test_labels, _, _, _, _ in tqdm(test_dataloader):
            test_images = test_images.to(device)
            

            output = model(test_images)

            _, predicted = torch.max(output.logits.data, 1)
            predicted = predicted.cpu().data.numpy().tolist()  # Convert predicted to a list

            preds.extend([mapping[str(pred)] for pred in predicted])


        print(preds)
    return preds

predict(model, test_dataset)

# Saving the predictions to a CSV file
sub_df['image_class'] = preds
sub_df.to_csv('classification_fin.csv', index=False)
print(sub_df.head())


