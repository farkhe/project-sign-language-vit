import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.models as models
from transformers import ViTFeatureExtractor, ViTForImageClassification
import torch
from PIL import Image
from tqdm.notebook import tqdm
from sklearn import model_selection
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from transformers import ViTForImageClassification
from statistics import mean
import numpy as np
from scipy.spatial.distance import cdist
import random

plt.style.use('fivethirtyeight')

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

BASE_PATH = 'alpha2'

IMG_SIZE = 224
BATCH_SIZE = 16
LR = 1e-4
EPOCHS = 5
num_classes = 28
num_clusters = num_classes

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train = pd.read_csv('alpha2/classification.csv')
train.head()
print(train.shape)

plt.figure(figsize=(6, 3))

train['image_class'].value_counts().plot(
    kind='bar',
    color='#558364',
    width=0.7
)

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
        img_name, label ,_,_,_,_= self.df_data[index]
        img_path = os.path.join(self.data_path, self.data_dir, img_name)
        img = Image.open(img_path).convert('RGB')
            
        if self.transforms is not None:
            img = self.transforms(img)
                
        return img, label



transforms_train = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.2),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomResizedCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

transforms_val = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

train_dataset = signDataset(df=train_df, data_path=BASE_PATH, mode='train', transforms=transforms_train)
val_dataset = signDataset(df=val_df, data_path=BASE_PATH, mode='train', transforms=transforms_val)


model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")

model.classifier = nn.Linear(in_features=768, out_features=num_classes, bias=True)
model.to(device)



train_losses = []
train_accs = []
val_losses = []
val_accs = []


#Defines the Fuzzy C-Means (FCM) clustering function.
def FCM(data, centers, m=2, max_iters=30, tol=1e-5):

    for i in range(max_iters):
        # Compute distances between samples and centers
        distances = cdist(data.cpu().numpy(), centers.cpu().numpy())

        # Add epsilon to distances to avoid division by zero
        distances = np.maximum(distances, np.finfo(float).eps)

        # Compute membership probabilities
        membership = np.power(distances, -2 / (m - 1))
        membership[np.isinf(membership)] = np.finfo(float).max
        membership[np.isnan(membership)] = np.finfo(float).eps
        membership = membership / np.sum(membership, axis=1, keepdims=True)

        # Update cluster centers
        centers_prev = centers.clone()
        centers_denom = np.power(membership, m)
        centers_numer = np.matmul(data.cpu().numpy().T, centers_denom)
        centers_numer /= np.sum(centers_denom, axis=0, keepdims=True)
        centers_numer[np.isnan(centers_numer)] = 0
        centers = torch.Tensor(centers_numer.T).to(device)

        # Check convergence
        center_shift = torch.norm(centers - centers_prev)
        
        if center_shift < tol:
            break

    # Compute final membership probabilities
    distances = cdist(data.cpu().numpy(), centers.cpu().numpy())
    distances = np.maximum(distances, np.finfo(float).eps)
    membership = np.power(distances, -2 / (m - 1))
    membership[np.isinf(membership)] = np.finfo(float).max
    membership[np.isnan(membership)] = np.finfo(float).eps
    denominator = np.sum(membership, axis=1, keepdims=True)
    denominator[denominator == 0] = 1e-8
    membership = membership / denominator
    print(membership)

    # Assign samples to clusters
    preds = torch.Tensor(np.argmax(membership, axis=1)).long().to(device)

    return preds



to_pil = transforms.ToPILImage()
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
        
        feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")   
        for train_images, train_labels in tqdm(train_dataloader):
            train_images = train_images.to(device)
            train_labels = train_labels.to(device)
            features = []
        
            optimizer.zero_grad()
        
            output = model(train_images)
        
            batch_loss = criterion(output.logits, train_labels.long())
            total_loss_train += batch_loss.item()
        
            _, predicted = torch.max(output.logits.data, 1)
            acc = (predicted == train_labels).sum().item()
            total_acc_train += acc
            
            
            normalized_images =  train_images
            
            # Convert normalized images to PIL images
            pil_images = [to_pil(image) for image in normalized_images]
            
            # Extract features using the feature_extractor
            inputs = feature_extractor(images=pil_images, return_tensors="pt")
            last_hidden_state = inputs.pixel_values
        
            features.append(last_hidden_state)
            features = torch.cat(features, dim=0)
            features = features.view(features.size(0), -1)
        
            centers = torch.stack([random.choice(features) for _ in range(num_clusters)])
            centers = centers.to(device)
            preds = FCM(features, centers)
        
            clustering_loss = criterion(output.logits, preds.long())
            total_loss = clustering_loss + batch_loss
        
            total_loss.backward()
            optimizer.step()

        scheduler.step()

        total_acc_val = 0
        total_loss_val = 0

        with torch.no_grad():

            for val_images, val_labels in val_dataloader:

                val_images = val_images.to(device)
                val_labels = val_labels.to(device)

                output = model(val_images)

                batch_loss = criterion(output.logits, val_labels.long())
                
                total_loss_val += batch_loss.item()

                _, predicted = torch.max(output.logits.data, 1)
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





