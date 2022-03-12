import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from constants import DEVICE
from tqdm import tqdm
from sklearn.metrics import accuracy_score


def train_classifier_single_loop(model, images, labels, criterion, optimizer):
    
    pred = model(images)
    loss = criterion(pred, labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    acc = accuracy_score(labels.cpu(), pred.argmax(dim=1).cpu())
    
    return loss.item() * images.shape[0], acc * images.shape[0]

def val_classifier_single_loop(model, images, labels, criterion):
    pred = model(images)
    loss = criterion(pred, labels)
    acc = accuracy_score(labels.cpu(), pred.argmax(dim=1).cpu())
    return loss.item() * images.shape[0], acc * images.shape[0]

def train_frozen_and_tuned(model_frozen, model_tuned, train_dataloader, test_dataloader, epochs=40):
    
    frozen_metrics = []
    tuned_metrics = []
    
    optim_frozen = optim.Adam(model_frozen.parameters())
    optim_tuned = optim.Adam([
                          {"params": model_tuned.vgg.features.parameters(), "lr": 1e-5},
                          {"params": model_tuned.vgg.classifier.parameters(), "lr": 1e-3}
                          ]
                         )
    
    criterion = nn.CrossEntropyLoss()       
    
    for epoch in range(epochs):
        frozen_train_loss = 0
        tuned_train_loss = 0
        frozen_train_acc = 0
        tuned_train_acc = 0 
        
        frozen_test_loss = 0
        tuned_test_loss = 0
        frozen_test_acc = 0
        tuned_test_acc = 0

        model_frozen.train()
        model_tuned.train()
        print(f"Training Epoch {epoch}")
        for images, labels in tqdm(train_dataloader):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            loss_frozen, acc_frozen =  train_classifier_single_loop(model_frozen, images, labels, criterion, optim_frozen)
            loss_tuned, acc_tuned = train_classifier_single_loop(model_tuned, images, labels, criterion, optim_tuned)
            
            frozen_train_loss += loss_frozen
            tuned_train_loss += loss_tuned
            frozen_train_acc += acc_frozen
            tuned_train_acc += acc_tuned
           
        frozen_train_loss = frozen_train_loss / len(train_dataloader.dataset)
        tuned_train_loss = tuned_train_loss / len(train_dataloader.dataset)
        frozen_train_acc = frozen_train_acc / len(train_dataloader.dataset)
        tuned_train_acc = tuned_train_acc / len(train_dataloader.dataset)
        
        print(f"frozen train loss: {frozen_train_loss:.3f} | tuned train loss: {tuned_train_loss:.3f}")
        print(f"frozen train acc: {frozen_train_acc:.3f} | tuned train acc: {tuned_train_acc:.3f}")

        model_frozen.eval()
        model_tuned.eval()

        print(f"Validation Epoch {epoch}")
        with torch.no_grad():
            for images, labels in tqdm(test_dataloader):
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                
                loss_frozen, acc_frozen = val_classifier_single_loop(model_frozen, images, labels, criterion)
                loss_tuned, acc_tuned = val_classifier_single_loop(model_tuned, images, labels, criterion)

                frozen_test_loss += loss_frozen
                tuned_test_loss += loss_tuned
                frozen_test_acc += acc_frozen
                tuned_test_acc += acc_tuned
                
        frozen_test_loss = frozen_test_loss / len(test_dataloader.dataset)
        tuned_test_loss = tuned_test_loss / len(test_dataloader.dataset)
        frozen_test_acc = frozen_test_acc / len(test_dataloader.dataset)
        tuned_test_acc = tuned_test_acc / len(test_dataloader.dataset)

        print(f"frozen test loss: {frozen_test_loss:.3f} | tuned test loss: {tuned_test_loss:.3f}")
        print(f"frozen test acc: {frozen_test_acc:.3f} | tuned test acc: {tuned_test_acc:.3f}")
        
        
        frozen_metrics.append([frozen_train_loss, frozen_test_loss, frozen_train_acc, frozen_test_acc])
        tuned_metrics.append([tuned_train_loss, tuned_test_loss, tuned_train_acc, tuned_test_acc])
    
    return frozen_metrics, tuned_metrics

# ======= AUTO ENCODER ========== 

def VAE_criterion(pred, mu, logvar, target, labels, num_classes=10):
    mse_loss = F.mse_loss(pred, target, reduction="sum")
    KL_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    centers = torch.stack([mu[labels == i].mean(axis=0) for i in range(num_classes)])
    center_loss = F.mse_loss(mu, centers[labels], reduction="sum")
    return mse_loss + KL_divergence + center_loss

def train_encoder_single_loop(encoder, representations, labels, optimizer):
    pred, mu, logvar = encoder(representations)
    loss = VAE_criterion(pred, mu, logvar, representations, labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss.item()

def train_encoder(encoder, representations, labels, epochs=1000):
    optimizer = optim.Adam(encoder.parameters())
    losses = []
    for epoch in tqdm(range(epochs)):
        loss = train_encoder_single_loop(encoder, representations, labels, optimizer)
        losses.append(loss)
    return losses