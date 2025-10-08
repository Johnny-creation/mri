import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import datetime
from evaluate import evaluate_multitask

def train_multitask(model, train_loader, val_loader, epochs, lr, dropout_rate, weight_decay, time_loss_weight=2.0, log_dir="output/"):
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_dir, f"training_log_{timestamp}.txt")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    lesion_criterion = nn.CrossEntropyLoss()
    time_criterion = nn.CrossEntropyLoss()
    
    device = next(model.parameters()).device
    
    best_val_auc = 0.0
    best_model_path = os.path.join(log_dir, f"model_best.pt")
    
    for epoch in range(epochs):
        model.train()
        total_lesion_loss = 0
        total_time_loss = 0
        lesion_correct = 0
        time_correct = 0
        total_samples = 0
        
        for batch in train_loader:
            if batch is None:
                continue
                
            images, lesion_labels, time_labels, _ = batch
            images = images.to(device)
            lesion_labels = lesion_labels.to(device)
            time_labels = time_labels.to(device)
            
            optimizer.zero_grad()
            
            lesion_out, time_out = model(images)
            
            lesion_loss = lesion_criterion(lesion_out, lesion_labels)
            time_loss = time_criterion(time_out, time_labels)
            
            total_loss = lesion_loss + time_loss_weight * time_loss
            total_loss.backward()
            optimizer.step()
            
            total_lesion_loss += lesion_loss.item()
            total_time_loss += time_loss.item()
            
            _, lesion_preds = torch.max(lesion_out, 1)
            _, time_preds = torch.max(time_out, 1)
            
            lesion_correct += (lesion_preds == lesion_labels).sum().item()
            time_correct += (time_preds == time_labels).sum().item()
            total_samples += images.size(0)
        
        avg_lesion_loss = total_lesion_loss / len(train_loader)
        avg_time_loss = total_time_loss / len(train_loader)
        lesion_acc = lesion_correct / total_samples
        time_acc = time_correct / total_samples
        

        if (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
            val_metrics = evaluate_multitask(model, val_loader, device, log_filename)
            log_line = (f"Epoch {epoch+1}:\n"
                       f"Train - Lesion Loss: {avg_lesion_loss:.4f}, Time Loss: {avg_time_loss:.4f}\n"
                       f"Train - Lesion Acc: {lesion_acc:.4f}, Time Acc: {time_acc:.4f}\n"
                       f"Val - Lesion Acc: {val_metrics['lesion_acc']:.4f}, Time Acc: {val_metrics['time_acc']:.4f}\n"
                       f"Val - Lesion AUC: {val_metrics['lesion_auc']:.4f}, Time AUC: {val_metrics['time_auc']:.4f}\n"
                       f"Val - Lesion F1: {val_metrics['lesion_f1']:.4f}, Time F1: {val_metrics['time_f1']:.4f}\n"
                       f"Val - Lesion Sensitivity: {val_metrics['lesion_sensitivity']:.4f}, Specificity: {val_metrics['lesion_specificity']:.4f}\n"
                       f"Val - Lesion PPV: {val_metrics['lesion_ppv']:.4f}, NPV: {val_metrics['lesion_npv']:.4f}\n"
                       f"Val - Time Sensitivity: {val_metrics['time_sensitivity']:.4f}, Specificity: {val_metrics['time_specificity']:.4f}\n"
                       f"Val - Time PPV: {val_metrics['time_ppv']:.4f}, NPV: {val_metrics['time_npv']:.4f}\n")
        else:
            log_line = (f"Epoch {epoch+1}:\n"
                       f"Train - Lesion Loss: {avg_lesion_loss:.4f}, Time Loss: {avg_time_loss:.4f}\n"
                       f"Train - Lesion Acc: {lesion_acc:.4f}, Time Acc: {time_acc:.4f}\n")
        
        print(log_line.strip())
        with open(log_filename, "a") as f:
            f.write(log_line)
        

        if (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
            if val_metrics['time_auc'] > best_val_auc:
                best_val_auc = val_metrics['time_auc']
                torch.save(model.state_dict(), best_model_path)
        
        scheduler.step()
    
    final_model_path = os.path.join(log_dir, f"model_final.pt")
    torch.save(model.state_dict(), final_model_path)
    
    return log_filename, final_model_path, best_model_path, val_metrics if (epoch + 1) % 10 == 0 or (epoch + 1) == epochs else None