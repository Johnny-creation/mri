import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

def evaluate_multitask(model, loader, device, log_file=None):
    model.eval()
    lesion_correct = 0
    time_correct = 0
    total_samples = 0
    
    lesion_probs = []
    time_probs = []
    lesion_labels = []
    time_labels = []
    

    lesion_tp, lesion_fp, lesion_fn, lesion_tn = 0, 0, 0, 0
    time_tp, time_fp, time_fn, time_tn = 0, 0, 0, 0
    
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
                
            images, lesion_labels_batch, time_labels_batch, _ = batch
            images = images.to(device)
            lesion_labels_batch = lesion_labels_batch.to(device)
            time_labels_batch = time_labels_batch.to(device)
            
            lesion_out, time_out = model(images)
            
            _, lesion_preds = torch.max(lesion_out, 1)
            _, time_preds = torch.max(time_out, 1)
            
            lesion_correct += (lesion_preds == lesion_labels_batch).sum().item()
            time_correct += (time_preds == time_labels_batch).sum().item()
            total_samples += images.size(0)
            
            lesion_probs_batch = torch.softmax(lesion_out, dim=1)[:, 1].cpu().numpy()
            time_probs_batch = torch.softmax(time_out, dim=1)[:, 1].cpu().numpy()
            lesion_labels_np = lesion_labels_batch.cpu().numpy()
            time_labels_np = time_labels_batch.cpu().numpy()
            lesion_preds_np = lesion_preds.cpu().numpy()
            time_preds_np = time_preds.cpu().numpy()
            

            for pred, label in zip(lesion_preds_np, lesion_labels_np):
                if pred == 1 and label == 1:
                    lesion_tp += 1
                elif pred == 1 and label == 0:
                    lesion_fp += 1
                elif pred == 0 and label == 1:
                    lesion_fn += 1
                elif pred == 0 and label == 0:
                    lesion_tn += 1
            
            lesion_probs.extend(lesion_probs_batch)
            lesion_labels.extend(lesion_labels_np)


            for i in range(len(lesion_preds_np)):
                if lesion_preds_np[i] == 1:
                    time_probs.append(time_probs_batch[i])
                    time_labels.append(time_labels_np[i])

                    if time_preds_np[i] == 1 and time_labels_np[i] == 1:
                        time_tp += 1
                    elif time_preds_np[i] == 1 and time_labels_np[i] == 0:
                        time_fp += 1
                    elif time_preds_np[i] == 0 and time_labels_np[i] == 1:
                        time_fn += 1
                    elif time_preds_np[i] == 0 and time_labels_np[i] == 0:
                        time_tn += 1
    
    lesion_acc = lesion_correct / total_samples if total_samples > 0 else 0.0
    time_acc = time_correct / total_samples if total_samples > 0 else 0.0
    
    lesion_auc = roc_auc_score(lesion_labels, lesion_probs) if len(set(lesion_labels)) > 1 else 0.0
    try:
        time_auc = roc_auc_score(time_labels, time_probs) if time_labels else 0.0
    except ValueError:
        time_auc = 0.0
    
    lesion_f1 = f1_score(lesion_labels, [1 if p > 0.5 else 0 for p in lesion_probs]) if lesion_labels else 0.0
    time_f1 = f1_score(time_labels, [1 if p > 0.5 else 0 for p in time_probs]) if time_labels else 0.0
    

    lesion_sensitivity = lesion_tp / (lesion_tp + lesion_fn + 1e-5)
    lesion_specificity = lesion_tn / (lesion_tn + lesion_fp + 1e-5)
    lesion_ppv = lesion_tp / (lesion_tp + lesion_fp + 1e-5)
    lesion_npv = lesion_tn / (lesion_tn + lesion_fn + 1e-5)
    
    time_sensitivity = time_tp / (time_tp + time_fn + 1e-5) if time_labels else 0.0
    time_specificity = time_tn / (time_tn + time_fp + 1e-5) if time_labels else 0.0
    time_ppv = time_tp / (time_tp + time_fp + 1e-5) if time_labels else 0.0
    time_npv = time_tn / (time_tn + time_fn + 1e-5) if time_labels else 0.0
    
    metrics = {
        'lesion_acc': lesion_acc,
        'time_acc': time_acc,
        'lesion_auc': lesion_auc,
        'time_auc': time_auc,
        'lesion_f1': lesion_f1,
        'time_f1': time_f1,
        'lesion_sensitivity': lesion_sensitivity,
        'lesion_specificity': lesion_specificity,
        'lesion_ppv': lesion_ppv,
        'lesion_npv': lesion_npv,
        'time_sensitivity': time_sensitivity,
        'time_specificity': time_specificity,
        'time_ppv': time_ppv,
        'time_npv': time_npv,
        'lesion_probs': lesion_probs,  
        'lesion_labels': lesion_labels,
        'time_probs': time_probs,
        'time_labels': time_labels
    }
    
    if log_file:
        with open(log_file, "a") as f:
            f.write(f"\nEvaluation Metrics:\n")
            f.write(f"Lesion Detection - Accuracy: {lesion_acc:.4f}, AUC: {lesion_auc:.4f}, F1: {lesion_f1:.4f}\n")
            f.write(f"Lesion Detection - Sensitivity: {lesion_sensitivity:.4f}, Specificity: {lesion_specificity:.4f}, PPV: {lesion_ppv:.4f}, NPV: {lesion_npv:.4f}\n")
            f.write(f"Time Classification - Accuracy: {time_acc:.4f}, AUC: {time_auc:.4f}, F1: {time_f1:.4f}\n")
            f.write(f"Time Classification - Sensitivity: {time_sensitivity:.4f}, Specificity: {time_specificity:.4f}, PPV: {time_ppv:.4f}, NPV: {time_npv:.4f}\n")
    
    return metrics