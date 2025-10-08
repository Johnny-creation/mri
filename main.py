import argparse
import torch
import traceback
from dataset import get_multitask_loaders
from model import MultiTaskModel
from train import train_multitask
from evaluate import evaluate_multitask
from utils import set_seed

def main():
    parser = argparse.ArgumentParser(description='Train MultiTaskModel for stroke classification')
    parser.add_argument('--base_path', type=str, default="../data/lesion-selected", help='Base path for dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=70, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0008, help='Learning rate')
    parser.add_argument('--dropout_rate', type=float, default=0.4, help='Dropout rate')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--time_loss_weight', type=float, default=1.0, help='Weight for time classification loss')
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        print("Loading data...")
        train_loader, val_loader, test_loader = get_multitask_loaders(args.base_path, args.batch_size)
        
        print("Initializing multi-task model...")
        model = MultiTaskModel(dropout_rate=args.dropout_rate).to(device)
        
        print("\nStarting multi-task training...")
        log_filename, final_model_path, best_model_path, val_metrics = train_multitask(
            model, train_loader, val_loader, args.epochs, args.lr, args.dropout_rate, args.weight_decay, args.time_loss_weight
        )
        
        print("\nEvaluating on test set...")
        model.load_state_dict(torch.load(final_model_path, map_location=device))
        test_metrics = evaluate_multitask(model, test_loader, device, log_filename)
        
        print(f"\nTest Results:")
        print(f"Lesion Detection - Accuracy: {test_metrics['lesion_acc']:.4f}, AUC: {test_metrics['lesion_auc']:.4f}, F1: {test_metrics['lesion_f1']:.4f}")
        print(f"Lesion Detection - Sensitivity: {test_metrics['lesion_sensitivity']:.4f}, Specificity: {test_metrics['lesion_specificity']:.4f}")
        print(f"Lesion Detection - PPV: {test_metrics['lesion_ppv']:.4f}, NPV: {test_metrics['lesion_npv']:.4f}")
        print(f"Time Classification - Accuracy: {test_metrics['time_acc']:.4f}, AUC: {test_metrics['time_auc']:.4f}, F1: {test_metrics['time_f1']:.4f}")
        print(f"Time Classification - Sensitivity: {test_metrics['time_sensitivity']:.4f}, Specificity: {test_metrics['time_specificity']:.4f}")
        print(f"Time Classification - PPV: {test_metrics['time_ppv']:.4f}, NPV: {test_metrics['time_npv']:.4f}")

        with open(log_filename, 'a') as f:
            f.write(f"\nTest Results:\n")
            f.write(f"Lesion Detection - Accuracy: {test_metrics['lesion_acc']:.4f}, AUC: {test_metrics['lesion_auc']:.4f}, F1: {test_metrics['lesion_f1']:.4f}\n")
            f.write(f"Lesion Detection - Sensitivity: {test_metrics['lesion_sensitivity']:.4f}, Specificity: {test_metrics['lesion_specificity']:.4f}\n")
            f.write(f"Lesion Detection - PPV: {test_metrics['lesion_ppv']:.4f}, NPV: {test_metrics['lesion_npv']:.4f}\n")
            f.write(f"Time Classification - Accuracy: {test_metrics['time_acc']:.4f}, AUC: {test_metrics['time_auc']:.4f}, F1: {test_metrics['time_f1']:.4f}\n")
            f.write(f"Time Classification - Sensitivity: {test_metrics['time_sensitivity']:.4f}, Specificity: {test_metrics['time_specificity']:.4f}\n")
            f.write(f"Time Classification - PPV: {test_metrics['time_ppv']:.4f}, NPV: {test_metrics['time_npv']:.4f}\n")
        

        params = {
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'lr': args.lr,
            'dropout_rate': args.dropout_rate,
            'weight_decay': args.weight_decay,
            'seed': args.seed,
            'time_loss_weight': args.time_loss_weight
        }
        return params, test_metrics, log_filename, best_model_path
        
    except Exception as e:
        print(f"Error in main program: {e}")
        traceback.print_exc()
        return None

if __name__ == '__main__':
    main()