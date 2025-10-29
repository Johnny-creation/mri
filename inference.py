import os
import argparse
import torch
import numpy as np
import pydicom
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from model import MultiTaskModel
from pathlib import Path

class StrokeInference:
    """
    Inference class for stroke lesion detection and time classification.
    Supports single slice or case-level (multiple slices) inference with visualization.
    """

    def __init__(self, model_path, device=None, use_snn_head=False, T=10, beta=0.9):
        """
        Initialize inference model.

        Args:
            model_path: Path to trained model checkpoint (.pt file)
            device: Device to run inference on ('cuda' or 'cpu')
            use_snn_head: Whether model uses spiking neural network head
            T: Number of timesteps for LIF neuron (if use_snn_head=True)
            beta: Leak factor for LIF neuron (if use_snn_head=True)
        """
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Load model
        self.model = MultiTaskModel(dropout_rate=0.0, use_snn_head=use_snn_head, T=T, beta=beta)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded from {model_path}")

    def preprocess_dicom(self, dwi_path, flair_path):
        """
        Preprocess DWI and FLAIR DICOM files.

        Args:
            dwi_path: Path to DWI DICOM file
            flair_path: Path to FLAIR DICOM file

        Returns:
            torch.Tensor: Preprocessed 2-channel image tensor (1, 2, 224, 224)
            tuple: Original images (dwi_array, flair_array) for visualization
        """
        # Read DICOM files
        dwi_ds = pydicom.dcmread(dwi_path)
        flair_ds = pydicom.dcmread(flair_path)

        dwi = dwi_ds.pixel_array.astype(np.float32)
        flair = flair_ds.pixel_array.astype(np.float32)

        # Handle multi-frame DICOM
        if len(dwi.shape) > 2:
            dwi = dwi[0]
        if len(flair.shape) > 2:
            flair = flair[0]

        # Store original for visualization
        dwi_original = dwi.copy()
        flair_original = flair.copy()

        # Normalize to [0, 1]
        dwi = (dwi - np.min(dwi)) / (np.ptp(dwi) + 1e-5)
        flair = (flair - np.min(flair)) / (np.ptp(flair) + 1e-5)

        # Convert to PIL and resize
        dwi_img = Image.fromarray((dwi * 255).astype(np.uint8)).convert('L')
        flair_img = Image.fromarray((flair * 255).astype(np.uint8)).convert('L')

        dwi_img = dwi_img.resize((224, 224), resample=Image.BILINEAR)
        flair_img = flair_img.resize((224, 224), resample=Image.BILINEAR)

        # Convert to tensor
        dwi_tensor = torch.from_numpy(np.array(dwi_img)).float() / 255.0
        flair_tensor = torch.from_numpy(np.array(flair_img)).float() / 255.0

        img_tensor = torch.stack([dwi_tensor, flair_tensor], dim=0).unsqueeze(0)  # (1, 2, 224, 224)

        return img_tensor, (dwi_original, flair_original)

    def predict(self, dwi_path, flair_path):
        """
        Perform inference on a single slice.

        Args:
            dwi_path: Path to DWI DICOM file
            flair_path: Path to FLAIR DICOM file

        Returns:
            dict: Prediction results with probabilities and class predictions
        """
        # Preprocess
        img_tensor, originals = self.preprocess_dicom(dwi_path, flair_path)
        img_tensor = img_tensor.to(self.device)

        # Inference
        with torch.no_grad():
            lesion_out, time_out = self.model(img_tensor)

            # Get probabilities
            lesion_probs = torch.softmax(lesion_out, dim=1).cpu().numpy()[0]
            time_probs = torch.softmax(time_out, dim=1).cpu().numpy()[0]

            # Get predictions
            lesion_pred = int(torch.argmax(lesion_out, dim=1).cpu().numpy()[0])
            time_pred = int(torch.argmax(time_out, dim=1).cpu().numpy()[0])

        results = {
            'lesion_pred': lesion_pred,
            'lesion_prob': float(lesion_probs[1]),  # Probability of lesion present
            'lesion_confidence': float(max(lesion_probs)),
            'time_pred': time_pred,
            'time_prob': float(time_probs[1]),  # Probability of late stage (>=270 min)
            'time_confidence': float(max(time_probs)),
            'originals': originals
        }

        return results

    def predict_case(self, case_dir):
        """
        Perform inference on all slices in a case directory.

        Args:
            case_dir: Path to case directory containing DWI/ and FLAIR/ subdirectories

        Returns:
            list: List of results for each slice
            dict: Aggregated case-level results
        """
        dwi_dir = os.path.join(case_dir, "DWI")
        flair_dir = os.path.join(case_dir, "FLAIR")

        if not (os.path.isdir(dwi_dir) and os.path.isdir(flair_dir)):
            raise ValueError(f"Case directory must contain DWI/ and FLAIR/ subdirectories")

        dwi_files = sorted([f for f in os.listdir(dwi_dir) if f.endswith('.dcm')])
        flair_files = sorted([f for f in os.listdir(flair_dir) if f.endswith('.dcm')])

        common_files = sorted(set(dwi_files).intersection(set(flair_files)))

        if not common_files:
            raise ValueError("No matching DWI/FLAIR slices found")

        print(f"Processing {len(common_files)} slices...")

        slice_results = []
        for fname in common_files:
            dwi_path = os.path.join(dwi_dir, fname)
            flair_path = os.path.join(flair_dir, fname)

            results = self.predict(dwi_path, flair_path)
            results['filename'] = fname
            slice_results.append(results)

        # Aggregate case-level results
        lesion_count = sum(1 for r in slice_results if r['lesion_pred'] == 1)
        avg_lesion_prob = np.mean([r['lesion_prob'] for r in slice_results])
        avg_time_prob = np.mean([r['time_prob'] for r in slice_results if r['lesion_pred'] == 1]) if lesion_count > 0 else 0.0

        case_results = {
            'total_slices': len(slice_results),
            'lesion_positive_slices': lesion_count,
            'lesion_ratio': lesion_count / len(slice_results),
            'avg_lesion_prob': float(avg_lesion_prob),
            'avg_time_prob': float(avg_time_prob),
            'case_has_lesion': lesion_count > 0,
            'estimated_time_stage': 'Late (≥270 min)' if avg_time_prob > 0.5 else 'Early (<270 min)'
        }

        return slice_results, case_results

    def visualize_slice(self, dwi_path, flair_path, results, save_path=None):
        """
        Visualize inference results for a single slice.

        Args:
            dwi_path: Path to DWI DICOM file
            flair_path: Path to FLAIR DICOM file
            results: Prediction results from predict()
            save_path: Optional path to save visualization
        """
        dwi_original, flair_original = results['originals']

        # Normalize for display
        dwi_display = (dwi_original - np.min(dwi_original)) / (np.ptp(dwi_original) + 1e-5)
        flair_display = (flair_original - np.min(flair_original)) / (np.ptp(flair_original) + 1e-5)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # DWI
        axes[0].imshow(dwi_display, cmap='gray')
        axes[0].set_title('DWI Sequence', fontsize=14, fontweight='bold')
        axes[0].axis('off')

        # FLAIR
        axes[1].imshow(flair_display, cmap='gray')
        axes[1].set_title('FLAIR Sequence', fontsize=14, fontweight='bold')
        axes[1].axis('off')

        # Add lesion detection result
        lesion_label = "Lesion Detected" if results['lesion_pred'] == 1 else "No Lesion"
        lesion_color = 'red' if results['lesion_pred'] == 1 else 'green'

        # Add colored border based on lesion detection
        for ax in axes:
            rect = patches.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                     linewidth=4, edgecolor=lesion_color, facecolor='none')
            ax.add_patch(rect)

        # Overall title with results
        title = f"Lesion Detection: {lesion_label} (Confidence: {results['lesion_confidence']:.2%})\n"
        if results['lesion_pred'] == 1:
            time_stage = "Late Stage (≥270 min)" if results['time_pred'] == 1 else "Early Stage (<270 min)"
            title += f"Time Classification: {time_stage} (Confidence: {results['time_confidence']:.2%})"

        fig.suptitle(title, fontsize=12, fontweight='bold', y=0.98)

        # Add text box with detailed probabilities
        textstr = f"Lesion Probability: {results['lesion_prob']:.2%}\n"
        if results['lesion_pred'] == 1:
            textstr += f"Late Stage Probability: {results['time_prob']:.2%}"

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        fig.text(0.5, 0.02, textstr, ha='center', fontsize=10, bbox=props)

        plt.tight_layout(rect=[0, 0.05, 1, 0.95])

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")

        plt.show()

    def visualize_case(self, slice_results, case_results, save_path=None):
        """
        Visualize case-level aggregated results.

        Args:
            slice_results: List of slice-level results
            case_results: Aggregated case-level results
            save_path: Optional path to save visualization
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Lesion detection per slice
        ax1 = axes[0, 0]
        slices = [r['filename'] for r in slice_results]
        lesion_probs = [r['lesion_prob'] for r in slice_results]
        colors = ['red' if r['lesion_pred'] == 1 else 'green' for r in slice_results]

        x_pos = np.arange(len(slices))
        ax1.bar(x_pos, lesion_probs, color=colors, alpha=0.7, edgecolor='black')
        ax1.axhline(y=0.5, color='black', linestyle='--', linewidth=1, label='Decision Threshold')
        ax1.set_xlabel('Slice Index', fontsize=11)
        ax1.set_ylabel('Lesion Probability', fontsize=11)
        ax1.set_title('Lesion Detection per Slice', fontsize=12, fontweight='bold')
        ax1.set_ylim([0, 1])
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)

        # 2. Time classification for lesion-positive slices
        ax2 = axes[0, 1]
        lesion_positive = [r for r in slice_results if r['lesion_pred'] == 1]
        if lesion_positive:
            time_probs = [r['time_prob'] for r in lesion_positive]
            slice_names = [r['filename'] for r in lesion_positive]
            colors_time = ['orange' if r['time_pred'] == 1 else 'blue' for r in lesion_positive]

            x_pos_time = np.arange(len(lesion_positive))
            ax2.bar(x_pos_time, time_probs, color=colors_time, alpha=0.7, edgecolor='black')
            ax2.axhline(y=0.5, color='black', linestyle='--', linewidth=1, label='Decision Threshold')
            ax2.set_xlabel('Lesion-Positive Slice Index', fontsize=11)
            ax2.set_ylabel('Late Stage Probability', fontsize=11)
            ax2.set_title('Time Classification (Lesion-Positive Slices)', fontsize=12, fontweight='bold')
            ax2.set_ylim([0, 1])
            ax2.legend()
            ax2.grid(axis='y', alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No lesion-positive slices',
                    ha='center', va='center', fontsize=14, transform=ax2.transAxes)
            ax2.set_title('Time Classification (Lesion-Positive Slices)', fontsize=12, fontweight='bold')

        # 3. Case summary (pie chart)
        ax3 = axes[1, 0]
        lesion_count = case_results['lesion_positive_slices']
        no_lesion_count = case_results['total_slices'] - lesion_count

        sizes = [lesion_count, no_lesion_count]
        labels = [f'Lesion Present\n({lesion_count} slices)',
                 f'No Lesion\n({no_lesion_count} slices)']
        colors_pie = ['#ff6b6b', '#51cf66']
        explode = (0.1, 0)

        ax3.pie(sizes, explode=explode, labels=labels, colors=colors_pie,
               autopct='%1.1f%%', shadow=True, startangle=90, textprops={'fontsize': 10})
        ax3.set_title('Case-Level Lesion Distribution', fontsize=12, fontweight='bold')

        # 4. Summary statistics
        ax4 = axes[1, 1]
        ax4.axis('off')

        summary_text = f"""
        CASE SUMMARY
        {'='*40}

        Total Slices: {case_results['total_slices']}
        Lesion-Positive Slices: {case_results['lesion_positive_slices']}
        Lesion Ratio: {case_results['lesion_ratio']:.1%}

        Average Lesion Probability: {case_results['avg_lesion_prob']:.2%}
        Average Late Stage Probability: {case_results['avg_time_prob']:.2%}

        Case Diagnosis:
        • Lesion Status: {'POSITIVE' if case_results['case_has_lesion'] else 'NEGATIVE'}
        • Estimated Time Stage: {case_results['estimated_time_stage']}

        {'='*40}
        """

        props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
        ax4.text(0.1, 0.5, summary_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='center', fontfamily='monospace', bbox=props)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Case visualization saved to {save_path}")

        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Stroke Lesion Inference with Visualization')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint (.pt file)')
    parser.add_argument('--mode', type=str, choices=['slice', 'case'], required=True,
                       help='Inference mode: "slice" for single slice, "case" for entire case')
    parser.add_argument('--dwi', type=str, help='Path to DWI DICOM file (for slice mode)')
    parser.add_argument('--flair', type=str, help='Path to FLAIR DICOM file (for slice mode)')
    parser.add_argument('--case_dir', type=str, help='Path to case directory (for case mode)')
    parser.add_argument('--output_dir', type=str, default='./inference_output',
                       help='Directory to save visualization results')
    parser.add_argument('--use_snn_head', action='store_true',
                       help='Enable if model uses spiking neural network head')
    parser.add_argument('--T', type=int, default=10, help='Number of timesteps for LIF neuron')
    parser.add_argument('--beta', type=float, default=0.9, help='Leak factor for LIF neuron')
    parser.add_argument('--no_viz', action='store_true', help='Disable visualization')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize inference
    inferencer = StrokeInference(
        model_path=args.model_path,
        use_snn_head=args.use_snn_head,
        T=args.T,
        beta=args.beta
    )

    if args.mode == 'slice':
        # Single slice inference
        if not args.dwi or not args.flair:
            raise ValueError("For slice mode, --dwi and --flair paths are required")

        print(f"\nProcessing slice:")
        print(f"  DWI: {args.dwi}")
        print(f"  FLAIR: {args.flair}")

        results = inferencer.predict(args.dwi, args.flair)

        # Print results
        print("\n" + "="*50)
        print("INFERENCE RESULTS")
        print("="*50)
        print(f"Lesion Detection: {'POSITIVE' if results['lesion_pred'] == 1 else 'NEGATIVE'}")
        print(f"  Probability: {results['lesion_prob']:.2%}")
        print(f"  Confidence: {results['lesion_confidence']:.2%}")

        if results['lesion_pred'] == 1:
            time_stage = "Late Stage (≥270 min)" if results['time_pred'] == 1 else "Early Stage (<270 min)"
            print(f"\nTime Classification: {time_stage}")
            print(f"  Late Stage Probability: {results['time_prob']:.2%}")
            print(f"  Confidence: {results['time_confidence']:.2%}")
        print("="*50 + "\n")

        # Visualize
        if not args.no_viz:
            save_path = os.path.join(args.output_dir, 'slice_inference.png')
            inferencer.visualize_slice(args.dwi, args.flair, results, save_path=save_path)

    elif args.mode == 'case':
        # Case-level inference
        if not args.case_dir:
            raise ValueError("For case mode, --case_dir path is required")

        print(f"\nProcessing case: {args.case_dir}")

        slice_results, case_results = inferencer.predict_case(args.case_dir)

        # Print results
        print("\n" + "="*50)
        print("CASE-LEVEL RESULTS")
        print("="*50)
        print(f"Total Slices: {case_results['total_slices']}")
        print(f"Lesion-Positive Slices: {case_results['lesion_positive_slices']} ({case_results['lesion_ratio']:.1%})")
        print(f"Average Lesion Probability: {case_results['avg_lesion_prob']:.2%}")
        print(f"Average Late Stage Probability: {case_results['avg_time_prob']:.2%}")
        print(f"\nCase Diagnosis: {'LESION DETECTED' if case_results['case_has_lesion'] else 'NO LESION'}")
        print(f"Estimated Time Stage: {case_results['estimated_time_stage']}")
        print("="*50)

        # Print slice-by-slice summary
        print("\nSlice-by-Slice Summary:")
        print("-" * 70)
        print(f"{'Slice':<30} {'Lesion':<15} {'Prob':<10} {'Time Stage':<15}")
        print("-" * 70)
        for r in slice_results:
            lesion_status = "POSITIVE" if r['lesion_pred'] == 1 else "NEGATIVE"
            time_stage = ("Late" if r['time_pred'] == 1 else "Early") if r['lesion_pred'] == 1 else "N/A"
            print(f"{r['filename']:<30} {lesion_status:<15} {r['lesion_prob']:<10.2%} {time_stage:<15}")
        print("-" * 70 + "\n")

        # Visualize
        if not args.no_viz:
            save_path = os.path.join(args.output_dir, 'case_inference.png')
            inferencer.visualize_case(slice_results, case_results, save_path=save_path)


if __name__ == '__main__':
    main()
