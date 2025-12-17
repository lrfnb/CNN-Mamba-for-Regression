import os
import sys
import glob
import random
import numpy as np
from PIL import Image
import torch
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.resolution_regression_net import ResolutionRegressionNet


class ResolutionTester:
    def __init__(self, model_path, test_dir, crop_size=128, base_resolution=0.1):
        self.model_path = model_path
        self.test_dir = test_dir
        self.crop_size = crop_size
        self.base_resolution = base_resolution
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.high_label_range = (0.01, 0.1)
        self.medium_label_range = (0.1, 0.4)
        self.low_label_range = (0.4, 1.0)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5))
        ])

        self.model = self.load_model()

    def load_model(self):
        model = ResolutionRegressionNet()
        checkpoint = torch.load(self.model_path, map_location=self.device)

        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        new_state = {}
        for k, v in state_dict.items():
            new_state[k.replace("module.", "")] = v

        model.load_state_dict(new_state, strict=False)
        model.to(self.device).eval()
        return model

    def get_test_images(self):
        exts = ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"]
        paths = []
        for ext in exts:
            paths += glob.glob(os.path.join(self.test_dir, ext))
        return paths

    def crop_random_patch(self, img):
        w, h = img.size
        if w < self.crop_size or h < self.crop_size:
            return img.resize((self.crop_size, self.crop_size), Image.BICUBIC)

        x = random.randint(0, w - self.crop_size)
        y = random.randint(0, h - self.crop_size)
        return img.crop((x, y, x + self.crop_size, y + self.crop_size))

    def sample_labels_uniform(self):
        high_label = np.random.uniform(*self.high_label_range)
        medium_label = np.random.uniform(*self.medium_label_range)
        low_label = np.random.uniform(*self.low_label_range)

        high_factor = self.base_resolution / high_label
        medium_factor = self.base_resolution / medium_label
        low_factor = self.base_resolution / low_label

        return (
            (high_label, high_factor, "up"),
            (medium_label, medium_factor, "down"),
            (low_label, low_factor, "down")
        )

    def process_image_with_sampling(self, img, factor, mode):
        w, h = img.size
        if mode == "up":
            new_size = (int(w * factor), int(h * factor))
        else:
            new_size = (max(1, int(w / factor)), max(1, int(h / factor)))

        img_scaled = img.resize(new_size, Image.BICUBIC)
        return img_scaled.resize((w, h), Image.BICUBIC)

    def process_single_image(self, img_path):
        img = Image.open(img_path).convert("RGB")
        img = self.crop_random_patch(img)

        samples = self.sample_labels_uniform()

        imgs, labels, factors = [], [], []

        for label, factor, mode in samples:
            out = self.process_image_with_sampling(img, factor, mode)
            imgs.append(self.transform(out))
            labels.append(label)
            factors.append(factor)

        return {
            "images": torch.cat(imgs, dim=0),
            "labels": torch.tensor(labels, dtype=torch.float32),
            "factors": factors,
            "success": True,
            "img_path": img_path
        }

    def calculate_metrics(self, y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        return dict(MSE=mse, RMSE=rmse, MAE=mae, R2=r2, MAPE=mape)

    def visualize_gt_pred_curves(self, results, save_dir=None):

        detailed = results["details"]

        gt = np.array(
            [r["gt_high"] for r in detailed] +
            [r["gt_medium"] for r in detailed] +
            [r["gt_low"] for r in detailed]
        )

        pred = np.array(
            [r["pred_high"] for r in detailed] +
            [r["pred_medium"] for r in detailed] +
            [r["pred_low"] for r in detailed]
        )

        # 三个区间分别取
        ranges = {
            "High (0.01–0.1 m/px)": (0.01, 0.1),
            "Medium (0.1–0.4 m/px)": (0.1, 0.4),
            "Low (0.4–1.0 m/px)": (0.4, 1.0),
        }

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

        for name, (low, high) in ranges.items():
            mask = (gt >= low) & (gt < high)
            gt_sub = gt[mask]
            pred_sub = pred[mask]

            order = np.argsort(gt_sub)
            gt_sorted = gt_sub[order]
            pred_sorted = pred_sub[order]

            plt.figure(figsize=(10, 4))
            plt.plot(gt_sorted, color="red", linewidth=2, label="Ground Truth")
            plt.plot(pred_sorted, color="tab:blue", linewidth=2, label="Prediction")

            plt.title(f"GT vs Prediction — {name}")
            plt.xlabel("Sample Index (sorted by GT)")
            plt.ylabel("Resolution (m/px)")
            plt.grid(True, alpha=0.3)
            plt.legend()

            if save_dir is not None:
                path = os.path.join(
                    save_dir,
                    f"curve_{name.split()[0].lower()}.png"
                )
                plt.savefig(path, dpi=300, bbox_inches="tight")
                print(f"[OK] Saved: {path}")

            plt.show()

        order_all = np.argsort(gt)
        gt_all_sorted = gt[order_all]
        pred_all_sorted = pred[order_all]

        plt.figure(figsize=(12, 4))
        plt.plot(gt_all_sorted, color="red", linewidth=2, label="Ground Truth")
        plt.plot(pred_all_sorted, color="tab:green", linewidth=2, label="Prediction")

        plt.title("GT vs Prediction — Full Range (0.01–1.0 m/px)")
        plt.xlabel("Sample Index (sorted by GT)")
        plt.ylabel("Resolution (m/px)")
        plt.grid(True, alpha=0.3)
        plt.legend()

        if save_dir is not None:
            path = os.path.join(save_dir, "curve_full_range.png")
            plt.savefig(path, dpi=300, bbox_inches="tight")
            print(f"[OK] Saved: {path}")

        plt.show()

    def run_test(self, num_samples=None):
        paths = self.get_test_images()
        if num_samples:
            paths = random.sample(paths, num_samples)

        preds, gts, details = [], [], []

        with torch.no_grad():
            for p in paths:
                data = self.process_single_image(p)
                img = data["images"].unsqueeze(0).to(self.device)
                pred = self.model(img).cpu().squeeze(0)
                gt = data["labels"]

                preds.extend(pred.numpy())
                gts.extend(gt.numpy())

                details.append({
                    "image": os.path.basename(p),
                    "gt_high": gt[0].item(),
                    "gt_medium": gt[1].item(),
                    "gt_low": gt[2].item(),
                    "pred_high": pred[0].item(),
                    "pred_medium": pred[1].item(),
                    "pred_low": pred[2].item(),
                })

        return {
            "overall_metrics": self.calculate_metrics(gts, preds),
            "details": details,
            "predictions": np.array(preds),
            "ground_truths": np.array(gts)
        }


def main():
    tester = ResolutionTester(
        model_path="/data2/lrf/CMCNet/checkpoints/ResolutionRegression/best_loss_net_R.pth",
        test_dir="/data2/lrf/data/spatial-resolution_0.1m/test_mix",
        crop_size=128,
        base_resolution=0.1
    )

    results = tester.run_test(num_samples=1596)
    tester.visualize_gt_pred_curves(
        results,
        save_dir="./resolution_regression_test_results/curves"
    )

    print(results["overall_metrics"])


if __name__ == "__main__":
    main()
