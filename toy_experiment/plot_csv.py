from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_ssim(df, output_path):
    x = df["img_num"]

    plt.figure(figsize=(12, 6))
    plt.plot(x, df["High_SSIM"], color="lightblue", label="High SSIM")
    plt.plot(x, df["Low_SSIM"], color="blue", label="Low SSIM")
    plt.axhline(y=0.92, color="black", linestyle="--", label="SSIM = 0.92")

    plt.xlabel("img_num")
    plt.ylabel("SSIM")
    plt.title("High/Low Frequency SSIM")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_psnr(df, output_path):
    x = df["img_num"]

    plt.figure(figsize=(12, 6))
    plt.plot(x, df["High_PSNR"], color="lightcoral", label="High PSNR")
    plt.plot(x, df["Low_PSNR"], color="red", label="Low PSNR")
    plt.axhline(y=32, color="black", linestyle="--", label="PSNR = 32")

    plt.xlabel("img_num")
    plt.ylabel("PSNR")
    plt.title("High/Low Frequency PSNR")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    current_dir = Path(__file__).resolve().parent
    csv_dir = current_dir / "csv_results"

    for csv_path in sorted(csv_dir.glob("*_frequency_metrics.csv")):
        df = pd.read_csv(csv_path)
        method_name = csv_path.stem.replace("_frequency_metrics", "")

        ssim_output = csv_dir / f"{method_name}_ssim.png"
        psnr_output = csv_dir / f"{method_name}_psnr.png"

        plot_ssim(df, ssim_output)
        plot_psnr(df, psnr_output)

        print(f"Saved plot -> {ssim_output}")
        print(f"Saved plot -> {psnr_output}")

if __name__ == "__main__":
    main()
