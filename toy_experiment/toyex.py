import csv
from pathlib import Path

import cv2
import numpy as np
import hydra
from omegaconf import DictConfig
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

class workspace:
    def __init__(self, cfg):
        self.cfg = cfg
    
    def run(self):
        '''
        Distinguish Image to High, low Frequency
        Calculate each image's PSNR and SSIM
        '''
        for method in self.cfg.frequency_methods:
            rows = []

            for i in tqdm(range(self.cfg.num_images), desc=f"Processing {method}"):
                # upload blured image and sharp image
                blured_image = cv2.imread(self.get_blurred_image(i))
                sharp_image = cv2.imread(self.get_sharp_image(i))

                if blured_image is None or sharp_image is None:
                    print(f"Skip {i:06d}: image not found")
                    continue

                # Devide the image into high and low frequency
                HF_blured_image, LF_blured_image = self.divide_frequency(blured_image, method)
                HF_sharp_image, LF_sharp_image = self.divide_frequency(sharp_image, method)

                # Calculate PSNR and SSIM
                HF_SSIM = self.cal_SSIM(HF_blured_image, HF_sharp_image)
                HF_PSNR = self.cal_PSNR(HF_blured_image, HF_sharp_image)

                LF_SSIM = self.cal_SSIM(LF_blured_image, LF_sharp_image)
                LF_PSNR = self.cal_PSNR(LF_blured_image, LF_sharp_image)

                rows.append([i, HF_SSIM, HF_PSNR, LF_SSIM, LF_PSNR])

            self.save_csv(rows, method)

    def get_blurred_image(self, index):
        print("Getting Blurred Image...")
        return self.cfg.blured_img_dir + f"/{index:06d}.png"

    def get_sharp_image(self, index):
        print("Getting Sharp Image...")
        return self.cfg.sharp_img_dir + f"/{index:06d}.png"

    def divide_frequency(self, image, method):
        image = image.astype(np.float32)

        if method == "gaussian":
            low = cv2.GaussianBlur(image, (0, 0), self.cfg.gaussian_sigma)
            high = image - low
        elif method == "box":
            kernel_size = int(self.cfg.box_kernel_size)
            low = cv2.blur(image, (kernel_size, kernel_size))
            high = image - low
        elif method == "median":
            kernel_size = int(self.cfg.median_kernel_size)
            low = cv2.medianBlur(image.astype(np.uint8), kernel_size).astype(np.float32)
            high = image - low
        elif method == "laplacian":
            gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
            high_gray = cv2.Laplacian(gray, cv2.CV_32F)
            high = np.stack([high_gray, high_gray, high_gray], axis=2)
            low = image - high
        elif method == "fft":
            high, low = self.divide_frequency_fft(image)
        else:
            raise ValueError(f"Unsupported frequency method: {method}")

        return high, low

    def divide_frequency_fft(self, image):
        low_channels = []
        high_channels = []

        for c in range(image.shape[2]):
            channel = image[:, :, c]
            f = np.fft.fft2(channel)
            fshift = np.fft.fftshift(f)

            rows, cols = channel.shape
            crow, ccol = rows // 2, cols // 2
            radius = max(1, int(min(rows, cols) * self.cfg.fft_cutoff_ratio))

            mask = np.zeros((rows, cols), dtype=np.float32)
            mask[max(0, crow - radius):min(rows, crow + radius),
                 max(0, ccol - radius):min(cols, ccol + radius)] = 1.0

            low_f = fshift * mask
            high_f = fshift * (1.0 - mask)

            low = np.fft.ifft2(np.fft.ifftshift(low_f)).real.astype(np.float32)
            high = np.fft.ifft2(np.fft.ifftshift(high_f)).real.astype(np.float32)

            low_channels.append(low)
            high_channels.append(high)

        low = np.stack(low_channels, axis=2)
        high = np.stack(high_channels, axis=2)
        return high, low

    def cal_SSIM(self, blured_image, sharp_image):
        min_value = min(float(blured_image.min()), float(sharp_image.min()))
        max_value = max(float(blured_image.max()), float(sharp_image.max()))
        data_range = max(max_value - min_value, 1e-8)
        return float(ssim(blured_image, sharp_image, channel_axis=2, data_range=data_range))

    def cal_PSNR(self, blured_image, sharp_image):
        mse = np.mean((blured_image.astype(np.float32) - sharp_image.astype(np.float32)) ** 2)
        if mse == 0:
            return float("inf")
        max_value = max(
            float(np.max(np.abs(blured_image))),
            float(np.max(np.abs(sharp_image))),
            1.0,
        )
        return float(20 * np.log10(max_value / np.sqrt(mse)))

    def save_csv(self, rows, method):
        csv_dir = Path(self.cfg.csv_dir)
        csv_dir.mkdir(parents=True, exist_ok=True)
        csv_path = csv_dir / f"{method}_frequency_metrics.csv"

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["img_num", "High_SSIM", "High_PSNR", "Low_SSIM", "Low_PSNR"])
            writer.writerows(rows)


@hydra.main(version_base=None, config_path=".", config_name="toyex")
def main(cfg: DictConfig) -> None:
    work = workspace(cfg)
    work.run()


if __name__ == "__main__":
    main()
