"""
Image Preprocessing and Augmentation Pipeline for Facial Expression Recognition.
"""
from typing import Tuple, Union, Optional
import numpy as np
from PIL import Image, ImageEnhance, ImageOps, ImageChops
import matplotlib.pyplot as plt

from .config import IMAGE_SIZE, GRAPHS_DIR


def normalize_image(image: np.ndarray, method: str = "minmax") -> np.ndarray:
    """
    Normalizes pixel values of an image array.

    Args:
        image: Numpy array of image pixels.
        method: 'minmax' ([0, 1]), 'centered' ([-1, 1]), or 'standard' (z-score).

    Returns:
        Normalized numpy float32 array.
    """
    img_float = image.astype(np.float32)

    if method == "minmax":
        return img_float / 255.0
    elif method == "centered":
        return (img_float / 127.5) - 1.0
    elif method == "standard":
        mean = np.mean(img_float)
        std = np.std(img_float)
        return (img_float - mean) / (std + 1e-7)
    else:
        raise ValueError(f"Unknown normalization method: {method}. Choose from 'minmax', 'centered', 'standard'.")


def denormalize_image(image: np.ndarray, method: str = "minmax") -> np.ndarray:
    """
    Reverses normalization back to [0, 255] uint8 array.
    """
    if method == "minmax":
        img = image * 255.0
    elif method == "centered":
        img = (image + 1.0) * 127.5
    elif method == "standard":
        # Approximate standard denormalization
        img = (image - image.min()) / (image.max() - image.min() + 1e-7) * 255.0
    else:
        img = image

    return np.clip(img, 0, 255).astype(np.uint8)


def enhance_contrast(image: Union[Image.Image, np.ndarray], method: str = "hist_eq") -> Union[Image.Image, np.ndarray]:
    """
    Enhances contrast for facial feature clarity under uneven lighting.

    Args:
        image: PIL Image or numpy array.
        method: 'hist_eq' (Histogram Equalization) or 'autocontrast'.

    Returns:
        Enhanced image of same type as input.
    """
    is_numpy = isinstance(image, np.ndarray)
    if is_numpy:
        pil_img = Image.fromarray(image.squeeze().astype(np.uint8))
    else:
        pil_img = image

    if method == "hist_eq":
        enhanced = ImageOps.equalize(pil_img)
    elif method == "autocontrast":
        enhanced = ImageOps.autocontrast(pil_img, cutoff=2)
    else:
        enhanced = pil_img

    if is_numpy:
        arr = np.array(enhanced)
        if image.ndim == 3 and image.shape[-1] == 1:
            arr = np.expand_dims(arr, axis=-1)
        return arr
    return enhanced


def load_and_preprocess_image(
    image_path: str,
    target_size: Tuple[int, int] = IMAGE_SIZE,
    grayscale: bool = True,
    normalize: bool = True,
    norm_method: str = "minmax",
    enhance: bool = False,
) -> np.ndarray:
    """
    Loads an image from disk and applies preprocessing.

    Args:
        image_path: Path to the image file.
        target_size: (width, height) tuple for resizing.
        grayscale: If True, convert to 1-channel grayscale; else 3-channel RGB.
        normalize: If True, normalize pixel values to [0, 1].
        norm_method: Normalization strategy ('minmax', 'centered', 'standard').
        enhance: If True, applies histogram equalization.

    Returns:
        Preprocessed image as a numpy array with shape (H, W, C).
    """
    mode = "L" if grayscale else "RGB"
    with Image.open(image_path) as img:
        img = img.convert(mode)

        if img.size != target_size:
            img = img.resize(target_size, Image.Resampling.BILINEAR)

        if enhance:
            img = enhance_contrast(img, method="hist_eq")

        arr = np.array(img, dtype=np.float32)

    if grayscale:
        # Add channel dimension: (H, W) -> (H, W, 1)
        arr = np.expand_dims(arr, axis=-1)

    if normalize:
        arr = normalize_image(arr, method=norm_method)

    return arr


class DataAugmentation:
    """
    Data Augmentation pipeline specifically tailored for facial expression images.
    Applies subtle geometric and photometric variations without distorting key expression features.
    """

    def __init__(
        self,
        horizontal_flip_prob: float = 0.5,
        rotation_angle: float = 12.0,
        rotation_prob: float = 0.5,
        shift_range: float = 0.08,
        shift_prob: float = 0.5,
        zoom_range: Tuple[float, float] = (0.92, 1.08),
        zoom_prob: float = 0.4,
        brightness_range: Tuple[float, float] = (0.85, 1.15),
        brightness_prob: float = 0.4,
        contrast_range: Tuple[float, float] = (0.85, 1.15),
        contrast_prob: float = 0.4,
        random_seed: Optional[int] = None,
    ):
        self.horizontal_flip_prob = horizontal_flip_prob
        self.rotation_angle = rotation_angle
        self.rotation_prob = rotation_prob
        self.shift_range = shift_range
        self.shift_prob = shift_prob
        self.zoom_range = zoom_range
        self.zoom_prob = zoom_prob
        self.brightness_range = brightness_range
        self.brightness_prob = brightness_prob
        self.contrast_range = contrast_range
        self.contrast_prob = contrast_prob

        self.rng = np.random.default_rng(random_seed)

    def apply_to_pil(self, img: Image.Image) -> Image.Image:
        """Applies augmentation to a PIL Image."""
        # 1. Random Horizontal Flip (Expression is largely symmetric)
        if self.rng.random() < self.horizontal_flip_prob:
            img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        # 2. Random Rotation (Head tilt)
        if self.rng.random() < self.rotation_prob:
            angle = self.rng.uniform(-self.rotation_angle, self.rotation_angle)
            img = img.rotate(angle, resample=Image.Resampling.BILINEAR)

        # 3. Random Translation / Shift
        if self.rng.random() < self.shift_prob:
            w, h = img.size
            dx = int(self.rng.uniform(-self.shift_range, self.shift_range) * w)
            dy = int(self.rng.uniform(-self.shift_range, self.shift_range) * h)
            img = ImageChops.offset(img, dx, dy)

        # 4. Random Zoom (Crop & Resize)
        if self.rng.random() < self.zoom_prob:
            w, h = img.size
            factor = self.rng.uniform(self.zoom_range[0], self.zoom_range[1])
            new_w, new_h = int(w * factor), int(h * factor)
            resized = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
            if factor > 1.0:
                # Crop center back to original size
                left = (new_w - w) // 2
                top = (new_h - h) // 2
                img = resized.crop((left, top, left + w, top + h))
            else:
                # Pad to original size
                new_img = Image.new(img.mode, (w, h), color=0)
                paste_x = (w - new_w) // 2
                paste_y = (h - new_h) // 2
                new_img.paste(resized, (paste_x, paste_y))
                img = new_img

        # 5. Brightness adjustment
        if self.rng.random() < self.brightness_prob:
            factor = self.rng.uniform(self.brightness_range[0], self.brightness_range[1])
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(factor)

        # 6. Contrast adjustment
        if self.rng.random() < self.contrast_prob:
            factor = self.rng.uniform(self.contrast_range[0], self.contrast_range[1])
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(factor)

        return img

    def apply(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """
        Applies augmentation pipeline to either a numpy array or PIL Image.
        Returns a float32 numpy array with identical normalization.
        """
        if isinstance(image, Image.Image):
            pil_img = image
            is_normalized = False
        else:
            is_normalized = (image.max() <= 1.0 and image.min() >= -1.0)
            if is_normalized:
                raw_pixels = (image * 255.0).clip(0, 255).astype(np.uint8)
            else:
                raw_pixels = image.clip(0, 255).astype(np.uint8)

            if raw_pixels.ndim == 3 and raw_pixels.shape[-1] == 1:
                raw_pixels = raw_pixels.squeeze(-1)
                mode = "L"
            elif raw_pixels.ndim == 2:
                mode = "L"
            else:
                mode = "RGB"

            pil_img = Image.fromarray(raw_pixels, mode=mode)

        augmented_pil = self.apply_to_pil(pil_img)
        arr = np.array(augmented_pil, dtype=np.float32)

        if arr.ndim == 2:
            arr = np.expand_dims(arr, axis=-1)

        if is_normalized:
            arr = arr / 255.0

        return arr


def visualize_augmentations(
    image_path: str,
    num_samples: int = 7,
    save_path: Optional[str] = None,
) -> str:
    """
    Visualizes original image alongside augmented variations and saves the plot.

    Args:
        image_path: Path to sample face image.
        num_samples: Number of augmented samples to generate.
        save_path: Destination path for graph plot. Defaults to results/graphs/augmentation_samples.png.

    Returns:
        Path to saved figure.
    """
    if save_path is None:
        save_path = str(GRAPHS_DIR / "augmentation_samples.png")

    aug = DataAugmentation()
    with Image.open(image_path) as raw_img:
        orig = raw_img.convert("L")

    fig, axes = plt.subplots(1, num_samples + 1, figsize=(2.2 * (num_samples + 1), 3))

    # Original
    axes[0].imshow(orig, cmap="gray")
    axes[0].set_title("Original", fontsize=11, fontweight="bold", color="darkblue")
    axes[0].axis("off")

    # Augmented samples
    for i in range(1, num_samples + 1):
        aug_img = aug.apply_to_pil(orig)
        axes[i].imshow(aug_img, cmap="gray")
        axes[i].set_title(f"Augmented #{i}", fontsize=10)
        axes[i].axis("off")

    plt.suptitle("Facial Expression Recognition - Data Augmentation Pipeline", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved augmentation visualization to: {save_path}")
    return save_path
