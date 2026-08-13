import numpy as np
import cv2


def apply_sensor_noise(img, poisson_scale=18.0, gaussian_sigma=6.0, rng=None):
    rng = rng or np.random.default_rng()
    img_f = img.astype(np.float64)

    shot = rng.poisson(img_f / poisson_scale * 255.0 / 255.0 + 1e-6) * poisson_scale
    shot = shot - img_f
    read_noise = rng.normal(0, gaussian_sigma, img.shape)

    noisy = img_f + 0.35 * shot + read_noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def apply_edge_brightening(img, edge_gain=0.55, blur_ksize=3):
    img_f = img.astype(np.float32)
    gx = cv2.Sobel(img_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_f, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    magnitude = cv2.GaussianBlur(magnitude, (blur_ksize, blur_ksize), 0)
    peak = magnitude.max()
    if peak > 1e-6:
        magnitude = magnitude / peak * 255.0
    boosted = img_f + edge_gain * magnitude
    return np.clip(boosted, 0, 255).astype(np.uint8)


def apply_blur(img, sigma=0.8):
    if sigma <= 0:
        return img
    k = max(3, int(sigma * 4) | 1)
    return cv2.GaussianBlur(img, (k, k), sigma)


def apply_rotation(img, angle_deg, fill_value=None):
    if angle_deg == 0:
        return img
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REFLECT101)


def apply_scale_jitter(img, scale):
    h, w = img.shape[:2]
    nh, nw = max(4, int(h * scale)), max(4, int(w * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(resized, (w, h), interpolation=cv2.INTER_LINEAR)


def degrade_pipeline(img, rng, blur_sigma, rotation_deg, scale_jitter,
                      poisson_scale, gaussian_sigma, edge_gain):
    out = img.copy()
    if scale_jitter != 1.0:
        out = apply_scale_jitter(out, scale_jitter)
    out = apply_rotation(out, rotation_deg)
    out = apply_blur(out, blur_sigma)
    out = apply_edge_brightening(out, edge_gain)
    out = apply_sensor_noise(out, poisson_scale, gaussian_sigma, rng)
    return out
