import numpy as np
import cv2


def _leaky_relu(x, slope=0.05):
    return np.where(x > 0, x, slope * x)


def _strided_conv_image(img_f32, W, b, stride):
    out_ch, in_ch = W.shape[0], W.shape[1]
    h, w = img_f32.shape[-2:]
    oh, ow = h // stride, w // stride
    out = np.zeros((out_ch, oh, ow), dtype=np.float32)
    for oc in range(out_ch):
        acc = np.zeros((h, w), dtype=np.float32)
        for ic in range(in_ch):
            kernel = W[oc, ic]
            src = img_f32[ic] if img_f32.ndim == 3 else img_f32
            acc += cv2.filter2D(src, -1, kernel, borderType=cv2.BORDER_REPLICATE)
        acc += b[oc]
        out[oc] = acc[:oh * stride:stride, :ow * stride:stride]
    return out


def extract_feature_map(gray_img, net):
    img = gray_img.astype(np.float32) / 255.0
    f1 = _strided_conv_image(img, net.conv1.W, net.conv1.b, net.conv1.stride)
    f1 = _leaky_relu(f1)
    f2 = _strided_conv_image(f1, net.conv2.W, net.conv2.b, net.conv2.stride)
    f2 = _leaky_relu(f2)
    return f2.transpose(1, 2, 0)
