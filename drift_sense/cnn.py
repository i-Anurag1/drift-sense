import numpy as np


def _im2col(x, kh, kw, stride):
    n, c, h, w = x.shape
    oh = (h - kh) // stride + 1
    ow = (w - kw) // stride + 1
    cols = np.zeros((n, c, kh, kw, oh, ow), dtype=x.dtype)
    for i in range(kh):
        i_max = i + stride * oh
        for j in range(kw):
            j_max = j + stride * ow
            cols[:, :, i, j, :, :] = x[:, :, i:i_max:stride, j:j_max:stride]
    cols = cols.transpose(0, 4, 5, 1, 2, 3).reshape(n * oh * ow, -1)
    return cols, oh, ow


def _col2im(cols, x_shape, kh, kw, stride):
    n, c, h, w = x_shape
    oh = (h - kh) // stride + 1
    ow = (w - kw) // stride + 1
    cols_reshaped = cols.reshape(n, oh, ow, c, kh, kw).transpose(0, 3, 4, 5, 1, 2)
    dx = np.zeros((n, c, h, w), dtype=cols.dtype)
    for i in range(kh):
        i_max = i + stride * oh
        for j in range(kw):
            j_max = j + stride * ow
            dx[:, :, i:i_max:stride, j:j_max:stride] += cols_reshaped[:, :, i, j, :, :]
    return dx


class Conv2D:
    def __init__(self, in_ch, out_ch, k, stride, rng):
        scale = np.sqrt(2.0 / (in_ch * k * k))
        self.W = rng.normal(0, scale, (out_ch, in_ch, k, k)).astype(np.float32)
        self.b = np.zeros(out_ch, dtype=np.float32)
        self.k, self.stride = k, stride
        self.cache = None

    def forward(self, x):
        n, c, h, w = x.shape
        cols, oh, ow = _im2col(x, self.k, self.k, self.stride)
        W_flat = self.W.reshape(self.W.shape[0], -1)
        out = cols @ W_flat.T + self.b
        out = out.reshape(n, oh, ow, -1).transpose(0, 3, 1, 2)
        self.cache = (x.shape, cols)
        return out

    def backward(self, dout, lr=None, accumulate=False):
        x_shape, cols = self.cache
        n, out_ch, oh, ow = dout.shape
        dout_flat = dout.transpose(0, 2, 3, 1).reshape(-1, out_ch)
        W_flat = self.W.reshape(out_ch, -1)

        dW = (dout_flat.T @ cols).reshape(self.W.shape)
        db = dout_flat.sum(axis=0)
        dcols = dout_flat @ W_flat
        dx = _col2im(dcols, x_shape, self.k, self.k, self.stride)

        if accumulate:
            self._dW, self._db = dW, db
            return dx

        self.W -= lr * dW
        self.b -= lr * db
        return dx

    def apply_grad(self, lr):
        self.W -= lr * self._dW
        self.b -= lr * self._db

    def state(self):
        return {"W": self.W, "b": self.b, "k": self.k, "stride": self.stride}

    def load(self, d):
        self.W, self.b = d["W"], d["b"]
        self.k, self.stride = int(d["k"]), int(d["stride"])


def leaky_relu_forward(x, slope=0.05):
    return np.where(x > 0, x, slope * x)


def leaky_relu_backward(dout, x, slope=0.05):
    return dout * np.where(x > 0, 1.0, slope)


class DriftSenseCNN:
    """Tiny 2-layer conv embedding network, trained from scratch with NumPy
    (no external DL framework — none is installable in this offline sandbox).
    Produces a dense multi-channel feature map used for cross-correlation
    matching in feature space, analogous in spirit to a fully-convolutional
    Siamese embedding (Bertinetto et al., 2016)."""

    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        self.conv1 = Conv2D(1, 8, 5, 2, rng)
        self.conv2 = Conv2D(8, 16, 5, 2, rng)
        self._cache = {}

    def forward(self, x, train=False):
        a1 = self.conv1.forward(x)
        r1 = leaky_relu_forward(a1)
        a2 = self.conv2.forward(r1)
        r2 = leaky_relu_forward(a2)
        if train:
            self._cache = {"a1": a1, "r1": r1, "a2": a2, "r2": r2}
        return r2

    def embed(self, x):
        feat = self.forward(x, train=False)
        n = feat.shape[0]
        pooled = feat.reshape(n, feat.shape[1], -1).mean(axis=2)
        norm = np.linalg.norm(pooled, axis=1, keepdims=True) + 1e-8
        return pooled / norm

    def backward_from_pooled_grad(self, dpooled, lr, clip_norm=2.0):
        gnorm = np.linalg.norm(dpooled)
        if gnorm > clip_norm:
            dpooled = dpooled * (clip_norm / (gnorm + 1e-8))
        r2 = self._cache["r2"]
        n, c, h, w = r2.shape
        dr2 = np.repeat(dpooled[:, :, None] / (h * w), h * w, axis=2).reshape(n, c, h, w)
        da2 = leaky_relu_backward(dr2, self._cache["a2"])
        dr1 = self.conv2.backward(da2, accumulate=True)
        da1 = leaky_relu_backward(dr1, self._cache["a1"])
        self.conv1.backward(da1, accumulate=True)
        self.conv2.apply_grad(lr)
        self.conv1.apply_grad(lr)

    def save(self, path):
        np.savez(path,
                  c1W=self.conv1.W, c1b=self.conv1.b, c1k=self.conv1.k, c1s=self.conv1.stride,
                  c2W=self.conv2.W, c2b=self.conv2.b, c2k=self.conv2.k, c2s=self.conv2.stride)

    def load(self, path):
        d = np.load(path)
        self.conv1.load({"W": d["c1W"], "b": d["c1b"], "k": d["c1k"], "stride": d["c1s"]})
        self.conv2.load({"W": d["c2W"], "b": d["c2b"], "k": d["c2k"], "stride": d["c2s"]})
