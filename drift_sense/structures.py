import numpy as np
import cv2


def _draw_via(canvas, cx, cy, r, intensity):
    cv2.circle(canvas, (int(cx), int(cy)), r, intensity, -1, lineType=cv2.LINE_AA)


def generate_dram_canvas(size=3000, pitch=42, line_width=6, via_radius=5,
                          base_intensity=205, bg_intensity=25, seed=None,
                          fid_len=90, fid_thickness=10):
    rng = np.random.default_rng(seed)
    canvas = np.full((size, size), bg_intensity, dtype=np.float32)

    xs = np.arange(pitch // 2, size, pitch)
    ys = np.arange(pitch // 2, size, pitch)

    for x in xs:
        jitter = rng.normal(0, 0.4)
        cv2.line(canvas, (int(x + jitter), 0), (int(x + jitter), size),
                  base_intensity, line_width, lineType=cv2.LINE_AA)
    for y in ys:
        jitter = rng.normal(0, 0.4)
        cv2.line(canvas, (0, int(y + jitter)), (size, int(y + jitter)),
                  base_intensity, line_width, lineType=cv2.LINE_AA)

    for x in xs:
        for y in ys:
            _draw_via(canvas, x, y, via_radius, min(255, base_intensity + 35))

    margin = max(pitch * 4, fid_len + 20)
    tx = int(rng.uniform(margin, size - margin))
    ty = int(rng.uniform(margin, size - margin))
    tx = int(xs[np.argmin(np.abs(xs - tx))])
    ty = int(ys[np.argmin(np.abs(ys - ty))])

    cv2.line(canvas, (tx - fid_len, ty - fid_len), (tx - fid_len, ty + fid_len),
              255, fid_thickness, cv2.LINE_AA)
    cv2.line(canvas, (tx - fid_len, ty - fid_len), (tx + fid_len, ty - fid_len),
              255, fid_thickness, cv2.LINE_AA)
    _draw_via(canvas, tx, ty, via_radius + fid_thickness // 2, 255)

    return np.clip(canvas, 0, 255).astype(np.uint8), (tx, ty)


def generate_finfet_canvas(size=3000, pitch=20, fin_width=5, gate_width=34,
                            base_intensity=200, bg_intensity=22, seed=None,
                            fid_len=90, fid_thickness=10):
    rng = np.random.default_rng(seed)
    canvas = np.full((size, size), bg_intensity, dtype=np.float32)

    xs = np.arange(pitch // 2, size, pitch)
    for x in xs:
        jitter = rng.normal(0, 0.3)
        cv2.line(canvas, (int(x + jitter), 0), (int(x + jitter), size),
                  base_intensity, fin_width, lineType=cv2.LINE_AA)

    gate_pitch = max(pitch * 16, fid_len * 3)
    ys = np.arange(gate_pitch // 2, size, gate_pitch)
    for y in ys:
        cv2.rectangle(canvas, (0, int(y - gate_width / 2)), (size, int(y + gate_width / 2)),
                       min(255, base_intensity + 25), -1)

    margin = max(gate_pitch, fid_len + 20)
    ty = int(ys[np.argmin(np.abs(ys - rng.uniform(margin, size - margin)))])
    tx = int(xs[np.argmin(np.abs(xs - rng.uniform(margin, size - margin)))])

    cv2.line(canvas, (tx - fid_len, ty - fid_len), (tx - fid_len, ty + fid_len),
              255, fid_thickness, cv2.LINE_AA)
    cv2.line(canvas, (tx - fid_len, ty - fid_len), (tx + fid_len, ty - fid_len),
              255, fid_thickness, cv2.LINE_AA)
    cv2.circle(canvas, (tx, ty), fin_width + fid_thickness // 2, 255, -1, cv2.LINE_AA)

    return np.clip(canvas, 0, 255).astype(np.uint8), (tx, ty)


GENERATORS = {
    "dram": generate_dram_canvas,
    "finfet": generate_finfet_canvas,
}
