#!/usr/bin/env python3

# A partial implementation of "list mode" for a given JSON CYZ file.
# Outputs a CSV file with summary data for each of the particles.

import json
import statistics
import pandas as pd
import sys
import os
import math
import base64
from io import BytesIO
from PIL import Image
import os.path
import numpy as np
import cv2



def _save_segmentation_summary(pil_img, meas, out_png):
    """
    Summary PNG for background-subtraction segmentation.

    Expects meas from _measure_largest_object_bgsub, i.e. keys like:
      - mask_small (bool or 0/255), diff_small (uint8), threshold_used (int)
      - bbox (x,y,w,h) in full-res coords (approx), centroid (cy,cx) in full-res
      - area_px, equiv_diameter_px, major_axis_px, minor_axis_px

    Falls back gracefully if some keys are missing.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    # --- original as grayscale array ---
    img = np.asarray(pil_img.convert("L"))
    H, W = img.shape[:2]

    # --- pull mask/diff from meas (prefer *_small) ---
    mask = meas.get("mask_small", meas.get("mask", None))
    diff = meas.get("diff_small", None)
    t = meas.get("threshold_used", None)

    # --- helper: resize to full-res for overlay ---
    def _resize_to_full(arr, is_mask=False):
        if arr is None:
            return None
        a = np.asarray(arr)
        if a.shape[:2] == (H, W):
            return a
        # nearest for masks; bilinear-ish for diff visual
        try:
            import cv2
            interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA
            a_u8 = a.astype(np.uint8)
            resized = cv2.resize(a_u8, (W, H), interpolation=interp)
            return resized.astype(bool) if is_mask else resized
        except Exception:
            # pure numpy fallback (cruder)
            # scale factors:
            ys = H / a.shape[0]
            xs = W / a.shape[1]
            yy = (np.arange(H) / ys).astype(int).clip(0, a.shape[0] - 1)
            xx = (np.arange(W) / xs).astype(int).clip(0, a.shape[1] - 1)
            resized = a[np.ix_(yy, xx)]
            return resized.astype(bool) if is_mask else resized

    mask_full = _resize_to_full(mask, is_mask=True) if mask is not None else None
    diff_full = _resize_to_full(diff, is_mask=False) if diff is not None else None

    # --- gather metrics for text overlay ---
    area = meas.get("area_px", None)
    ed = meas.get("equiv_diameter_px", None)
    major = meas.get("major_axis_px", None)
    minor = meas.get("minor_axis_px", None)
    bbox = meas.get("bbox", None)          # (x, y, w, h)
    centroid = meas.get("centroid", None)  # (cy, cx)

    # --- plotting ---
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)

    # Panel 1: original
    ax = axes[0]
    ax.imshow(img, cmap="gray")
    ax.set_title("Original")
    ax.axis("off")

    # Panel 2: diff (background - image)
    ax = axes[1]
    if diff_full is not None:
        # show with a robust contrast stretch
        vmin = np.percentile(diff_full, 5)
        vmax = np.percentile(diff_full, 99)
        ax.imshow(diff_full, cmap="magma", vmin=vmin, vmax=vmax)
        title = "Background-subtracted (bg - img)"
        if t is not None:
            title += f"\n threshold (pixel value difference, how many grey‑levels darker in 8‑bit grayscale units is the foreground object)  = {t}"
        ax.set_title(title)
    else:
        ax.imshow(np.zeros_like(img), cmap="magma")
        ax.set_title("Background-subtracted (missing)")
    ax.axis("off")

    # Panel 3: overlay
    ax = axes[2]
    ax.imshow(img, cmap="gray")
    if mask_full is not None:
        ax.imshow(mask_full, cmap="Reds", alpha=0.35)
    ax.set_title("Segmentation overlay + measurements")
    ax.axis("off")

    # draw bbox and centroid if present
    if bbox is not None and isinstance(bbox, (tuple, list)) and len(bbox) == 4:
        x, y, w, h = bbox
        try:
            import matplotlib.patches as patches
            rect = patches.Rectangle((x, y), w, h, linewidth=1.5,
                                     edgecolor="lime", facecolor="none")
            ax.add_patch(rect)
        except Exception:
            pass

    if centroid is not None and isinstance(centroid, (tuple, list)) and len(centroid) == 2:
        cy, cx = centroid
        ax.plot([cx], [cy], marker="x", color="cyan", markersize=8, mew=2)

    # annotate metrics
    lines = []
    if area is not None:
        lines.append(f"area: {area:.0f} px²")
    if ed is not None:
        lines.append(f"eq. diam: {ed:.1f} px")
    if major is not None and minor is not None:
        lines.append(f"axes: {major:.1f} × {minor:.1f} px")
    if t is not None:
        lines.append(f"thr: {t}")
    txt = " | ".join(lines) if lines else "no measurements"

    # put text on panel 3
    ax.text(
        0.01, 0.99,
        txt,
        transform=ax.transAxes,
        va="top", ha="left",
        fontsize=9,
        color="white",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="black", alpha=0.55),
    )

    # ensure output dir exists
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)

def _measure_largest_object_bgsub(
    pil_img,
    background_gray_u8,
    *,
    crop_rectangle=None,         # dict with X,Y,Width,Height if available
    downsample=0.1,              # 0.1 = 1/10th linear resolution
    sigma_k=3.0,                 # threshold = mean + k*std on bg-sub image
    min_area_px=50,              # minimum object area (FULL-res pixels) to accept
    morph_iters=1,               # small cleanup on the small mask
):
    """
    Fast background-subtraction segmentation:

    1) Take matching background patch (using crop_rectangle if possible)
    2) Downsample both (INTER_AREA)
    3) Compute diff = bg - img (objects darker => positive diff)
    4) Threshold using mean + k*std (cheap, stable, no Otsu)
    5) Contours -> largest object -> measurements

    Returns dict with area_px, equiv_diameter_px, major_axis_px, minor_axis_px,
    plus centroid/bbox/mask_small for debugging.
    """
    import numpy as np
    import cv2
    import time
    _t0 = time.perf_counter()
    
    if background_gray_u8 is None:
        return None

    img = np.asarray(pil_img.convert("L"), dtype=np.uint8)
    h, w = img.shape[:2]

    # --- choose background patch aligned to the particle image ---
    bg = background_gray_u8
    if crop_rectangle and isinstance(crop_rectangle, dict):
        x = int(crop_rectangle.get("X", 0))
        y = int(crop_rectangle.get("Y", 0))
        cw = int(crop_rectangle.get("Width", w))
        ch = int(crop_rectangle.get("Height", h))

        # safe bounds
        x2 = max(0, min(bg.shape[1], x + cw))
        y2 = max(0, min(bg.shape[0], y + ch))
        x = max(0, min(bg.shape[1], x))
        y = max(0, min(bg.shape[0], y))

        bg_patch = bg[y:y2, x:x2]
        # if patch mismatch, resize to image shape
        if bg_patch.size == 0:
            bg_patch = bg
        if bg_patch.shape != img.shape:
            bg_patch = cv2.resize(bg_patch, (w, h), interpolation=cv2.INTER_AREA)
        offset_xy = (x, y)  # for mapping centroid back to full frame, if needed
    else:
        # no crop info — best effort resize background to image size
        bg_patch = bg
        if bg_patch.shape != img.shape:
            bg_patch = cv2.resize(bg_patch, (w, h), interpolation=cv2.INTER_AREA)
        offset_xy = (0, 0)

    # --- downsample aggressively for speed ---
    if downsample is not None and 0 < downsample < 1.0:
        ws = max(8, int(round(w * downsample)))
        hs = max(8, int(round(h * downsample)))
        img_s = cv2.resize(img, (ws, hs), interpolation=cv2.INTER_AREA)
        bg_s  = cv2.resize(bg_patch, (ws, hs), interpolation=cv2.INTER_AREA)
        scale = float(downsample)
    else:
        img_s = img
        bg_s = bg_patch
        scale = 1.0

    # --- background subtraction: objects darker => bg - img is high at object ---
    diff = cv2.subtract(bg_s, img_s)

    # optional tiny blur to suppress sensor speckle
    diff = cv2.GaussianBlur(diff, (3, 3), 0)

    # --- fast threshold: mean + k*std ---
    mu = float(diff.mean())
    sd = float(diff.std())
    t = int(np.clip(mu + sigma_k * sd, 5, 255))

    _, mask = cv2.threshold(diff, t, 255, cv2.THRESH_BINARY)

    # --- small morphology cleanup on small mask ---
    if morph_iters and morph_iters > 0:
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.erode(mask, kernel, iterations=int(morph_iters))
        mask = cv2.dilate(mask, kernel, iterations=int(morph_iters))

    # --- contours ---
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    c = max(cnts, key=cv2.contourArea)
    
    
    area_small = float(cv2.contourArea(c))
    if area_small <= 0:
        return None
        

    # scale area back to full resolution
    area_full = area_small / (scale ** 2)

    if area_full < float(min_area_px):
        return None

    equiv_d = float((4 * area_full / np.pi) ** 0.5)

    # centroid (in small coords), then scale back
    M = cv2.moments(c)
    if M["m00"] == 0:
        return None
    cx_s = (M["m10"] / M["m00"])
    cy_s = (M["m01"] / M["m00"])

    cx = (cx_s / scale) + offset_xy[0]
    cy = (cy_s / scale) + offset_xy[1]

    # --- centroid gate: require object to be in central 50% ---
    xmin = 0.25 * w
    xmax = 0.75 * w
    ymin = 0.25 * h
    ymax = 0.75 * h

    if not (xmin <= cx <= xmax and ymin <= cy <= ymax):
        return None


    # major/minor axis from ellipse fit in small coords, scaled back
    if len(c) >= 5:
        (_, _), (ew, eh), _ = cv2.fitEllipse(c)
        major = float(max(ew, eh) / scale)
        minor = float(min(ew, eh) / scale)
    else:
        x, y, bw, bh = cv2.boundingRect(c)
        major = float(max(bw, bh) / scale)
        minor = float(min(bw, bh) / scale)

    # bbox in full-res image coords (approx)
    x, y, bw, bh = cv2.boundingRect(c)
    bbox = (
        int(round(x / scale)) + offset_xy[0],
        int(round(y / scale)) + offset_xy[1],
        int(round(bw / scale)),
        int(round(bh / scale)),
    )
    print("time in seconds:")
    print(time.perf_counter() - _t0)

    return {
        "area_px": float(area_full),
        "equiv_diameter_px": float(equiv_d),
        "major_axis_px": float(major),
        "minor_axis_px": float(minor),
        "centroid": (float(cy), float(cx)),
        "bbox": bbox,
        "mask_small": mask.astype(bool),   # small mask (for debug images)
        "diff_small": diff,                # optional: can visualise
        "threshold_used": int(t),
    }


def _decode_base64_image_to_gray_u8(b64_str):
    """
    Decode base64-encoded image (often PNG) to grayscale uint8 numpy array.
    Accepts raw base64 or data URI variants.
    """
    import base64
    import numpy as np
    from io import BytesIO
    from PIL import Image

    if b64_str is None:
        return None

    # Handle data URI prefix if present
    if isinstance(b64_str, str) and "base64," in b64_str:
        b64_str = b64_str.split("base64,", 1)[1]

    # Fix padding if truncated by transport (common)
    if isinstance(b64_str, str):
        b64_str = b64_str.strip()
        pad = (-len(b64_str)) % 4
        if pad:
            b64_str += "=" * pad
        raw = base64.b64decode(b64_str)
    else:
        # already bytes
        raw = base64.b64decode(b64_str)

    img = Image.open(BytesIO(raw)).convert("L")
    return np.asarray(img, dtype=np.uint8)

def extract(
    particles,
    dateandtime,
    images='',
    save_images_to='',
    segment_largest_object=False,
    images_only: bool = False,
    background=False,
    image_scale_um_per_px=None,
):
    """
    Extract listmode rows from particle JSON.

    Default behaviour (images_only=False):
        - All particles are emitted
        - Images are saved where available if `images` is provided

    Explicit opt-in behaviour (images_only=True):
        - Only particles that have an associated image are emitted
        - Requires `images` to be provided to have any effect
    """
    import os
    import base64
    from io import BytesIO
    from PIL import Image

    lines = []

    images_passed = bool(images)
    image_index = {}

    # ------------------------------------------------------------------
    # Build image lookup ONCE
    # ------------------------------------------------------------------
    if images_passed:
        for idx, img in enumerate(images):
            pid = img.get("particleId")
            if pid is not None:
                image_index[pid] = idx

        bg_gray = None
        if segment_largest_object and background:
            bg_gray = _decode_base64_image_to_gray_u8(background)



    # ------------------------------------------------------------------
    # Iterate particles
    # ------------------------------------------------------------------
    for particle in particles:
        pid = particle.get("particleId")

        # Explicit opt-in filter: images only
        if images_only and images_passed:
            if pid not in image_index:
                continue

        line = {}
        line["id"] = pid
        line["datetime"] = dateandtime

        # ------------------------------------------------------------------
        # Image handling (saving + optional segmentation)
        # ------------------------------------------------------------------
        if images_passed and pid in image_index:
            idx = image_index[pid]
            image_data = base64.b64decode(images[idx]["base64"])
            image = Image.open(BytesIO(image_data))

            if save_images_to:
                os.makedirs(save_images_to, exist_ok=True)
                image.save(os.path.join(save_images_to, f"{pid}.tif"))

            # Optional segmentation block (only runs if symbols exist)
            if segment_largest_object:
                imgrec = images[idx]
                meas = _measure_largest_object_bgsub(
                    image,
                    bg_gray,
                    downsample=1,
                    sigma_k=3.0,
                )      




                if meas is not None:
                    # Always keep the raw pixel measurements (useful for debugging/correlation)
                    line["img_area_px"] = meas.get("area_px")
                    line["img_equiv_diameter_px"] = meas.get("equiv_diameter_px")
                    line["img_major_axis_px"] = meas.get("major_axis_px")
                    line["img_minor_axis_px"] = meas.get("minor_axis_px")

                    # Add the scale so it’s recorded per-row (and the CSV remains self-describing)
                    # instrument field name says "MuPerPixel" so this is µm / pixel
                    line["img_scale_um_per_px"] = float(image_scale_um_per_px) if image_scale_um_per_px not in (None, "", False) else None

                    # Convert to microns where possible
                    s = line["img_scale_um_per_px"]
                    if s is not None:
                        # lengths: px * (µm/px) -> µm
                        if line["img_equiv_diameter_px"] is not None:
                            line["img_equiv_diameter_um"] = float(line["img_equiv_diameter_px"]) * s
                        else:
                            line["img_equiv_diameter_um"] = None

                        if line["img_major_axis_px"] is not None:
                            line["img_major_axis_um"] = float(line["img_major_axis_px"]) * s
                        else:
                            line["img_major_axis_um"] = None

                        if line["img_minor_axis_px"] is not None:
                            line["img_minor_axis_um"] = float(line["img_minor_axis_px"]) * s
                        else:
                            line["img_minor_axis_um"] = None

                        # area: px² * (µm/px)² -> µm²
                        if line["img_area_px"] is not None:
                            line["img_area_um2"] = float(line["img_area_px"]) * (s ** 2)
                        else:
                            line["img_area_um2"] = None
                    else:
                        # If scale is missing, still emit the µm columns as None
                        line["img_equiv_diameter_um"] = None
                        line["img_major_axis_um"] = None
                        line["img_minor_axis_um"] = None
                        line["img_area_um2"] = None
                else:
                    line["img_area_px"] = None
                    line["img_equiv_diameter_px"] = None
                    line["img_major_axis_px"] = None
                    line["img_minor_axis_px"] = None
                    line["img_scale_um_per_px"] = None
                    line["img_equiv_diameter_um"] = None
                    line["img_major_axis_um"] = None
                    line["img_minor_axis_um"] = None
                    line["img_area_um2"] = None




        # ------------------------------------------------------------------
        # Particle parameters (unchanged)
        # ------------------------------------------------------------------
        for parameter in particle.get("parameters", []):
            desc = parameter.get("description")
            if not desc:
                continue

            line[f"{desc}_length"] = parameter.get("length")
            line[f"{desc}_total"] = parameter.get("total")
            line[f"{desc}_maximum"] = parameter.get("maximum")
            line[f"{desc}_average"] = parameter.get("average")
            line[f"{desc}_inertia"] = parameter.get("inertia")
            line[f"{desc}_centreOfGravity"] = parameter.get("centreOfGravity")
            line[f"{desc}_fillFactor"] = parameter.get("fillFactor")
            line[f"{desc}_asymmetry"] = parameter.get("asymmetry")
            line[f"{desc}_numberOfCells"] = parameter.get("numberOfCells")
            line[f"{desc}_sampleLength"] = parameter.get("sampleLength")
            line[f"{desc}_timeOfArrival"] = parameter.get("timeOfArrival")
            line[f"{desc}_first"] = parameter.get("first")
            line[f"{desc}_last"] = parameter.get("last")
            line[f"{desc}_minimum"] = parameter.get("minimum")
            line[f"{desc}_swscov"] = parameter.get("swscov")
            line[f"{desc}_variableLength"] = parameter.get("variableLength")

        lines.append(line)

    return lines

# filename = "C:/Users/JR13/Documents/LOCAL_NOT_ONEDRIVE/lucinda-flow-cytometry/data/interim/json/nano_cend16_20 2020-10-09 00u03.cyz.json"

# filename = "C:/Users/JR13/Documents/LOCAL_NOT_ONEDRIVE/lucinda-flow-cytometry/temp/tempfile.json"

def main(filename,fileout,save_images_to = ''):
    #print(os.path.dirname(filename))
    data = json.load(open(filename, encoding="utf-8-sig"))
    lines = extract(particles=data["particles"],dateandtime = data["instrument"]["measurementResults"]["start"],images = data["images"], save_images_to = save_images_to+'/'+data["filename"])
    df = pd.DataFrame(lines)
    print("save to " +fileout)
    df.insert(loc=0, column="filename", value=os.path.basename(filename))
    df.to_csv(fileout, index=False)
    #outfile = os.path.basename(fileout)
    #print("saving to " + outfile)
    #df.to_csv(outfile, index=False)
    #df.to_csv(f"{outfile}.csv", index=False)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        if len(sys.argv) != 6:
            print("2 or 3 command line arguments expected in the form ['../dir/listmode.py', '../in/nano_cend16_20 2020-10-09 00u03.cyz.json', '--output', '../out/nano_cend16_20 2020-10-09 00u03.cyz.json.csv', '--imagedir','../imagesout/']")
            sys.exit(1)
    main(sys.argv[1],sys.argv[3],sys.argv[5])