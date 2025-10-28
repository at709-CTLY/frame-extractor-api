import io, math, os, shutil, tempfile, zipfile
from pathlib import Path
from typing import Optional, Literal, List
import os, tempfile, shutil, zipfile, subprocess, re
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse
from starlette.background import BackgroundTask

import cv2
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse
from starlette.middleware.cors import CORSMiddleware
app = FastAPI(title="Frame Extractor API (FFmpeg)")

app = FastAPI(title="Frame Extractor API", version="1.0.0")

# CORS (set CORS_ALLOW_ORIGINS env to restrict in prod, e.g. "https://your-softr.app")
# CORS (open – Make/Softr friendly)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def _extract_frames(
    video_path: Path,
    every_s: float = 1.0,
    start_s: float = 0.0,
    end_s: Optional[float] = None,
    fmt: Literal["jpg", "png", "webp"] = "jpg",
    quality: int = 95,
    max_frames: int = 1000,
) -> List[Path]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("Could not open video.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration = frame_count / fps if fps else 0.0
    if end_s is None or end_s <= 0:
        end_s = duration

    if start_s < 0 or start_s >= end_s:
        raise ValueError("Invalid start/end range.")

    # Build timestamps at the given interval
    timestamps = []
    t = start_s
    while t <= end_s + 1e-6:
        timestamps.append(round(t, 3))
        t += max(every_s, 0.001)

    # Output dir
    tmp_dir = Path(tempfile.mkdtemp(prefix="frames_"))
    written = []
    try:
        for i, sec in enumerate(timestamps):
            if i >= max_frames:
                break
            cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            fname = tmp_dir / f"frame_{int(sec):04d}s.{fmt}"
            if fmt == "jpg":
                cv2.imwrite(str(fname), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            elif fmt == "png":
                comp = max(0, min(9, int((100-quality)/11)))  # map quality→png compression
                cv2.imwrite(str(fname), frame, [cv2.IMWRITE_PNG_COMPRESSION, comp])
            else:  # webp
                cv2.imwrite(str(fname), frame, [cv2.IMWRITE_WEBP_QUALITY, quality])
            written.append(fname)

        cap.release()
        if not written:
            raise ValueError("No frames were extracted (check your time range).")
        return written
    except Exception as e:
        cap.release()
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise e
@app.get("/health")
def health():
    return {"ok": True}

def _q_from_quality(quality: int) -> str:
    """
    FFmpeg JPEG quality uses qscale 2(best)..31(worst).
    Map 0..100 -> 31..2 (clamped).
    """
    q = max(2, min(31, int(round(31 - (quality / 100.0) * 29))))
    return str(q)

def _safe_zip_name(name: str) -> str:
    name = (name or "frames.zip").strip()
    # keep safe chars only
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    if not name.lower().endswith(".zip"):
        name += ".zip"
    return name

def _ffmpeg_extract(src_path: str, out_dir: str, every_s: int, start_s: int, end_s: int, fmt: str, quality: int):
    fmt = (fmt or "jpg").lower()
    if fmt not in ("jpg", "jpeg", "png", "webp"):
        raise HTTPException(status_code=400, detail="fmt must be one of: jpg, png, webp")

    args = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]

    # trim start
    if start_s and start_s > 0:
        args += ["-ss", str(int(start_s))]

    args += ["-i", src_path]

    # trim end via duration
    if end_s and end_s > 0 and (not start_s or end_s > start_s):
        dur = end_s - (start_s or 0)
        if dur > 0:
            args += ["-t", str(int(dur))]

    # 1 frame every N seconds
    args += ["-vf", f"fps=1/{max(1, int(every_s))}"]

    # JPEG/WebP quality
    if fmt in ("jpg", "jpeg"):
        args += ["-q:v", _q_from_quality(int(quality))]
    elif fmt == "webp":
        # 0(best)..100(worst) for libwebp, invert input (95 -> ~5)
        webp_q = max(0, min(100, 100 - int(quality)))
        args += ["-quality", str(webp_q)]

    out_pattern = os.path.join(out_dir, f"frame_%06d.{fmt}")
    args += [out_pattern]

    subprocess.check_call(args)

@app.post("/extract_frames")
async def extract_frames(
    file: UploadFile = File(..., description="Video file (mp4, mov, etc.)"),
    every_s: float = Form(1.0, description="Interval in seconds (e.g., 1.0)"),
    start_s: float = Form(0.0, description="Start time (sec)"),
    end_s: float = Form(0.0, description="End time (sec, 0 means full length)"),
    fmt: Literal["jpg", "png", "webp"] = Form("jpg"),
    quality: int = Form(95, description="Image quality 1–100"),
    max_frames: int = Form(1000, description="Safety cap"),
    zip_name: str = Form("frames_1s.zip"),
    file: UploadFile = File(...),          # field name MUST be "file"
    every_s: int = Form(1),                # 1 frame every N seconds
    start_s: int = Form(0),                # optional trim start
    end_s:   int = Form(0),                # optional trim end
    fmt:     str = Form("jpg"),            # jpg|png|webp
    quality: int = Form(95),               # 0..100
    zip_name: str = Form("frames.zip"),    # returned filename
):
    if quality < 1 or quality > 100:
        raise HTTPException(400, "quality must be between 1 and 100")
    if file is None:
        raise HTTPException(status_code=422, detail="file is required")

    # temp workspace
    tmp_root = tempfile.mkdtemp(prefix="frames_")
    src_path = os.path.join(tmp_root, file.filename or "input.bin")
    frames_dir = os.path.join(tmp_root, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    # save upload
    try:
        suffix = Path(file.filename or "video").suffix.lower() or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            video_path = Path(tmp.name)
            content = await file.read()
            size_limit_mb = int(os.getenv("MAX_UPLOAD_MB", "300"))
            if len(content) > size_limit_mb * 1024 * 1024:
                raise HTTPException(413, f"File exceeds {size_limit_mb}MB limit.")
            tmp.write(content)

        frames = _extract_frames(
            video_path=video_path,
            every_s=every_s,
            start_s=start_s,
            end_s=end_s if end_s > 0 else None,
            fmt=fmt,
            quality=quality,
            max_frames=max_frames,
        )
        with open(src_path, "wb") as f:
            f.write(await file.read())
    except Exception as e:
        shutil.rmtree(tmp_root, ignore_errors=True)
        raise HTTPException(status_code=400, detail=f"could not save upload: {e}")

    # extract & zip
    try:
        _ffmpeg_extract(src_path, frames_dir, every_s, start_s, end_s, fmt, quality)

        mem = io.BytesIO()
        with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
            for p in frames:
                z.write(p, arcname=p.name)
        mem.seek(0)
        files = sorted(os.listdir(frames_dir))
        if not files:
            raise HTTPException(status_code=500, detail="No frames produced")

        # Cleanup temp
        try:
            shutil.rmtree(frames[0].parent, ignore_errors=True)
        finally:
            video_path.unlink(missing_ok=True)
        zip_path = os.path.join(tmp_root, _safe_zip_name(zip_name))
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for name in files:
                full = os.path.join(frames_dir, name)
                zf.write(full, arcname=name)

        return StreamingResponse(
            mem,
        # return the zip and clean up temp dir afterwards
        return FileResponse(
            zip_path,
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{zip_name}"'},
            filename=os.path.basename(zip_path),
            background=BackgroundTask(lambda: shutil.rmtree(tmp_root, ignore_errors=True)),
        )
    except HTTPException:

    except subprocess.CalledProcessError as e:
        shutil.rmtree(tmp_root, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"ffmpeg failed: {e}") from e
    except Exception:
        shutil.rmtree(tmp_root, ignore_errors=True)
        raise
    except Exception as e:
        raise HTTPException(500, f"Processing failed: {e}") from e
