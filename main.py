import io, math, os, shutil, tempfile, zipfile
from pathlib import Path
from typing import Optional, Literal, List

import cv2
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse
from starlette.middleware.cors import CORSMiddleware

app = FastAPI(title="Frame Extractor API", version="1.0.0")

# CORS (set CORS_ALLOW_ORIGINS env to restrict in prod, e.g. "https://your-softr.app")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
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
):
    if quality < 1 or quality > 100:
        raise HTTPException(400, "quality must be between 1 and 100")
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

        mem = io.BytesIO()
        with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
            for p in frames:
                z.write(p, arcname=p.name)
        mem.seek(0)

        # Cleanup temp
        try:
            shutil.rmtree(frames[0].parent, ignore_errors=True)
        finally:
            video_path.unlink(missing_ok=True)

        return StreamingResponse(
            mem,
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{zip_name}"'},
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Processing failed: {e}") from e
