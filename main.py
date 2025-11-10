import os
import tempfile
import shutil
import zipfile
import subprocess
import re

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse
from starlette.background import BackgroundTask

app = FastAPI(title="Frame Extractor API (FFmpeg) — PNG Only")

# CORS (open – Make/Softr friendly)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True}

def _safe_zip_name(name: str) -> str:
    name = (name or "frames.zip").strip()
    # keep safe chars only
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    if not name.lower().endswith(".zip"):
        name += ".zip"
    return name

def _ffmpeg_extract(src_path: str, out_dir: str, every_s: int, start_s: int, end_s: int):
    """
    Extract frames using ffmpeg at 1 frame every `every_s` seconds.
    Output is ALWAYS PNG (lossless). Any caller-provided format/quality is ignored.
    """
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

    # ALWAYS PNG
    out_pattern = os.path.join(out_dir, "frame_%06d.png")
    args += [out_pattern]

    subprocess.check_call(args)

@app.post("/extract_frames")
async def extract_frames(
    file: UploadFile = File(...),          # field name MUST be "file"
    every_s: int = Form(1),                # 1 frame every N seconds
    start_s: int = Form(0),                # optional trim start (seconds)
    end_s:   int = Form(0),                # optional trim end (seconds)
    fmt:     str = Form("png"),            # kept for backward compatibility, ignored
    quality: int = Form(95),               # kept for backward compatibility, ignored
    zip_name: str = Form("frames.zip"),    # returned filename
):
    """
    Extracts frames from the uploaded video and returns a ZIP of PNGs.

    Notes:
    - Output format is forced to PNG regardless of `fmt` provided.
    - `quality` is ignored for PNG (lossless).
    """
    if file is None:
        raise HTTPException(status_code=422, detail="file is required")

    # temp workspace
    tmp_root = tempfile.mkdtemp(prefix="frames_")
    src_path = os.path.join(tmp_root, file.filename or "input.bin")
    frames_dir = os.path.join(tmp_root, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    # save upload
    try:
        with open(src_path, "wb") as f:
            f.write(await file.read())
    except Exception as e:
        shutil.rmtree(tmp_root, ignore_errors=True)
        raise HTTPException(status_code=400, detail=f"could not save upload: {e}")

    # extract & zip
    try:
        _ffmpeg_extract(src_path, frames_dir, every_s, start_s, end_s)

        files = sorted(os.listdir(frames_dir))
        if not files:
            raise HTTPException(status_code=500, detail="No frames produced")

        zip_path = os.path.join(tmp_root, _safe_zip_name(zip_name))
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for name in files:
                full = os.path.join(frames_dir, name)
                zf.write(full, arcname=name)

        # return the zip and clean up temp dir afterwards
        return FileResponse(
            zip_path,
            media_type="application/zip",
            filename=os.path.basename(zip_path),
            background=BackgroundTask(lambda: shutil.rmtree(tmp_root, ignore_errors=True)),
        )

    except subprocess.CalledProcessError as e:
        shutil.rmtree(tmp_root, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"ffmpeg failed: {e}") from e
    except Exception:
        shutil.rmtree(tmp_root, ignore_errors=True)
        raise
