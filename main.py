import os, tempfile, shutil, zipfile, subprocess, re
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse
from starlette.background import BackgroundTask

app = FastAPI(title="Frame Extractor API (FFmpeg)")

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

# --------------------------
# Helpers
# --------------------------

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

def _download_youtube(youtube_url: str, dest_path: str):
    """
    Download a YouTube video to dest_path using yt-dlp.
    Requires yt-dlp to be installed in the environment.
    """
    if not youtube_url:
        raise HTTPException(status_code=400, detail="youtube_url is empty")

    # Prefer mp4 if available; otherwise best available format
    cmd = [
        "yt-dlp",
        "-f", "best[ext=mp4]/best",
        "-o", dest_path,     # exact output path
        "--quiet",
        "--no-warnings",
        youtube_url,
    ]
    try:
        subprocess.check_call(cmd)
    except FileNotFoundError:
        # yt-dlp not installed
        raise HTTPException(status_code=500, detail="yt-dlp is not installed on the server")
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=400, detail=f"yt-dlp failed: {e}")

    if not os.path.exists(dest_path) or os.path.getsize(dest_path) == 0:
        raise HTTPException(status_code=400, detail="Failed to download YouTube video")

# If you prefer pytube instead of yt-dlp, replace _download_youtube with:
#
# from pytube import YouTube
# def _download_youtube(youtube_url: str, dest_path: str):
#     yt = YouTube(youtube_url)
#     stream = yt.streams.filter(progressive=True, file_extension='mp4').first() or yt.streams.first()
#     if not stream:
#         raise HTTPException(status_code=400, detail="No downloadable streams found")
#     stream.download(filename=dest_path)

# --------------------------
# Main endpoint (accepts BOTH: multipart upload OR YouTube URL via form OR YouTube URL via JSON)
# --------------------------

@app.post("/extract_frames")
async def extract_frames(
    request: Request,
    # Multipart/form-data fields (old behavior)
    file: Optional[UploadFile] = File(None),   # field name "file"
    every_s: int = Form(1),
    start_s: int = Form(0),
    end_s:   int = Form(0),
    fmt:     str = Form("jpg"),
    quality: int = Form(95),
    zip_name: str = Form("frames.zip"),
    youtube_url: Optional[str] = Form(None),   # support YouTube via form-data too
):
    """
    Accepts:
      • Multipart upload with 'file' (previous behavior), or
      • Multipart/form-data with 'youtube_url', or
      • application/json with keys: youtube_url, interval_seconds, start_s, end_s, fmt, quality, zip_name
    Returns:
      • A ZIP of extracted frames.
    """

    # If neither 'file' nor 'youtube_url' were provided as form-data,
    # check for application/json and parse it.
    if not file and not youtube_url:
        ct = request.headers.get("content-type", "")
        if "application/json" in ct:
            try:
                data = await request.json()
            except Exception:
                data = {}
            youtube_url = data.get("youtube_url") or data.get("url")
            # allow alternative key name for convenience
            every_s   = int(data.get("interval_seconds", data.get("every_s", every_s)))
            start_s   = int(data.get("start_s", start_s))
            end_s     = int(data.get("end_s", end_s))
            fmt       = (data.get("fmt") or fmt).lower()
            quality   = int(data.get("quality", quality))
            zip_name  = data.get("zip_name", zip_name)

    # Validate fmt early
    if (fmt or "").lower() not in ("jpg", "jpeg", "png", "webp"):
        raise HTTPException(status_code=400, detail="fmt must be one of: jpg, png, webp")

    # temp workspace
    tmp_root = tempfile.mkdtemp(prefix="frames_")
    frames_dir = os.path.join(tmp_root, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    src_path = os.path.join(tmp_root, "input.mp4")

    def _cleanup():
        shutil.rmtree(tmp_root, ignore_errors=True)

    try:
        # Case 1: file upload (old path)
        if file is not None:
            with open(src_path, "wb") as f:
                f.write(await file.read())

        # Case 2: YouTube URL (form or JSON)
        elif youtube_url:
            _download_youtube(youtube_url, src_path)

        else:
            _cleanup()
            raise HTTPException(status_code=422, detail="Either 'file' or 'youtube_url' is required")

        # Extract & zip
        _ffmpeg_extract(src_path, frames_dir, every_s, start_s, end_s, fmt, quality)

        files = sorted(os.listdir(frames_dir))
        if not files:
            _cleanup()
            raise HTTPException(status_code=500, detail="No frames produced")

        zip_path = os.path.join(tmp_root, _safe_zip_name(zip_name))
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for name in files:
                full = os.path.join(frames_dir, name)
                zf.write(full, arcname=name)

        # Return the zip and clean up temp dir afterwards
        return FileResponse(
            zip_path,
            media_type="application/zip",
            filename=os.path.basename(zip_path),
            background=BackgroundTask(_cleanup),
        )

    except subprocess.CalledProcessError as e:
        _cleanup()
        raise HTTPException(status_code=500, detail=f"ffmpeg failed: {e}") from e
    except HTTPException:
        _cleanup()
        raise
    except Exception as e:
        _cleanup()
        raise HTTPException(status_code=500, detail=str(e))
