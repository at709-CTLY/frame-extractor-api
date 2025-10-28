import os, tempfile, shutil, zipfile, subprocess, re
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
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
    file: UploadFile = File(...),          # field name MUST be "file"
    every_s: int = Form(1),                # 1 frame every N seconds
    start_s: int = Form(0),                # optional trim start
    end_s:   int = Form(0),                # optional trim end
    fmt:     str = Form("jpg"),            # jpg|png|webp
    quality: int = Form(95),               # 0..100
    zip_name: str = Form("frames.zip"),    # returned filename
):
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
        _ffmpeg_extract(src_path, frames_dir, every_s, start_s, end_s, fmt, quality)

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
