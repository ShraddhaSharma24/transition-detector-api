import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from scenedetect import detect, ContentDetector, ThresholdDetector, AdaptiveDetector
import tempfile
import os

app = FastAPI()

# static mount so render can serve images
os.makedirs("/tmp/frames", exist_ok=True)
app.mount("/frames", StaticFiles(directory="/tmp/frames"), name="frames")


class TransitionFrameViewer:
    def __init__(self, video_path: str):
        self.video_path = video_path
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): raise ValueError("cannot open video")
        self.fps         = cap.get(cv2.CAP_PROP_FPS)
        self.total_frames= int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

    def brute_delta_pass(self, delta_thr=18):
        hits=[]
        cap=cv2.VideoCapture(self.video_path)
        last=None
        for f in range(self.total_frames):
            ret,fr=cap.read()
            if not ret: break
            g=cv2.cvtColor(fr,cv2.COLOR_BGR2GRAY)
            if last is not None:
                diff=cv2.absdiff(g,last).mean()
                if diff > delta_thr: hits.append(f)
            last=g
        cap.release()
        return hits

    def detect_transitions(self, content_threshold=9.0, adaptive_threshold=1.0, fade_threshold=4, min_scene_len=1, delta_thr=14, merge_window=10):
        all_trans=[]

        scenes = detect(self.video_path, ContentDetector(threshold=content_threshold, min_scene_len=min_scene_len))
        all_trans.extend([s.get_frames() for i,(s,_) in enumerate(scenes) if i>0])

        scenes = detect(self.video_path, AdaptiveDetector(adaptive_threshold=adaptive_threshold, min_scene_len=min_scene_len))
        all_trans.extend([s.get_frames() for i,(s,_) in enumerate(scenes) if i>0])

        scenes = detect(self.video_path, ThresholdDetector(threshold=fade_threshold, min_scene_len=min_scene_len))
        all_trans.extend([s.get_frames() for i,(s,_) in enumerate(scenes) if i>0])

        all_trans.extend(self.brute_delta_pass(delta_thr=delta_thr))
        all_trans=sorted(set(all_trans))

        merged=[all_trans[0]]
        for f in all_trans[1:]:
            if f-merged[-1] > merge_window: merged.append(f)
        return merged


@app.post("/detect")
async def detect_api(video: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(await video.read())
        tmp_path = tmp.name

    transitions_output = []

    try:
        viewer = TransitionFrameViewer(tmp_path)
        frames = viewer.detect_transitions()

        cap = cv2.VideoCapture(tmp_path)
        for f in frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ret, fr = cap.read()
            if ret:
                outpath = f"/tmp/frames/frame_{f}.png"
                cv2.imwrite(outpath, fr)
                transitions_output.append({
                    "frame_no": f,
                    "url": f"/frames/frame_{f}.png"
                })
        cap.release()

        return JSONResponse({"transitions": transitions_output})

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


