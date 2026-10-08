"""SyncCart camera agent. Runs on the laptop that has the webcam.

It watches the shelf with YOLO and sends PICKED / RETURNED events (and small snapshots)
to your public SyncCart website. No video is stored on the website.

Windows PowerShell:
    $env:SYNCCART_SERVER="https://your-site.onrender.com"; $env:SYNCCART_KEY="your-secret"; python agent.py
Mac / Linux:
    SYNCCART_SERVER=https://your-site.onrender.com SYNCCART_KEY=your-secret python agent.py
"""
import os
import queue
import threading
import time

import cv2
import requests
from ultralytics import YOLO

from core import PickTracker

SERVER = os.environ["SYNCCART_SERVER"].rstrip("/")
KEY = os.environ.get("SYNCCART_KEY", "")
MODEL_PATH = os.environ.get("SYNCCART_MODEL", "best.pt")
CAMERA = int(os.environ.get("SYNCCART_CAMERA", "0"))
ZONE = (0.15, 0.30, 0.85, 0.85)   # shelf zone as fractions of the frame: x1, y1, x2, y2
CONF = 0.5
HEAD = {"X-Key": KEY}

events_q = queue.Queue()
frame_box = {"jpg": None}
active = {"sid": None}


def post(path, **kw):
    return requests.post(SERVER + path, headers={**HEAD, **kw.pop("headers", {})}, timeout=8, **kw)


def sender():
    last_frame = 0
    while True:
        try:
            while not events_q.empty():
                name, ev = events_q.get()
                r = post("/api/agent/event", json={"product": name, "event": ev})
                active["sid"] = r.json().get("active")
                print("sent:", name, ev, "->", "recorded" if r.json().get("recorded") else "no active session")
            if frame_box["jpg"] and time.time() - last_frame > 0.5:
                r = post("/api/agent/frame", data=frame_box["jpg"], headers={"Content-Type": "image/jpeg"})
                active["sid"] = r.json().get("active")
                last_frame = time.time()
        except Exception as e:
            print("send error (is the website awake?):", e)
            time.sleep(2)
        time.sleep(0.05)


def main():
    prod = YOLO(MODEL_PATH)
    person = YOLO("yolo11n.pt")

    while True:  # register product names; retry while the website wakes up
        try:
            post("/api/agent/products", json={"products": list(prod.names.values())}).raise_for_status()
            print("Connected to", SERVER)
            break
        except Exception as e:
            print("Waiting for the website...", e)
            time.sleep(3)

    threading.Thread(target=sender, daemon=True).start()
    tracker = PickTracker()
    last_sid = None
    cap = cv2.VideoCapture(CAMERA)

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.2)
            continue
        if active["sid"] != last_sid:      # a new customer session started: reset the shelf tracking
            tracker.reset()
            last_sid = active["sid"]

        h, w = frame.shape[:2]
        z = (int(ZONE[0] * w), int(ZONE[1] * h), int(ZONE[2] * w), int(ZONE[3] * h))
        seen = set()
        for b in prod(frame, conf=CONF, verbose=False)[0].boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            name = prod.names[int(b.cls)]
            if z[0] < (x1 + x2) / 2 < z[2] and z[1] < (y1 + y2) / 2 < z[3]:
                seen.add(name)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 165, 245), 2)
            cv2.putText(frame, name, (int(x1), int(y1) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 245), 2)

        person_near, m = False, 80
        for p in person(frame, classes=[0], verbose=False)[0].boxes:
            x1, y1, x2, y2 = p.xyxy[0].tolist()
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (177, 163, 15), 2)
            if not (x2 < z[0] - m or x1 > z[2] + m or y2 < z[1] - m or y1 > z[3] + m):
                person_near = True

        for name, ev in tracker.update(seen, person_near, time.time()):
            events_q.put((name, ev))

        cv2.rectangle(frame, z[:2], z[2:], (0, 255, 255), 2)
        y = 25
        for n, st in tracker.statuses().items():
            cv2.putText(frame, f"{n}: {st}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y += 28
        small = cv2.resize(frame, (480, int(480 * h / w)))
        frame_box["jpg"] = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 60])[1].tobytes()
        cv2.imshow("SyncCart agent (press q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
