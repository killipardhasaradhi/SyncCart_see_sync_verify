"""SyncCart web app (customer mobile web app + staff screens).

Customer pages (send customers the site link):
    /            login (nickname) -> temporary customer code
    /shop        scan product QR codes / barcodes with the phone camera, cart, pay, exit QR
    /exit        shows the verification result for one customer code
Staff pages:
    /staff       live camera snapshot, physical items vs paid cart
    /gate        scan a customer's exit QR with the camera -> VERIFIED / POTENTIAL DISCREPANCY
    /labels      printable QR codes for the products

SYNCCART_MODE:
    local  (default): camera + YOLO run inside this app, on your laptop.
    server: public website (for example on Render). No camera here; agent.py on your laptop sends events.
"""
import hashlib
import hmac
import io
import json
import os
import secrets
import socket
import threading
import time
from contextlib import asynccontextmanager

import qrcode
from PIL import Image, ImageDraw
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel

from core import PickTracker, Store

MODE = os.environ.get("SYNCCART_MODE", "local")           # "local" or "server"
API_KEY = os.environ.get("SYNCCART_KEY", "")              # secret shared with agent.py (server mode)
STAFF_PIN = os.environ.get("SYNCCART_PIN", "")            # optional PIN for /staff, /gate, /labels (add ?pin=...)
MOCK = os.environ.get("SYNCCART_MOCK") == "1"             # run without camera/models, to test the website
MODEL_PATH = os.environ.get("SYNCCART_MODEL", "best.pt")  # your custom product model
CAMERA = int(os.environ.get("SYNCCART_CAMERA", "0"))
PORT = int(os.environ.get("PORT", os.environ.get("SYNCCART_PORT", "8000")))
PUBLIC_URL = os.environ.get("SYNCCART_PUBLIC_URL")        # set if you use a tunnel (ngrok / cloudflared)
TTL_HOURS = float(os.environ.get("SYNCCART_TTL_HOURS", "3"))
CODE_MAP = json.loads(os.environ.get("SYNCCART_CODES", "{}"))  # optional: {"8901234567890": "Notebook"} real barcodes
PRICES = json.loads(os.environ.get("SYNCCART_PRICES", "{}"))  # optional demo prices: {"Notebook": 60, "Chocolate": 20}
DEFAULT_PRICE = float(os.environ.get("SYNCCART_DEFAULT_PRICE", "50"))
ZONE = (0.15, 0.30, 0.85, 0.85)                           # shelf zone as fractions of the frame: x1, y1, x2, y2
CONF = 0.5
SECRET = (os.environ.get("SYNCCART_SECRET") or API_KEY or secrets.token_hex(16)).encode()

store = Store(ttl_hours=TTL_HOURS)
tracker = PickTracker()
lock = threading.Lock()
latest_jpeg = None
PRODUCTS = ["Notebook", "Chocolate", "Bottle"] if MOCK else []


def lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


BASE = (PUBLIC_URL or f"http://{lan_ip()}:{PORT}").rstrip("/")


def price_of(product):
    return PRICES.get(product, DEFAULT_PRICE)


def base(request: Request):
    if PUBLIC_URL:
        return PUBLIC_URL.rstrip("/")
    if MODE == "server":
        proto = request.headers.get("x-forwarded-proto", request.url.scheme)
        return f"{proto}://{request.headers.get('host')}"
    return BASE


def placeholder_frame(text):
    img = Image.new("RGB", (640, 480), (11, 31, 58))
    ImageDraw.Draw(img).text((30, 230), text, fill=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def camera_loop(prod_model, person_model):
    import cv2
    global latest_jpeg
    cap = cv2.VideoCapture(CAMERA)
    while True:
        ok, frame = cap.read()
        if not ok:
            latest_jpeg = placeholder_frame("Camera not available")
            time.sleep(0.3)
            continue
        h, w = frame.shape[:2]
        z = (int(ZONE[0] * w), int(ZONE[1] * h), int(ZONE[2] * w), int(ZONE[3] * h))

        seen = set()
        for b in prod_model(frame, conf=CONF, verbose=False)[0].boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            name = prod_model.names[int(b.cls)]
            if z[0] < (x1 + x2) / 2 < z[2] and z[1] < (y1 + y2) / 2 < z[3]:
                seen.add(name)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 165, 245), 2)
            cv2.putText(frame, name, (int(x1), int(y1) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 245), 2)

        person_near, m = False, 80
        for p in person_model(frame, classes=[0], verbose=False)[0].boxes:
            x1, y1, x2, y2 = p.xyxy[0].tolist()
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (177, 163, 15), 2)
            if not (x2 < z[0] - m or x1 > z[2] + m or y2 < z[1] - m or y1 > z[3] + m):
                person_near = True

        with lock:
            events = tracker.update(seen, person_near, time.time())
            sid = store.active if store.active and store.exists(store.active) else None
            if sid:
                for name, ev in events:
                    store.add_event(sid, name, ev)
            statuses = tracker.statuses()

        cv2.rectangle(frame, z[:2], z[2:], (0, 255, 255), 2)
        y = 25
        for n, st in statuses.items():
            cv2.putText(frame, f"{n}: {st}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y += 28
        small = cv2.resize(frame, (480, int(480 * h / w)))
        latest_jpeg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 60])[1].tobytes()


@asynccontextmanager
async def lifespan(app):
    global latest_jpeg
    if MOCK:
        latest_jpeg = placeholder_frame("MOCK MODE - no camera")
    elif MODE == "server":
        latest_jpeg = placeholder_frame("Waiting for the camera agent...")
    else:
        from ultralytics import YOLO
        prod_model = YOLO(MODEL_PATH)
        person_model = YOLO("yolo11n.pt")
        PRODUCTS[:] = list(prod_model.names.values())
        threading.Thread(target=camera_loop, args=(prod_model, person_model), daemon=True).start()
    print(f"\nSyncCart running in {MODE} mode on port {PORT}.")
    print(f"Customers open: {PUBLIC_URL or BASE}\nStaff screen:  {PUBLIC_URL or BASE}/staff\n")
    yield


app = FastAPI(title="SyncCart", lifespan=lifespan)


class LoginBody(BaseModel):
    name: str = ""


class ScanBody(BaseModel):
    product: str
    qty: int = 1


class CodeBody(BaseModel):
    code: str


class SimBody(BaseModel):
    product: str
    event: str


class ProductsBody(BaseModel):
    products: list[str]


def staff_ok(pin):
    return (not STAFF_PIN) or pin == STAFF_PIN


def check_key(request: Request):
    if API_KEY and request.headers.get("x-key") != API_KEY:
        raise HTTPException(401, "Bad key")


def get_session(sid):
    sid = (sid or "").upper()
    if not store.exists(sid):
        raise HTTPException(404, "Unknown or expired code")
    return sid


def resolve_code(code):
    """Product QR text is SC:<Product>#<unit>, e.g. SC:Notebook#003. Returns (product, unit) or (None, None)."""
    c = (code or "").strip()
    if c.upper().startswith("SC:"):
        c = c[3:].strip()
    unit = None
    if "#" in c:
        c, unit = c.rsplit("#", 1)
        unit = unit.strip() or None
    for p in PRODUCTS:
        if p.lower() == c.strip().lower():
            return p, unit
    mapped = CODE_MAP.get(c.strip())
    if mapped and (not PRODUCTS or mapped in PRODUCTS):
        return mapped, unit or c.strip()
    return None, None


def _fmt(d):
    return ",".join(f"{p}:{q}" for p, q in sorted(d.items()))


def _sig(sid, cart, phys):
    return hmac.new(SECRET, f"{sid}|{cart}|{phys}".encode(), hashlib.sha256).hexdigest()[:12]


def exit_payload(sid):
    """Session QR text. The phone refreshes it, so it always carries the latest paid items and camera summary."""
    cart, phys = _fmt(store.state(sid)["cart"]), _fmt(store.physical(sid))
    return f"SCX|{sid}|{cart}|{phys}|{_sig(sid, cart, phys)}"


def parse_exit(text):
    """Returns (sid, signature_ok, cart_in_qr) from a session QR, or just the code typed by hand."""
    t = (text or "").strip()
    if t.startswith("SCX|"):
        parts = t.split("|")
        if len(parts) == 5:
            _, sid, cart, phys, sig = parts
            return sid.upper(), hmac.compare_digest(sig, _sig(sid, cart, phys)), cart
    if "session=" in t:
        t = t.split("session=", 1)[1].split("&")[0]
    return t.upper(), None, None


# ----------------------------------------------------------------------------- customer API
@app.get("/api/products")
def products():
    return PRODUCTS


@app.get("/api/catalog")
def catalog():
    return [{"name": p, "price": price_of(p)} for p in PRODUCTS]


@app.post("/api/login")
def login(body: LoginBody, request: Request):
    with lock:
        sid = store.new_session(body.name)
        tracker.reset()
    return {"session_id": sid, "shop_url": f"{base(request)}/shop?session={sid}"}


@app.post("/api/scan/{sid}")
def scan(sid: str, body: ScanBody):
    sid = get_session(sid)
    if body.qty not in (-1, 1):
        raise HTTPException(400, "Bad quantity")
    if PRODUCTS and body.product not in PRODUCTS:
        raise HTTPException(400, "Unknown product")
    with lock:
        if store.is_paid(sid):
            raise HTTPException(409, "Already paid")
        if body.qty < 0:
            store.unscan(sid, body.product)
        else:
            store.scan(sid, body.product, 1)
    return {"ok": True}


@app.post("/api/scan_code/{sid}")
def scan_code(sid: str, body: CodeBody):
    sid = get_session(sid)
    product, unit = resolve_code(body.code)
    if not product:
        raise HTTPException(404, "Unknown product code")
    with lock:
        if store.is_paid(sid):
            raise HTTPException(409, "Already paid")
        if unit:
            r = store.scan_unit(sid, product, unit)
            if r == "dup":
                raise HTTPException(409, "You already scanned this exact item")
            if r == "taken":
                raise HTTPException(423, "This item is already in another customer's cart")
        else:
            store.scan(sid, product, 1)
    return {"product": product}


@app.post("/api/pay/{sid}")
def pay(sid: str):
    sid = get_session(sid)
    with lock:
        store.pay(sid)               # the camera keeps recording until the customer is scanned at the exit
    return {"ok": True}


@app.get("/api/state/{sid}")
def state(sid: str, request: Request):
    sid = get_session(sid)
    with lock:
        d = store.state(sid)
        d["shelf"] = tracker.statuses() if store.active == sid else {}
    d["shop_url"] = f"{base(request)}/shop?session={sid}"
    d["exit_url"] = f"{base(request)}/exit?session={sid}"
    d["exit_qr"] = exit_payload(sid)
    d["prices"] = {p: price_of(p) for p in d["cart"]}
    d["total"] = sum(price_of(p) * q for p, q in d["cart"].items())
    return d


@app.get("/api/verify/{sid}")
def verify(sid: str):
    sid = get_session(sid)
    with lock:
        return store.verify(sid)


@app.post("/api/exit_scan")
def exit_scan(body: CodeBody):
    sid, sig_ok, items = parse_exit(body.code)
    if not store.exists(sid):
        raise HTTPException(404, "Unknown or expired code")
    with lock:
        v = store.verify(sid)
        v["closed"] = False
        if v["paid"]:                # customer has reached the exit: check now, then stop recording this customer
            v["closed"] = True
            if store.active == sid:
                store.active = None
                tracker.reset()
    v["signature_ok"] = sig_ok
    v["qr_items"] = items
    return v


@app.get("/api/active")
def active(pin: str = ""):
    if not staff_ok(pin):
        raise HTTPException(401, "Staff PIN required")
    with lock:
        sid = store.active if store.active and store.exists(store.active) else None
        return {"session_id": sid, "name": store.sessions[sid]["name"] if sid else None, "recent": store.recent()}


@app.get("/api/sessions")
def sessions(pin: str = ""):
    """Staff-only snapshot of all currently valid customer sessions."""
    if not staff_ok(pin):
        raise HTTPException(401, "Staff PIN required")
    with lock:
        rows = []
        for sid, meta in list(store.sessions.items()):
            if not store.exists(sid):
                continue
            try:
                state_data = store.state(sid)
                result = store.verify(sid)
                rows.append({
                    "session_id": sid,
                    "name": meta.get("name") or "Customer",
                    "cart": state_data.get("cart", {}),
                    "physical": store.physical(sid),
                    "events": state_data.get("events", [])[-8:],
                    "paid": bool(state_data.get("paid", False)),
                    "result": result.get("result", "PENDING"),
                    "message": result.get("message", "Awaiting verification"),
                })
            except (KeyError, TypeError, AttributeError):
                # Skip incomplete/expired sessions instead of breaking the dashboard.
                continue
        return {"sessions": rows}


@app.get("/api/qr")
def qr(data: str):
    if len(data) > 300:
        raise HTTPException(400, "Too long")
    img = qrcode.make(data)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return Response(buf.getvalue(), media_type="image/png")


@app.get("/frame.jpg")
def frame_jpg(pin: str = ""):
    if not staff_ok(pin):
        raise HTTPException(401, "Staff PIN required")
    return Response(latest_jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


# ----------------------------------------------------------------------------- camera agent / testing API
@app.post("/api/agent/products")
def agent_products(body: ProductsBody, request: Request):
    check_key(request)
    PRODUCTS[:] = body.products
    return {"ok": True}


@app.post("/api/agent/frame")
async def agent_frame(request: Request):
    global latest_jpeg
    check_key(request)
    data = await request.body()
    if data:
        latest_jpeg = data
    return {"active": store.active}


@app.post("/api/agent/event")
def agent_event(body: SimBody, request: Request):
    check_key(request)
    with lock:
        sid = store.active if store.active and store.exists(store.active) else None
        if sid and body.event in ("PICKED", "RETURNED"):
            store.add_event(sid, body.product, body.event)
            return {"active": sid, "recorded": True}
    return {"active": store.active, "recorded": False}


@app.post("/api/sim/{sid}")
def simulate(sid: str, body: SimBody):
    """Only in MOCK mode: fake camera events, to test the website without a camera."""
    if not MOCK:
        raise HTTPException(403, "Simulation is only available in MOCK mode")
    sid = get_session(sid)
    with lock:
        store.add_event(sid, body.product, body.event)
    return {"ok": True}


# ----------------------------------------------------------------------------- pages
CSS = r"""
:root{--teal:#0FA3B1;--teal2:#19C9C0;--ink:#0B1F3A;--bg:#F3F8FB;--card:#fff;--line:#E3ECF3;--muted:#6B7C93;--ok:#1FAF6B;--warn:#F5A623;--bad:#D64545;--glow:0 10px 30px rgba(15,163,177,.25)}
*{box-sizing:border-box;-webkit-tap-highlight-color:transparent}
body{margin:0;font-family:Figtree,system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;background:var(--bg);color:var(--ink);line-height:1.4}
body.ops{--bg:#071427;--card:#0E2444;--line:#1B3A63;--muted:#8FA5C2;background:radial-gradient(1200px 600px at 80% -10%,#0f3a5e 0,#071427 60%);color:#fff}
.wrap{max-width:560px;margin:auto;padding:0 16px 120px;position:relative;z-index:1}
.wide{max-width:1180px;margin:auto;padding:16px}
.top{display:flex;align-items:center;gap:10px;padding:14px 16px;max-width:1180px;margin:auto}
.logo{display:flex;align-items:center;gap:9px;font-weight:800;font-size:20px;letter-spacing:-.02em}
.logo i{width:32px;height:32px;border-radius:10px;background:linear-gradient(135deg,var(--teal),var(--teal2));display:grid;place-items:center;color:#fff;font-style:normal;box-shadow:var(--glow)}
.top .sp{flex:1}
.pill{display:inline-flex;align-items:center;gap:6px;padding:6px 12px;border-radius:999px;font-size:13px;font-weight:700;background:#E6F6F8;color:#07808c}
.pill.ok{background:#E3F6EC;color:#12804c}.pill.warn{background:#FFF2DB;color:#9a6200}.pill.dark{background:rgba(25,201,192,.14);color:#6ff2e8}
.dot{width:8px;height:8px;border-radius:50%;background:currentColor;animation:pulse 1.6s infinite}
@keyframes pulse{50%{opacity:.35;transform:scale(.8)}}
.hero{background:linear-gradient(145deg,#0B1F3A 0,#0e4a6b 55%,#19C9C0 140%);color:#fff;border-radius:0 0 32px 32px;padding:26px 20px 70px;position:relative;overflow:hidden}
.hero:before{content:"";position:absolute;inset:0;background-image:linear-gradient(rgba(255,255,255,.06) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.06) 1px,transparent 1px);background-size:34px 34px;mask-image:linear-gradient(#000,transparent)}
.hero>*{position:relative}
.hero h1{margin:26px 0 6px;font-size:34px;line-height:1.1;letter-spacing:-.03em}
.hero p{margin:0;color:#cfe9ef;font-size:16px}
.orb{position:absolute!important;right:-40px;top:-30px;width:190px;height:190px;border-radius:50%;background:radial-gradient(circle,#19C9C0 0,transparent 65%);opacity:.55;filter:blur(6px)}
.card{background:var(--card);border:1px solid var(--line);border-radius:20px;padding:18px;margin:14px 0;box-shadow:0 6px 24px rgba(11,31,58,.06)}
body.ops .card{box-shadow:0 0 0 1px rgba(25,201,192,.08),0 10px 30px rgba(0,0,0,.35)}
.lift{margin-top:-46px}
.card h3{margin:0 0 12px;font-size:15px;font-weight:700;color:var(--muted)}
.btn{display:block;width:100%;border:0;border-radius:16px;padding:16px;font-size:17px;font-weight:800;color:#fff;background:linear-gradient(135deg,var(--teal),var(--teal2));box-shadow:var(--glow);cursor:pointer;transition:transform .12s}
.btn:active{transform:scale(.98)}.btn.ghost{background:#fff;color:var(--teal);border:1.5px solid var(--teal);box-shadow:none}
.btn.dark{background:#0B1F3A;box-shadow:none}.btn:disabled{opacity:.45;box-shadow:none}
.btn.sm{display:inline-block;width:auto;padding:11px 16px;font-size:15px;border-radius:12px}
input{width:100%;padding:15px;border-radius:14px;border:1.5px solid var(--line);background:#fff;color:var(--ink);font-size:17px;margin:0 0 12px;outline:none}
input:focus{border-color:var(--teal);box-shadow:0 0 0 4px rgba(15,163,177,.15)}
body.ops input{background:#0a1b35;color:#fff}
.steps{display:flex;gap:10px;margin:18px 0}.steps div{flex:1;text-align:center;font-size:13px;font-weight:700}
.steps b{display:grid;place-items:center;width:46px;height:46px;margin:0 auto 6px;border-radius:15px;background:#E6F6F8;font-size:22px}
.muted{color:var(--muted);font-size:14px}.row{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
.steprow{display:flex;align-items:center;gap:6px;margin:6px 0 2px}.steprow span{flex:1;height:5px;border-radius:5px;background:var(--line)}.steprow span.on{background:linear-gradient(90deg,var(--teal),var(--teal2))}
.steplab{display:flex;justify-content:space-between;font-size:12px;color:var(--muted);font-weight:700}
.item{display:flex;align-items:center;gap:12px;padding:12px 0;border-bottom:1px solid var(--line)}.item:last-child{border:0}
.item .em{width:52px;height:52px;border-radius:16px;background:#EAF7F8;display:grid;place-items:center;font-size:28px}
.item .nm{font-weight:800}.item .pr{color:var(--muted);font-size:14px}.item .sp{flex:1}
.qty{display:flex;align-items:center;gap:10px;font-weight:800}.qty button{width:34px;height:34px;border-radius:11px;border:1.5px solid var(--line);background:#fff;font-size:20px;color:var(--ink)}
.empty{text-align:center;padding:26px 8px;color:var(--muted)}.empty big{display:block;font-size:44px;margin-bottom:6px}
.bar{position:fixed;left:0;right:0;bottom:0;background:rgba(255,255,255,.92);backdrop-filter:blur(14px);border-top:1px solid var(--line);padding:12px 16px calc(12px + env(safe-area-inset-bottom));z-index:5}
.bar .in{max-width:560px;margin:auto;display:flex;align-items:center;gap:14px}.bar .tot{flex:1}.bar .tot small{display:block;color:var(--muted);font-weight:700;font-size:12px}.bar .tot b{font-size:24px}
.pass{background:linear-gradient(150deg,#0B1F3A,#12496c);color:#fff;border-radius:22px;padding:20px;text-align:center;position:relative;overflow:hidden}
.pass img{width:210px;max-width:70%;background:#fff;padding:10px;border-radius:16px;box-shadow:0 0 0 4px rgba(25,201,192,.35),0 0 40px rgba(25,201,192,.35)}
.pass .code{font-size:26px;font-weight:800;letter-spacing:.2em;margin-top:10px}.pass .muted{color:#a9c6d6}
.passmini{display:flex;gap:14px;align-items:center;width:100%;text-align:left;background:linear-gradient(150deg,#0B1F3A,#12496c);color:#fff;border:0;border-radius:20px;padding:12px 14px;margin:14px 0;cursor:pointer;font:inherit;box-shadow:0 6px 24px rgba(11,31,58,.18)}
.passmini img{width:88px;height:88px;background:#fff;padding:6px;border-radius:12px;box-shadow:0 0 0 3px rgba(25,201,192,.35)}
.passmini b{font-size:15px}.passmini .code{font-size:22px;font-weight:800;letter-spacing:.16em;margin:2px 0}.passmini .muted{color:#a9c6d6;font-size:13px}
.passov{position:fixed;inset:0;z-index:35;display:none;place-items:center;padding:20px;background:rgba(3,16,25,.8)}
.passov.show{display:grid}.passov .pass{width:100%;max-width:380px}.passov .pass img{width:280px;max-width:80%}
.scanov{position:fixed;inset:0;background:#031019;z-index:20;display:none;flex-direction:column}
.scanov .hd{display:flex;align-items:center;justify-content:space-between;padding:16px;color:#fff;font-weight:800}
.scanov .view{flex:1;position:relative;display:grid;place-items:center}
#reader{width:100%;max-width:520px}
.frame{position:absolute;width:250px;height:250px;pointer-events:none;border-radius:24px;box-shadow:0 0 0 999px rgba(0,0,0,.45)}
.frame i{position:absolute;width:38px;height:38px;border:4px solid var(--teal2)}.frame i:nth-child(1){left:0;top:0;border-right:0;border-bottom:0;border-top-left-radius:22px}.frame i:nth-child(2){right:0;top:0;border-left:0;border-bottom:0;border-top-right-radius:22px}.frame i:nth-child(3){left:0;bottom:0;border-right:0;border-top:0;border-bottom-left-radius:22px}.frame i:nth-child(4){right:0;bottom:0;border-left:0;border-top:0;border-bottom-right-radius:22px}
.frame:after{content:"";position:absolute;left:8px;right:8px;height:3px;background:linear-gradient(90deg,transparent,var(--teal2),transparent);box-shadow:0 0 14px var(--teal2);animation:sweep 2s ease-in-out infinite}
@keyframes sweep{0%{top:8%}50%{top:90%}100%{top:8%}}
.sheet{position:fixed;left:0;right:0;bottom:0;background:#fff;border-radius:26px 26px 0 0;padding:20px 18px calc(22px + env(safe-area-inset-bottom));z-index:30;transform:translateY(105%);transition:transform .28s;box-shadow:0 -20px 60px rgba(0,0,0,.25);max-width:560px;margin:auto}
.sheet.open{transform:none}.veil{position:fixed;inset:0;background:rgba(3,16,25,.5);z-index:29;display:none}
.method{display:flex;align-items:center;gap:12px;padding:14px;border:1.5px solid var(--line);border-radius:16px;margin:10px 0;font-weight:800;cursor:pointer}
.method.sel{border-color:var(--teal);background:#EAF9FA}.method span{font-size:24px}
.okov{position:fixed;inset:0;z-index:40;display:none;place-items:center;text-align:center;background:linear-gradient(160deg,#0B1F3A,#0e4a6b)}
.okov.show{display:grid}.okov h2{font-size:28px;margin:14px 0 4px;color:#fff}.okov p{color:#bfe3ea;margin:0}
.tick{width:110px;height:110px;border-radius:50%;background:var(--ok);display:grid;place-items:center;margin:auto;font-size:60px;color:#fff;box-shadow:0 0 0 0 rgba(31,175,107,.6);animation:pop .5s,ring 1.6s infinite}
@keyframes pop{0%{transform:scale(.3)}70%{transform:scale(1.12)}100%{transform:scale(1)}}@keyframes ring{70%{box-shadow:0 0 0 34px rgba(31,175,107,0)}100%{box-shadow:0 0 0 0 rgba(31,175,107,0)}}
.result{min-height:100vh;display:grid;place-items:center;text-align:center;padding:24px;color:#fff;transition:background .4s}
.result.ok{background:linear-gradient(160deg,#0a6b44,#1FAF6B)}.result.warn{background:linear-gradient(160deg,#a86a00,#F5A623)}.result.wait{background:linear-gradient(160deg,#0B1F3A,#12496c)}
.result h1{font-size:34px;margin:16px 0 6px}.result p{font-size:18px;margin:0 0 18px;opacity:.95}
.result .box{background:rgba(255,255,255,.16);border-radius:18px;padding:14px 16px;text-align:left;max-width:420px;margin:auto;font-size:15px}
.result .tick{background:rgba(255,255,255,.22);animation:pop .5s}
.chip{display:inline-block;background:rgba(255,255,255,.18);border-radius:999px;padding:4px 11px;margin:3px 2px;font-weight:700}
.chipd{display:inline-block;background:#EAF7F8;color:#07808c;border-radius:999px;padding:5px 12px;margin:3px 2px;font-weight:800}
body.ops .chipd{background:rgba(25,201,192,.15);color:#8ff5ec}
.grid{display:grid;gap:16px;grid-template-columns:1.5fr 1fr}@media(max-width:860px){.grid{grid-template-columns:1fr}}
.live{position:relative;border-radius:20px;overflow:hidden;border:1px solid #1d4d7a;box-shadow:0 0 0 1px rgba(25,201,192,.25),0 0 50px rgba(25,201,192,.12)}
.live img{width:100%;display:block;background:#000;min-height:220px}.live .tag{position:absolute;left:12px;top:12px}
.tl div{padding:8px 0;border-bottom:1px solid var(--line);font-size:14px;display:flex;gap:10px}.tl div:last-child{border:0}.tl time{color:var(--muted);min-width:64px}
.meter{height:12px;border-radius:12px;background:rgba(255,255,255,.1);overflow:hidden;margin:10px 0 4px}.meter i{display:block;height:100%;width:8%;background:var(--warn);transition:all .5s}
.big{font-size:22px;font-weight:800;letter-spacing:-.01em}
.verdict{border-radius:16px;padding:16px;font-weight:800;font-size:20px}.verdict.ok{background:rgba(31,175,107,.2);color:#6dffb6}.verdict.warn{background:rgba(245,166,35,.2);color:#ffd27a}.verdict.wait{background:rgba(143,165,194,.15);color:#c8d6ea}
.verdict small{display:block;font-weight:500;font-size:14px;opacity:.9;margin-top:4px}
.stat{display:flex;gap:10px}.stat .card{flex:1;margin:0;text-align:center}.stat b{font-size:26px;display:block}
#toast{position:fixed;left:50%;top:16px;transform:translateX(-50%) translateY(-90px);padding:12px 18px;border-radius:14px;font-weight:800;color:#fff;z-index:60;transition:transform .25s;max-width:92%;box-shadow:0 10px 30px rgba(0,0,0,.25)}
#toast.show{transform:translateX(-50%)}#toast.ok{background:var(--ok)}#toast.bad{background:var(--bad)}
.lab{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:14px}.lab .card{text-align:center;margin:0;break-inside:avoid}.lab img{width:100%;max-width:170px}
@media print{.noprint{display:none}body{background:#fff}}
a{color:var(--teal);font-weight:700}
h1,h2,.logo,.big,.btn,.bar .tot b,.pass .code,.result h1,.stat b,.hero h1{font-family:Sora,Figtree,system-ui,sans-serif}
:focus-visible{outline:3px solid var(--teal2);outline-offset:2px}
@media (prefers-reduced-motion:reduce){*,*:before,*:after{animation:none!important;transition:none!important}}
"""

CSS += r"""
.logo{white-space:nowrap}
body.app{--bg:#080f1d;--card:#101c33;--line:#1d2d4a;--muted:#8fa3c0;--ink:#fff;--teal:#19C9C0;--teal2:#2fe09b;--glow:0 10px 30px rgba(47,224,155,.22);background:radial-gradient(900px 520px at 85% -8%,#0f3d49 0,#080f1d 62%) fixed;color:#fff}
body.app .card{box-shadow:none}
body.app .btn{color:#04241a}
body.app .btn.ghost{background:transparent;color:var(--teal2);border-color:var(--teal2)}
body.app .btn.dark{color:#fff}
body.app input{background:#0a1426;color:#fff;border-color:var(--line)}
body.app .pill{background:rgba(47,224,155,.12);color:#6ff2b8}
body.app .pill.warn{background:rgba(245,166,35,.15);color:#ffd27a}
body.app .qty button{background:#0a1426;color:#fff}
body.app .item .em,body.app .steps b{background:#16294a}
body.app .sheet{background:#101c33}
body.app .method.sel{background:rgba(47,224,155,.1);border-color:var(--teal2)}
body.app a{color:var(--teal2)}
body.app .hero{background:linear-gradient(145deg,#0a1830 0,#0c3b4a 60%,#14a37f 150%)}
body.app .bar{background:rgba(8,15,29,.94);bottom:64px;padding-bottom:10px}
.nav{position:fixed;left:0;right:0;bottom:0;z-index:6;display:flex;justify-content:center;background:rgba(8,15,29,.97);border-top:1px solid var(--line);padding:6px 8px calc(6px + env(safe-area-inset-bottom))}
.nav>*{flex:1;max-width:140px;background:none;border:0;color:var(--muted);font:inherit;font-size:12px;font-weight:700;padding:6px 0;text-align:center;text-decoration:none;cursor:pointer;position:relative}
.nav b{display:block;font-size:20px;line-height:1.2}.nav .on{color:var(--teal2)}
.nav i{position:absolute;top:0;left:56%;background:var(--teal2);color:#04241a;font-style:normal;font-size:11px;min-width:18px;border-radius:9px;padding:1px 5px}
.tab{display:none}.tab.on{display:block}
.pgrid{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.pc{background:#0a1426;border:1px solid var(--line);border-radius:18px;padding:10px}
.pc .im{height:84px;border-radius:12px;background:linear-gradient(145deg,#16294a,#12384a);display:grid;place-items:center;font-size:42px}
.pc .nm{font-weight:800;margin:8px 0 2px}.pc .ft{display:flex;align-items:center;justify-content:space-between;margin-top:6px}.pc .ft b{font-size:18px}
.pc button{background:linear-gradient(135deg,var(--teal),var(--teal2));color:#04241a;border:0;border-radius:11px;padding:9px 12px;font-weight:800;cursor:pointer}
.sess{display:flex;align-items:center;gap:12px;width:100%;text-align:left;background:linear-gradient(120deg,#0c3b4a,#0f5a55);color:#fff;border:1px solid rgba(47,224,155,.35);border-radius:16px;padding:11px 14px;margin:12px 0;font:inherit;cursor:pointer}
.sess .code{font-weight:800;letter-spacing:.16em;font-size:18px}
.sum{display:flex;justify-content:space-between;padding:6px 0;color:var(--muted)}.sum.t{color:#fff;font-weight:800;font-size:20px}
.pass .code{color:#fff}
"""

LIB = '<script src="https://cdn.jsdelivr.net/npm/html5-qrcode@2.3.8/html5-qrcode.min.js"></script>'

COMMON_JS = r"""
const J={'Content-Type':'application/json'};
const PRICES_FALLBACK=50;
function em(p){p=(p||'').toLowerCase();const m=[[/choc|candy|sweet|cookie|biscuit/,'🍫'],[/note|book|pen|pencil|stationery/,'📓'],[/bottle|water|juice|drink|cola|milk/,'🧃'],[/bread|bun|cake/,'🍞'],[/apple|fruit|banana|orange/,'🍎'],[/chips|snack/,'🥔'],[/ear|phone|head/,'🎧'],[/spec|glass/,'👓'],[/rice|dal|atta/,'🌾']];for(const x of m)if(x[0].test(p))return x[1];return '🛒'}
function rs(n){return '₹'+Number(n||0).toFixed(0)}
function chips(o,cls){const k=Object.keys(o||{});return k.length?k.map(p=>'<span class='+(cls||'chipd')+'>'+em(p)+' '+p+' x'+o[p]+'</span>').join(''):'<span class=muted>(none)</span>'}
function toast(msg,ok){const t=document.getElementById('toast');t.textContent=msg;t.className=(ok?'ok':'bad')+' show';clearTimeout(window._tt);window._tt=setTimeout(()=>t.className=ok?'ok':'bad',2400)}
function buzz(){try{navigator.vibrate&&navigator.vibrate(70)}catch(e){}}
function beep(){try{const c=new (window.AudioContext||window.webkitAudioContext)();const o=c.createOscillator();const g=c.createGain();o.connect(g);g.connect(c.destination);o.frequency.value=880;g.gain.value=.08;o.start();setTimeout(()=>{o.stop();c.close()},110)}catch(e){}}
function verdict(el,v){const k=v.result=='VERIFIED'?'ok':v.result=='POTENTIAL DISCREPANCY'?'warn':'wait';el.className='verdict '+k;
 el.innerHTML=(k=='ok'?'&#10003; ':k=='warn'?'&#9888; ':'')+v.result+'<small>'+v.message+'</small>'}
"""

HEAD = '<!doctype html><html><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1,viewport-fit=cover"><meta name=theme-color content="#080f1d"><link rel=preconnect href="https://fonts.googleapis.com"><link rel=preconnect href="https://fonts.gstatic.com" crossorigin><link rel=stylesheet href="https://fonts.googleapis.com/css2?family=Figtree:wght@400;600;700;800&family=Sora:wght@600;700;800&display=swap"><title>Sync Cart</title><link rel=icon href="data:image/svg+xml,%3Csvg xmlns=%27http://www.w3.org/2000/svg%27 viewBox=%270 0 64 64%27%3E%3Crect width=%2764%27 height=%2764%27 rx=%2716%27 fill=%27%230FA3B1%27/%3E%3Cpath d=%27M16 24h6l5 18h20l4-14H26%27 stroke=%27white%27 stroke-width=%275%27 fill=none stroke-linecap=round stroke-linejoin=round/%3E%3C/svg%3E"><style>__CSS__</style></head><body class="__BODY__"><div id=toast></div>'
LOGO = '<div class=logo><i>&#9889;</i>Sync Cart</div>'

LOGIN = HEAD + r"""
<div class=hero><div class=orb></div>
 <div class=top style="padding:0">__LOGO__<span class=sp></span><span class="pill dark"><span class=dot></span>Store open</span></div>
 <h1>Scan. Pay.<br>Walk out.</h1><p>Smart checkout with silent, privacy-first verification.</p></div>
<div class=wrap>
 <div class="card lift" id=resume style="display:none"><h3>Welcome back</h3><div class=muted style="margin-bottom:12px">You have a shopping session in progress.</div>
  <button class=btn onclick="cont()">Continue shopping</button><button class="btn ghost" style="margin-top:10px" onclick="forget()">Start over</button></div>
 <div class="card lift" id=login><h3>Start shopping</h3>
  <input id=name placeholder="Your name (optional)" maxlength=20 autocomplete=off>
  <button class=btn onclick="start()">Start shopping</button></div>
 <div class=steps><div><b>&#128247;</b>Scan items</div><div><b>&#128179;</b>Pay on phone</div><div><b>&#9989;</b>Show exit QR</div></div>
 <div class=card><h3>How Sync Cart works</h3><div class=muted>1. Scan the QR on each item to build your cart.<br>2. Pay on your phone.<br>3. Show your exit pass. Staff do a quick check.</div></div>
 <div class=card><h3>Private by design</h3><div class=muted>No face recognition. Only a temporary anonymous session. If something does not match, a staff member simply does a quick friendly check.</div></div>
</div>
<script>__COMMON__
let saved=null;try{saved=localStorage.getItem('synccart_session')}catch(e){}
if(saved){fetch('/api/state/'+saved).then(r=>r.ok?r.json():null).then(d=>{if(d&&!d.paid){document.getElementById('resume').style.display='block';document.getElementById('login').style.display='none'}else{try{localStorage.removeItem('synccart_session')}catch(e){}}})}
function cont(){location.href='/shop?session='+saved}
function forget(){try{localStorage.removeItem('synccart_session')}catch(e){};document.getElementById('resume').style.display='none';document.getElementById('login').style.display='block'}
async function start(){const r=await fetch('/api/login',{method:'POST',headers:J,body:JSON.stringify({name:document.getElementById('name').value})});
 const d=await r.json();try{localStorage.setItem('synccart_session',d.session_id)}catch(e){};location.href='/shop?session='+d.session_id}
</script></body></html>"""

SHOP = HEAD + LIB + r"""
<div class=top>__LOGO__<span class=sp></span><span class=pill id=status><span class=dot></span>Shopping</span></div>
<div class=wrap>
 <div class=tab id=t-shop>
  <div class=steplab style="margin-top:6px"><span>Shop</span><span>Pay</span><span>Exit</span></div>
  <div class=steprow><span id=s1 class=on></span><span id=s2></span><span id=s3></span></div>
  <button class=sess onclick="tab('pass')"><span style="font-size:26px">&#9638;</span><span style="flex:1"><b>Your exit pass</b><br><span class=muted style="color:#a9d9cf">Live. Tap to show the QR at the exit.</span></span><span style="font-size:20px">&#8250;</span></button>
  <div id=shopcard>
   <h2 style="margin:6px 0 10px;font-size:26px">Scan your items.</h2>
   <div class=card style="text-align:center"><div style="font-size:54px">&#128247;</div><div class=muted style="margin:6px 0 14px">Scan the QR on each item you pick up. Each item has its own QR.</div>
   <button class=btn onclick="openScan()">Scan item QR</button></div></div>
  <div class=card id=basketcard><h3>Your basket</h3><div id=cart></div>
   <div class=sum><span>Items</span><span id=nitems>0</span></div></div>
  <div id=paidnote class=card style="display:none"><h3>Payment complete</h3><div class=muted>Walk to the exit gate and show your exit pass. Thank you for shopping with Sync Cart.</div></div>
 </div>
 <div class=tab id=t-pass><h2 style="margin:14px 0 10px;font-size:26px">Exit pass</h2>
  <div class=pass><div class=muted style="font-weight:700;margin-bottom:10px">Show this at the exit gate</div><img id=exitqr alt="Session QR"><div class=muted style="margin-top:8px">It updates as you shop.</div></div></div>
</div>
<div class=bar id=bar><div class=in><div class=tot><small>Total</small><b id=total>&#8377;0</b></div><button class="btn sm" id=paybtn onclick="openPay()" style="min-width:150px">Pay now</button></div></div>
<div class=nav><a href="/"><b>&#8962;</b>Home</a><button id=n-shop onclick="tab('shop')"><b>&#128247;</b>Scan<i id=badge style="display:none">0</i></button><button id=n-pass onclick="tab('pass')"><b>&#10003;</b>Exit pass</button></div>
<div class=scanov id=scanov><div class=hd><span>Scan an item</span><button class="btn sm dark" onclick="closeScan()" style="background:#1b3347">Close</button></div>
 <div class=view><div id=reader></div><div class=frame><i></i><i></i><i></i><i></i></div></div>
 <div class=hd style="justify-content:center;font-weight:600;color:#9fd">Point at the QR on the item</div></div>
<div class=veil id=veil onclick="closePay()"></div>
<div class=sheet id=sheet><h3 style="margin:0 0 4px;font-size:20px">Pay <span id=paytot></span></h3><div class=muted>Demo payment. No real money is used.</div>
 <div class="method sel" data-m=upi onclick="pick(this)"><span>&#128241;</span>UPI</div>
 <div class=method data-m=card onclick="pick(this)"><span>&#128179;</span>Card</div>
 <div class=method data-m=cash onclick="pick(this)"><span>&#128181;</span>Pay at counter</div>
 <button class=btn style="margin-top:10px" onclick="pay()" id=confirm>Confirm payment</button></div>
<div class=okov id=okov><div><div class=tick>&#10003;</div><h2>Payment successful</h2><p>Your exit pass is ready</p></div></div>
<script>__COMMON__
const sid=new URLSearchParams(location.search).get('session');
let scanner=null,lastCode='',lastTime=0,paid=false,cur={},cat=[];
function tab(n){['shop','pass'].forEach(x=>{document.getElementById('t-'+x).classList.toggle('on',x==n);document.getElementById('n-'+x).classList.toggle('on',x==n)});document.getElementById('bar').style.display=(paid||n=='pass')?'none':'block';window._tab=n;scrollTo(0,0)}
async function init(){if(!sid){location.href='/';return}
 const r=await fetch('/api/state/'+sid);if(!r.ok){try{localStorage.removeItem('synccart_session')}catch(e){};location.href='/';return}
 tab('shop');
 refresh();setInterval(refresh,3000)}
async function scanProduct(p,q){const r=await fetch('/api/scan/'+sid,{method:'POST',headers:J,body:JSON.stringify({product:p,qty:q})});
 if(r.ok){toast((q>0?'Added ':'Removed ')+p,true);refresh()}else{toast('Could not update the basket',false)}}
async function onCode(text){const now=Date.now();if(text===lastCode&&now-lastTime<2500)return;lastCode=text;lastTime=now;
 const r=await fetch('/api/scan_code/'+sid,{method:'POST',headers:J,body:JSON.stringify({code:text})});
 if(r.ok){const d=await r.json();buzz();beep();toast('Added '+d.product,true);refresh()}else{let m='Unknown code';try{m=(await r.json()).detail||m}catch(e){}toast(m,false)}}
async function openScan(){document.getElementById('scanov').style.display='flex';scanner=new Html5Qrcode('reader');
 try{await scanner.start({facingMode:'environment'},{fps:10,qrbox:{width:230,height:230}},onCode,()=>{})}
 catch(e){toast('Camera blocked. Allow camera access (needs https) or add items manually.',false);closeScan()}}
async function closeScan(){if(scanner){try{await scanner.stop()}catch(e){}scanner=null}document.getElementById('scanov').style.display='none'}
function openPass(){document.getElementById('passov').classList.add('show')}
function closePass(){document.getElementById('passov').classList.remove('show')}
function openPay(){document.getElementById('paytot').textContent=rs(cur.total);document.getElementById('veil').style.display='block';document.getElementById('sheet').classList.add('open')}
function closePay(){document.getElementById('veil').style.display='none';document.getElementById('sheet').classList.remove('open')}
function pick(el){document.querySelectorAll('.method').forEach(m=>m.classList.remove('sel'));el.classList.add('sel')}
async function pay(){document.getElementById('confirm').disabled=true;const r=await fetch('/api/pay/'+sid,{method:'POST'});closePay();document.getElementById('confirm').disabled=false;
 if(r.ok){const o=document.getElementById('okov');o.classList.add('show');buzz();setTimeout(()=>o.classList.remove('show'),1800)}refresh()}
async function refresh(){const r=await fetch('/api/state/'+sid);if(!r.ok){toast('Session expired',false);return}const s=await r.json();paid=s.paid;cur=s;
 const c=document.getElementById('cart');c.innerHTML='';const k=Object.keys(s.cart);
 if(!k.length)c.innerHTML='<div class=empty><big>&#128722;</big>Nothing scanned yet.<br>Tap Scan item QR to begin.</div>';
 k.forEach(p=>{const d=document.createElement('div');d.className='item';
  d.innerHTML='<div class=em>'+em(p)+'</div><div><div class=nm>'+p+'</div><div class=pr>'+rs(s.prices[p])+' each</div></div><div class=sp></div>';
  const q=document.createElement('div');q.className='qty';
  if(!s.paid){const m=document.createElement('button');m.textContent='−';m.onclick=()=>scanProduct(p,-1);q.appendChild(m)}
  const n=document.createElement('span');n.textContent=(s.paid?'x':'')+s.cart[p];q.appendChild(n);d.appendChild(q);c.appendChild(d)});
 document.getElementById('total').textContent=rs(s.total);
 const pb=document.getElementById('paybtn');pb.disabled=s.paid;pb.textContent=s.paid?'Paid':'Pay now';
 document.getElementById('shopcard').style.display=s.paid?'none':'block';
 document.getElementById('bar').style.display=(s.paid||window._tab=='pass')?'none':'block';
 const ni=Object.values(s.cart).reduce((a,b)=>a+b,0);document.getElementById('nitems').textContent=ni;const bd=document.getElementById('badge');bd.textContent=ni;bd.style.display=ni?'block':'none';
 document.getElementById('paidnote').style.display=s.paid?'block':'none';
  
 document.getElementById('s2').className=s.paid?'on':'';document.getElementById('s3').className=s.paid?'on':'';
 const st=document.getElementById('status');st.className='pill'+(s.paid?' ok':'');st.innerHTML='<span class=dot></span>'+(s.paid?'Paid · head to exit':'Shopping');
 if(window._lq!==s.exit_qr){window._lq=s.exit_qr;{const u='/api/qr?data='+encodeURIComponent(s.exit_qr);document.getElementById('exitqr').src=u}}}
init();
</script></body></html>"""

EXIT = HEAD + r"""
<div class="result wait" id=box><div style="width:100%">
 <div class=tick id=ic>&#8987;</div><h1 id=h>Checking...</h1><p id=p></p><div class=box id=detail style="display:none"></div>
 <div style="margin-top:20px"><button class="btn sm" style="background:rgba(255,255,255,.2);box-shadow:none" onclick="check()">Check again</button></div></div></div>
<script>__COMMON__
const sid=new URLSearchParams(location.search).get('session');
async function check(){const box=document.getElementById('box');if(!sid){document.getElementById('h').textContent='No session code';return}
 const r=await fetch('/api/verify/'+sid);if(!r.ok){document.getElementById('h').textContent='Unknown or expired code';return}
 const v=await r.json();const k=v.result=='VERIFIED'?'ok':v.result=='POTENTIAL DISCREPANCY'?'warn':'wait';box.className='result '+k;
 document.getElementById('ic').innerHTML=k=='ok'?'&#10003;':k=='warn'?'&#9888;':'&#8987;';
 document.getElementById('h').textContent=k=='ok'?'Verified':k=='warn'?'Quick check needed':'Payment pending';
 document.getElementById('p').textContent=k=='ok'?'Have a nice day!':k=='warn'?'A staff member will help you in a moment.':v.message;
 const d=document.getElementById('detail');d.style.display='block';
 d.innerHTML='Paid: '+chips(v.digital,'chip')+'<br>Seen by camera: '+chips(v.physical,'chip')+(k=='warn'?'<br><br>This is a routine friendly check, not an accusation.':'')}
check();
</script></body></html>"""

GATE = HEAD + LIB + r"""
<div class=top>__LOGO__<span class=sp></span><span class="pill dark">Exit gate</span></div>
<div class=wrap>
 <div class=card><h3>Scan the customer's exit pass</h3><div id=reader></div>
  <div class=row style="margin-top:10px"><button class="btn sm" onclick="startScan()">Start camera</button></div>
  <div class=muted style="margin-top:12px">Or type the customer code:</div>
  <div class=row><input id=manual placeholder="e.g. K7P2QX" style="max-width:220px;text-transform:uppercase;margin:0"><button class="btn sm" onclick="lookup(document.getElementById('manual').value)">Check</button></div></div>
 <div class=card><h3>Result</h3><div id=res class="verdict wait">Waiting for a scan</div><div id=detail class=muted style="margin-top:12px"></div></div>
</div>
<script>__COMMON__
let scanner=null,last='',lastT=0;
async function lookup(t){const code=(t||'').trim();if(!code)return;const now=Date.now();if(code===last&&now-lastT<3000)return;last=code;lastT=now;
 const r=await fetch('/api/exit_scan',{method:'POST',headers:J,body:JSON.stringify({code:code})});const el=document.getElementById('res');
 if(!r.ok){el.className='verdict wait';el.textContent='Unknown or expired code';return}
 const v=await r.json();verdict(el,v);buzz();
 let d='Customer '+v.session_id+(v.signature_ok===false?'<br><b>Warning: this QR does not match our records.</b>':'')+'<br>Camera saw: '+chips(v.physical)+'<br>Paid cart: '+chips(v.digital);
 if(v.result=='POTENTIAL DISCREPANCY'){d+='<br>Not in paid cart: '+chips(v.not_in_paid_cart)+'<br>Paid but not seen: '+chips(v.paid_but_not_seen)+'<br><br>Please do a friendly secondary check.'}
 document.getElementById('detail').innerHTML=d}
async function startScan(){if(scanner)return;scanner=new Html5Qrcode('reader');
 try{await scanner.start({facingMode:'environment'},{fps:10,qrbox:{width:240,height:240}},lookup,()=>{})}catch(e){toast('Camera blocked (needs https). Type the code instead.',false);scanner=null}}
</script></body></html>"""

LABELS = HEAD + r"""
<div class="top noprint">__LOGO__<span class=sp></span><span class=muted>Item QR labels. Print and stick one on each item.</span></div>
<div class=wide><div class="card noprint"><div class=row>Labels per product:
 <input id=n type=number value=3 min=1 max=20 style="max-width:90px;margin:0"><button class="btn sm" onclick="load()">Generate</button><button class="btn sm ghost" onclick="print()">Print</button></div></div>
<div id=grid class=lab></div></div>
<script>__COMMON__
async function load(){const n=Math.max(1,Math.min(20,parseInt(document.getElementById('n').value)||1));
 const ps=await (await fetch('/api/products')).json();const g=document.getElementById('grid');g.innerHTML='';
 if(!ps.length){g.innerHTML='<span class=muted>No products yet. Start the camera agent (or mock mode) first.</span>';return}
 ps.forEach(p=>{for(let i=1;i<=n;i++){const u=String(i).padStart(3,'0');const d=document.createElement('div');d.className='card';
  d.innerHTML='<img src="/api/qr?data='+encodeURIComponent('SC:'+p+'#'+u)+'"><div style="font-weight:800;font-size:18px;margin-top:6px">'+em(p)+' '+p+' #'+u+'</div>';g.appendChild(d)}})}
load();
</script></body></html>"""

STAFF = HEAD + r"""
<div class=top>__LOGO__<span class=sp></span><span class="pill dark"><span class=dot></span>Live</span><a id=lg href="/gate" style="margin-left:8px">Exit gate</a><a id=ll href="/labels" style="margin-left:8px">Labels</a></div>
<div class=wide>
 <div class=stat style="margin-bottom:16px">
  <div class=card><h3>Active sessions</h3><b id=nsessions>0</b></div>
  <div class=card><h3>Paid sessions</h3><b id=npaid>0</b></div>
  <div class=card><h3>Need review</h3><b id=nreview>0</b></div>
 </div>
 <div class=grid>
  <div>
   <div class=live><span class="pill dark tag"><span class=dot></span>Camera view</span><img id=live alt="Live camera feed"></div>
   <div class=card style="margin-top:16px"><h3>Recent camera events</h3><div class=tl id=events><span class=muted>Waiting for activity...</span></div></div>
  </div>
  <div>
   <div class=card><h3>All customer sessions</h3><p class=muted>Each customer's cart and verification result is shown separately.</p><div id=customers><div class=empty><big>⌛</big>Loading customer sessions…</div></div></div>
  </div>
 </div>
</div>
<script>__COMMON__
const PIN=new URLSearchParams(location.search).get('pin')||'';
document.getElementById('lg').href='/gate?pin='+encodeURIComponent(PIN);document.getElementById('ll').href='/labels?pin='+encodeURIComponent(PIN);
function sum(o){return Object.values(o||{}).reduce((a,b)=>a+(Number(b)||0),0)}
function safeText(tag, text, cls){const e=document.createElement(tag);e.textContent=text==null?'':String(text);if(cls)e.className=cls;return e}
function productText(o){const entries=Object.entries(o||{});return entries.length?entries.map(([k,v])=>k+' × '+v).join(', '):'None recorded'}
function sessionCard(c){
 const card=document.createElement('div');card.className='card';card.style.margin='12px 0';
 const head=document.createElement('div');head.className='row';head.style.justifyContent='space-between';
 head.appendChild(safeText('strong',(c.name||'Customer')+' · '+c.session_id,'big'));
 const label=c.result==='VERIFIED'?'Verified':c.result==='POTENTIAL DISCREPANCY'?'Needs a quick check':(c.paid?'Awaiting verification':'Shopping');
 const pill=safeText('span',label,'pill '+(c.result==='VERIFIED'?'ok':c.result==='POTENTIAL DISCREPANCY'?'warn':'dark'));head.appendChild(pill);card.appendChild(head);
 card.appendChild(safeText('p','Payment: '+(c.paid?'Complete':'Not paid'),'muted'));
 card.appendChild(safeText('p','Scanned / paid cart: '+productText(c.cart)));
 card.appendChild(safeText('p','Camera-recorded items: '+productText(c.physical)));
 card.appendChild(safeText('p',c.message||'Verification has not completed.','muted'));
 if(c.events&&c.events.length){const ev=document.createElement('div');ev.className='tl';ev.appendChild(safeText('strong','Recent events'));c.events.slice().reverse().forEach(x=>ev.appendChild(safeText('div',(x.time?x.time+' · ':'')+(x.product||'Item')+' '+(x.event||'event').toLowerCase())));card.appendChild(ev)}
 return card;
}
async function poll(){
 try{
  const [sr,ar]=await Promise.all([
   fetch('/api/sessions?pin='+encodeURIComponent(PIN),{cache:'no-store'}),
   fetch('/api/active?pin='+encodeURIComponent(PIN),{cache:'no-store'})
  ]);
  if(!sr.ok||!ar.ok)throw new Error('Staff access unavailable. Check the staff PIN.');
  const data=await sr.json(),a=await ar.json(),rows=data.sessions||[];
  document.getElementById('nsessions').textContent=rows.length;
  document.getElementById('npaid').textContent=rows.filter(x=>x.paid).length;
  document.getElementById('nreview').textContent=rows.filter(x=>x.result==='POTENTIAL DISCREPANCY').length;
  const list=document.getElementById('customers');list.replaceChildren();
  if(!rows.length)list.appendChild(safeText('div','No customer sessions yet. Customers can open the website and press Start shopping.','empty'));
  rows.slice().reverse().forEach(c=>list.appendChild(sessionCard(c)));
  const events=[];rows.forEach(c=>(c.events||[]).forEach(e=>events.push({...e,session_id:c.session_id,name:c.name})));
  events.sort((x,y)=>String(y.time||'').localeCompare(String(x.time||'')));
  const ev=document.getElementById('events');ev.replaceChildren();
  if(!events.length)ev.appendChild(safeText('span','No camera events recorded yet.','muted'));
  events.slice(0,12).forEach(e=>{const row=document.createElement('div');row.appendChild(safeText('time',e.time||'—'));row.appendChild(safeText('span',(e.name||'Customer')+' · '+(e.product||'Item')+' '+(e.event||'event').toLowerCase()));ev.appendChild(row)});
 }catch(err){const list=document.getElementById('customers');list.replaceChildren(safeText('p',err.message||'Could not load sessions.','muted'))}
}
poll();setInterval(poll,2000);
setInterval(()=>{document.getElementById('live').src='/frame.jpg?pin='+encodeURIComponent(PIN)+'&t='+Date.now()},700);
</script></body></html>"""


def page(html, ops=False):
    return HTMLResponse(html.replace("__COMMON__", COMMON_JS).replace("__CSS__", CSS)
                        .replace("__LOGO__", LOGO).replace("__BODY__", "ops" if ops else "app"))


def staff_page(html, pin):
    if not staff_ok(pin):
        return HTMLResponse("Staff PIN required. Add ?pin=YOURPIN to the link.", status_code=401)
    return page(html, ops=True)


@app.get("/")
def home():
    return page(LOGIN)


@app.get("/shop")
def shop():
    return page(SHOP)


@app.get("/exit")
def exit_page():
    return page(EXIT)


@app.get("/staff")
def staff(pin: str = ""):
    return staff_page(STAFF, pin)


@app.get("/gate")
def gate(pin: str = ""):
    return staff_page(GATE, pin)


@app.get("/labels")
def labels(pin: str = ""):
    return staff_page(LABELS, pin)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
