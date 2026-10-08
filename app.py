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
import io
import json
import os
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
ZONE = (0.15, 0.30, 0.85, 0.85)                           # shelf zone as fractions of the frame: x1, y1, x2, y2
CONF = 0.5

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
    c = (code or "").strip()
    if c.upper().startswith("SC:"):
        c = c[3:].strip()
    for p in PRODUCTS:
        if p.lower() == c.lower():
            return p
    mapped = CODE_MAP.get(c)
    if mapped and (not PRODUCTS or mapped in PRODUCTS):
        return mapped
    return None


# ----------------------------------------------------------------------------- customer API
@app.get("/api/products")
def products():
    return PRODUCTS


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
        store.scan(sid, body.product, body.qty)
    return {"ok": True}


@app.post("/api/scan_code/{sid}")
def scan_code(sid: str, body: CodeBody):
    sid = get_session(sid)
    product = resolve_code(body.code)
    if not product:
        raise HTTPException(404, "Unknown product code")
    with lock:
        if store.is_paid(sid):
            raise HTTPException(409, "Already paid")
        store.scan(sid, product, 1)
    return {"product": product}


@app.post("/api/pay/{sid}")
def pay(sid: str):
    sid = get_session(sid)
    with lock:
        store.pay(sid)
    return {"ok": True}


@app.get("/api/state/{sid}")
def state(sid: str, request: Request):
    sid = get_session(sid)
    with lock:
        d = store.state(sid)
        d["shelf"] = tracker.statuses() if store.active == sid else {}
    d["shop_url"] = f"{base(request)}/shop?session={sid}"
    d["exit_url"] = f"{base(request)}/exit?session={sid}"
    return d


@app.get("/api/verify/{sid}")
def verify(sid: str):
    sid = get_session(sid)
    with lock:
        return store.verify(sid)


@app.get("/api/active")
def active(pin: str = ""):
    if not staff_ok(pin):
        raise HTTPException(401, "Staff PIN required")
    with lock:
        sid = store.active if store.active and store.exists(store.active) else None
        return {"session_id": sid, "name": store.sessions[sid]["name"] if sid else None, "recent": store.recent()}


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
*{box-sizing:border-box}body{margin:0;font-family:system-ui,Segoe UI,Arial,sans-serif;background:#0B1F3A;color:#fff}
header{padding:14px 18px;background:#13355E;display:flex;gap:12px;align-items:baseline;flex-wrap:wrap}
header b{font-size:22px}header span{color:#F5A623;font-weight:600}header small{margin-left:auto;color:#8A99AD}
main{padding:16px;max-width:1100px;margin:auto}.grid{display:grid;gap:16px;grid-template-columns:1.4fr 1fr}
@media(max-width:800px){.grid{grid-template-columns:1fr}}
.card{background:#13355E;border-radius:14px;padding:16px;margin-bottom:16px}
.card h3{margin:0 0 10px;font-size:13px;letter-spacing:.08em;color:#8A99AD}
button{background:#0FA3B1;color:#0B1F3A;border:0;border-radius:10px;padding:12px 16px;font-size:16px;font-weight:700;cursor:pointer}
button.alt{background:#F5A623}button.go{background:#2EAD6B;color:#fff}button.ghost{background:#0B1F3A;color:#fff}button:disabled{opacity:.5}
input{width:100%;padding:13px;border-radius:10px;border:1px solid #5B7DB1;background:#0B1F3A;color:#fff;font-size:16px;margin:8px 0}
img.live{width:100%;border-radius:12px;background:#000}img.qr{width:200px;max-width:100%;background:#fff;padding:8px;border-radius:10px}
.chip{display:inline-flex;gap:8px;align-items:center;background:#0B1F3A;border-radius:999px;padding:6px 12px;margin:3px;font-weight:600}
.chip button{padding:0 9px;font-size:16px;border-radius:999px}
.badge{border-radius:14px;padding:18px;font-size:22px;font-weight:800}
.ok{background:#2EAD6B}.warn{background:#F5A623;color:#0B1F3A}.wait{background:#5B7DB1}
.muted{color:#8A99AD;font-size:14px}.row{display:flex;gap:8px;flex-wrap:wrap;margin:8px 0}
.code{font-size:34px;font-weight:800;letter-spacing:.15em}
#toast{position:fixed;left:50%;bottom:24px;transform:translateX(-50%);padding:12px 18px;border-radius:12px;font-weight:700;display:none;z-index:9}
#reader{width:100%;border-radius:12px;overflow:hidden}
a{color:#0FA3B1}
"""

LIB = '<script src="https://cdn.jsdelivr.net/npm/html5-qrcode@2.3.8/html5-qrcode.min.js"></script>'

COMMON_JS = r"""
function chips(o){const k=Object.keys(o||{});return k.length?k.map(p=>'<span class=chip>'+p+' x'+o[p]+'</span>').join(''):'<span class=muted>(none)</span>'}
function toast(msg,ok){const t=document.getElementById('toast');t.textContent=msg;t.style.background=ok?'#2EAD6B':'#D64545';t.style.display='block';clearTimeout(window._tt);window._tt=setTimeout(()=>t.style.display='none',2200)}
function badge(el,v){el.className='badge '+(v.result=='VERIFIED'?'ok':v.result=='POTENTIAL DISCREPANCY'?'warn':'wait');
 el.innerHTML=(v.result=='VERIFIED'?'&#10003; ':v.result=='POTENTIAL DISCREPANCY'?'&#9888; ':'')+v.result+'<div style="font-size:15px;font-weight:400">'+v.message+'</div>'}
const J={'Content-Type':'application/json'};
"""

HEAD = '<!doctype html><html><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1"><title>SyncCart</title><style>__CSS__</style></head><body><div id=toast></div>'

LOGIN = HEAD + r"""
<header><b>SyncCart</b><span>See. Sync. Verify.</span></header>
<main>
<div class=card id=resume style="display:none"><h3>WELCOME BACK</h3>
 <div class=muted>You have an open shopping session</div><div class=code id=rc></div>
 <div class=row><button class=go onclick="cont()">Continue shopping</button><button class=ghost onclick="forget()">Start over</button></div></div>
<div class=card id=login><h3>START SHOPPING</h3>
 <p>Scan items with your phone, pay, and walk out. You get a temporary code and QR. We never use face recognition.</p>
 <input id=name placeholder="Nickname (optional)" maxlength=20 autocomplete=off>
 <button class=go style="width:100%" onclick="start()">Start shopping</button></div>
</main>
<script>__COMMON__
let saved=null;try{saved=localStorage.getItem('synccart_session')}catch(e){}
if(saved){fetch('/api/state/'+saved).then(r=>r.ok?r.json():null).then(d=>{if(d&&!d.paid){document.getElementById('rc').textContent=saved;document.getElementById('resume').style.display='block'}else{try{localStorage.removeItem('synccart_session')}catch(e){}}})}
function cont(){location.href='/shop?session='+saved}
function forget(){try{localStorage.removeItem('synccart_session')}catch(e){};document.getElementById('resume').style.display='none'}
async function start(){const r=await fetch('/api/login',{method:'POST',headers:J,body:JSON.stringify({name:document.getElementById('name').value})});
 const d=await r.json();try{localStorage.setItem('synccart_session',d.session_id)}catch(e){};location.href='/shop?session='+d.session_id}
</script></body></html>"""

SHOP = HEAD + LIB + r"""
<header><b>SyncCart</b><span>Scan and Go</span><small id=codeh></small></header>
<main>
<div class=card><h3>YOUR TEMPORARY CODE</h3><div class=code id=code>-</div><div class=muted id=left></div></div>
<div class=card id=shopcard><h3>SCAN AN ITEM</h3>
 <div id=readerbox style="display:none"><div id=reader></div><div class=row><button class=ghost onclick="stopScan()">Close camera</button></div></div>
 <div class=row><button id=scanbtn class=alt onclick="startScan()">Scan item QR / barcode</button></div>
 <div class=muted style="margin-top:6px">No camera? Tap an item instead:</div><div id=prods class=row></div></div>
<div class=card><h3>YOUR CART</h3><div id=cart></div>
 <div class=row><button class=go id=paybtn onclick="pay()">Pay now</button></div></div>
<div class=card id=exitcard style="display:none"><h3>EXIT CODE</h3>
 <div class=muted>Show this QR at the exit gate.</div><img class=qr id=exitqr><div class=muted id=exiturl style="word-break:break-all"></div></div>
</main>
<script>__COMMON__
const sid=new URLSearchParams(location.search).get('session');
let scanner=null,lastCode='',lastTime=0,paid=false;
async function init(){if(!sid){location.href='/';return}
 const r=await fetch('/api/state/'+sid);if(!r.ok){try{localStorage.removeItem('synccart_session')}catch(e){};location.href='/';return}
 document.getElementById('code').textContent=sid;document.getElementById('codeh').textContent=sid;
 const ps=await (await fetch('/api/products')).json();const box=document.getElementById('prods');
 ps.forEach(p=>{const b=document.createElement('button');b.className='ghost';b.textContent=p;b.onclick=()=>scanProduct(p,1);box.appendChild(b)});
 if(!ps.length)box.innerHTML='<span class=muted>Products are loading, refresh in a moment.</span>';
 refresh();setInterval(refresh,4000)}
async function scanProduct(p,q){const r=await fetch('/api/scan/'+sid,{method:'POST',headers:J,body:JSON.stringify({product:p,qty:q})});
 if(r.ok){toast((q>0?'Added ':'Removed ')+p,true);refresh()}else{toast('Could not update the cart',false)}}
async function onCode(text){const now=Date.now();if(text===lastCode&&now-lastTime<2500)return;lastCode=text;lastTime=now;
 const r=await fetch('/api/scan_code/'+sid,{method:'POST',headers:J,body:JSON.stringify({code:text})});
 if(r.ok){const d=await r.json();toast('Added '+d.product,true);refresh()}else if(r.status==409){toast('Already paid',false)}else{toast('Unknown code',false)}}
async function startScan(){document.getElementById('readerbox').style.display='block';scanner=new Html5Qrcode('reader');
 try{await scanner.start({facingMode:'environment'},{fps:10,qrbox:{width:240,height:240}},onCode,()=>{})}
 catch(e){toast('Camera blocked. Allow camera access (needs https). Use the buttons instead.',false);stopScan()}}
async function stopScan(){if(scanner){try{await scanner.stop()}catch(e){}scanner=null}document.getElementById('readerbox').style.display='none'}
async function pay(){await stopScan();const r=await fetch('/api/pay/'+sid,{method:'POST'});if(r.ok)toast('Payment complete',true);refresh()}
async function refresh(){const r=await fetch('/api/state/'+sid);if(!r.ok){toast('Code expired',false);return}const s=await r.json();paid=s.paid;
 document.getElementById('left').textContent='Valid for about '+Math.ceil(s.seconds_left/60)+' more minutes';
 const c=document.getElementById('cart');c.innerHTML='';const k=Object.keys(s.cart);
 if(!k.length)c.innerHTML='<span class=muted>(empty)</span>';
 k.forEach(p=>{const sp=document.createElement('span');sp.className='chip';sp.append(p+' x'+s.cart[p]);
  if(!s.paid){const m=document.createElement('button');m.textContent='-';m.onclick=()=>scanProduct(p,-1);sp.appendChild(m)}c.appendChild(sp)});
 const pb=document.getElementById('paybtn');pb.disabled=s.paid||!k.length;pb.textContent=s.paid?'Paid':'Pay now';
 document.getElementById('shopcard').style.display=s.paid?'none':'block';
 if(s.paid){document.getElementById('exitcard').style.display='block';
  document.getElementById('exitqr').src='/api/qr?data='+encodeURIComponent(s.exit_url);document.getElementById('exiturl').textContent=s.exit_url}}
init();
</script></body></html>"""

EXIT = HEAD + r"""
<header><b>SyncCart</b><span>Exit verification</span></header>
<main><div class=card><h3>RESULT</h3><div id=res class="badge wait">Checking...</div>
<div id=detail class=muted style="margin-top:12px"></div>
<div class=row><button onclick="check()">Check again</button></div></div></main>
<script>__COMMON__
const sid=new URLSearchParams(location.search).get('session');
async function check(){if(!sid){document.getElementById('res').textContent='No customer code';return}
 const r=await fetch('/api/verify/'+sid);if(!r.ok){document.getElementById('res').textContent='Unknown or expired code';return}
 const v=await r.json();badge(document.getElementById('res'),v);
 let d='Code '+v.session_id+'<br>Physical (camera): '+chips(v.physical)+'<br>Paid cart: '+chips(v.digital);
 if(v.result=='POTENTIAL DISCREPANCY'){d+='<br>Not in paid cart: '+chips(v.not_in_paid_cart)+'<br>Paid but not seen: '+chips(v.paid_but_not_seen)+'<br><br>A staff member will do a quick secondary check. This is not an accusation.'}
 document.getElementById('detail').innerHTML=d}
check();
</script></body></html>"""

GATE = HEAD + LIB + r"""
<header><b>SyncCart</b><span>Exit gate scanner</span></header>
<main>
<div class=card><h3>SCAN THE CUSTOMER'S EXIT QR</h3><div id=reader></div>
 <div class=row><button onclick="startScan()">Start camera</button></div>
 <div class=muted>Or type the customer code:</div>
 <div class=row><input id=manual placeholder="e.g. K7P2QX" style="max-width:220px;text-transform:uppercase"><button onclick="lookup(document.getElementById('manual').value)">Check</button></div></div>
<div class=card><h3>RESULT</h3><div id=res class="badge wait">Waiting for a scan</div><div id=detail class=muted style="margin-top:12px"></div></div>
</main>
<script>__COMMON__
let scanner=null,last='',lastT=0;
function extract(t){const m=/session=([A-Za-z0-9]+)/.exec(t);return (m?m[1]:t).trim().toUpperCase()}
async function lookup(t){const code=extract(t);if(!code)return;const now=Date.now();if(code===last&&now-lastT<3000)return;last=code;lastT=now;
 const r=await fetch('/api/verify/'+code);const el=document.getElementById('res');
 if(!r.ok){el.className='badge wait';el.textContent='Unknown or expired code';return}
 const v=await r.json();badge(el,v);
 let d='Code '+v.session_id+'<br>Physical (camera): '+chips(v.physical)+'<br>Paid cart: '+chips(v.digital);
 if(v.result=='POTENTIAL DISCREPANCY'){d+='<br>Not in paid cart: '+chips(v.not_in_paid_cart)+'<br>Paid but not seen: '+chips(v.paid_but_not_seen)+'<br><br>Please do a friendly secondary check.'}
 document.getElementById('detail').innerHTML=d}
async function startScan(){if(scanner)return;scanner=new Html5Qrcode('reader');
 try{await scanner.start({facingMode:'environment'},{fps:10,qrbox:{width:240,height:240}},lookup,()=>{})}catch(e){toast('Camera blocked (needs https). Type the code instead.',false);scanner=null}}
</script></body></html>"""

LABELS = HEAD + r"""
<header><b>SyncCart</b><span>Product QR labels</span><small>Print and stick one on each product</small></header>
<main><div id=grid class=row></div></main>
<script>__COMMON__
fetch('/api/products').then(r=>r.json()).then(ps=>{const g=document.getElementById('grid');
 if(!ps.length)g.innerHTML='<span class=muted>No products yet. Start the camera agent (or use mock mode) first.</span>';
 ps.forEach(p=>{const d=document.createElement('div');d.className='card';d.style.textAlign='center';
  d.innerHTML='<img class=qr src="/api/qr?data='+encodeURIComponent('SC:'+p)+'"><div style="font-weight:800;font-size:20px;margin-top:6px">'+p+'</div>';g.appendChild(d)})});
</script></body></html>"""

STAFF = HEAD + r"""
<header><b>SyncCart</b><span>Staff screen</span><small><a id=lg href="/gate">Exit gate scanner</a> | <a id=ll href="/labels">Product QR labels</a></small></header>
<main><div class=grid>
<div><div class=card><h3>LIVE CAMERA (SNAPSHOT)</h3><img class=live id=live></div></div>
<div>
 <div class=card><h3>ACTIVE CUSTOMER</h3><div id=who class=muted>No customer yet. Customers open the site link and press Start shopping.</div></div>
 <div class=card><h3>PHYSICAL (CAMERA)</h3><div id=phys class=muted>-</div><div id=events class=muted style="margin-top:8px"></div></div>
 <div class=card><h3>PAID CART</h3><div id=cart class=muted>-</div></div>
 <div class=card><h3>LIVE VERIFICATION</h3><div id=res class="badge wait">Waiting for a customer</div></div>
 <div class=card><h3>RECENT CUSTOMERS</h3><div id=recent class=muted>-</div></div>
</div></div></main>
<script>__COMMON__
const PIN=new URLSearchParams(location.search).get('pin')||'';
document.getElementById('lg').href='/gate?pin='+encodeURIComponent(PIN);document.getElementById('ll').href='/labels?pin='+encodeURIComponent(PIN);
async function poll(){const a=await (await fetch('/api/active?pin='+encodeURIComponent(PIN))).json();
 document.getElementById('recent').innerHTML=a.recent.length?a.recent.map(r=>r.session_id+' ('+r.name+') '+(r.paid?'paid':'shopping')+', '+r.items+' items').join('<br>'):'-';
 if(!a.session_id)return;const sid=a.session_id;
 const s=await (await fetch('/api/state/'+sid)).json();const v=await (await fetch('/api/verify/'+sid)).json();
 document.getElementById('who').innerHTML='<span class=code style="font-size:26px">'+sid+'</span> '+s.name;
 document.getElementById('phys').innerHTML=chips(s.physical);
 document.getElementById('events').innerHTML=s.events.slice().reverse().map(e=>e.time+' '+e.product+' '+e.event).join('<br>');
 document.getElementById('cart').innerHTML=chips(s.cart)+'<div class=muted>'+(s.paid?'Paid':'Not paid yet')+'</div>';
 badge(document.getElementById('res'),v)}
setInterval(poll,1000);poll();
setInterval(()=>{document.getElementById('live').src='/frame.jpg?pin='+encodeURIComponent(PIN)+'&t='+Date.now()},500);
</script></body></html>"""


def page(html):
    return HTMLResponse(html.replace("__COMMON__", COMMON_JS).replace("__CSS__", CSS))


def staff_page(html, pin):
    if not staff_ok(pin):
        return HTMLResponse("Staff PIN required. Add ?pin=YOURPIN to the link.", status_code=401)
    return page(html)


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
