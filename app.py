"""Sync Cart — redesigned responsive customer web app + staff-only verification screens.
Keep core.py from the existing repository in the same folder. Payment is DEMO ONLY.
"""
import hashlib, hmac, io, json, os, secrets, threading, time
from contextlib import asynccontextmanager
import qrcode
from PIL import Image, ImageDraw
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel
from core import PickTracker, Store

MODE=os.getenv('SYNCCART_MODE','server')
API_KEY=os.getenv('SYNCCART_KEY','')
STAFF_PIN=os.getenv('SYNCCART_PIN','')
MOCK=os.getenv('SYNCCART_MOCK','1')=='1'
MODEL_PATH=os.getenv('SYNCCART_MODEL','best.pt')
CAMERA=int(os.getenv('SYNCCART_CAMERA','0'))
PORT=int(os.getenv('PORT',os.getenv('SYNCCART_PORT','8000')))
PUBLIC_URL=os.getenv('SYNCCART_PUBLIC_URL')
TTL_HOURS=float(os.getenv('SYNCCART_TTL_HOURS','3'))
CODE_MAP=json.loads(os.getenv('SYNCCART_CODES','{"CHOC001":"Chocolate","NOTE001":"Notebook","BOTTLE001":"Bottle"}'))
SECRET=(os.getenv('SYNCCART_SECRET') or API_KEY or secrets.token_hex(24)).encode()
store=Store(ttl_hours=TTL_HOURS); tracker=PickTracker(); lock=threading.Lock(); latest_jpeg=None
PRODUCTS=['Chocolate','Notebook','Bottle'] if MOCK else []
PRICES={'Chocolate':40,'Notebook':80,'Bottle':120,'Milk':35,'Bread':45,'Apples':60,'Coffee':150,'Biscuits':30}

class LoginBody(BaseModel): name:str=''
class ScanBody(BaseModel): product:str; qty:int=1
class CodeBody(BaseModel): code:str
class SimBody(BaseModel): product:str; event:str
class ProductsBody(BaseModel): products:list[str]

def base(request):
    if PUBLIC_URL: return PUBLIC_URL.rstrip('/')
    if MODE=='server': return f"{request.headers.get('x-forwarded-proto',request.url.scheme)}://{request.headers.get('host')}"
    return f'http://127.0.0.1:{PORT}'
def staff_ok(pin): return (not STAFF_PIN) or pin==STAFF_PIN
def check_key(request):
    if API_KEY and request.headers.get('x-key')!=API_KEY: raise HTTPException(401,'Bad key')
def get_session(sid):
    sid=(sid or '').upper()
    if not store.exists(sid): raise HTTPException(404,'Unknown or expired code')
    return sid
def resolve_code(code):
    c=(code or '').strip(); c=c[3:].strip() if c.upper().startswith('SC:') else c
    unit=None
    if '#' in c: c,unit=c.rsplit('#',1); unit=unit.strip() or None
    for p in PRODUCTS:
        if p.lower()==c.lower(): return p,unit
    mapped=CODE_MAP.get(c)
    if mapped and (not PRODUCTS or mapped in PRODUCTS): return mapped,unit or c
    return None,None
def fmt(d): return ','.join(f'{p}:{q}' for p,q in sorted(d.items()))
def sig(sid,cart,phys): return hmac.new(SECRET,f'{sid}|{cart}|{phys}'.encode(),hashlib.sha256).hexdigest()[:12]
def exit_payload(sid):
    cart,phys=fmt(store.state(sid)['cart']),fmt(store.physical(sid)); return f'SCX|{sid}|{cart}|{phys}|{sig(sid,cart,phys)}'
def parse_exit(code):
    t=(code or '').strip()
    if t.startswith('SCX|'):
        parts=t.split('|')
        if len(parts)==5:
            _,sid,cart,phys,signature=parts; return sid.upper(),hmac.compare_digest(signature,sig(sid,cart,phys)),cart
    if 'session=' in t: t=t.split('session=',1)[1].split('&')[0]
    return t.upper(),None,None
def placeholder(text='Waiting for camera agent'):
    im=Image.new('RGB',(640,420),(12,20,37)); ImageDraw.Draw(im).text((24,200),text,fill=(230,245,240)); b=io.BytesIO(); im.save(b,format='JPEG'); return b.getvalue()

@asynccontextmanager
async def lifespan(app):
    global latest_jpeg
    latest_jpeg=placeholder('DEMO MODE — camera not connected')
    if MODE=='local' and not MOCK:
        from ultralytics import YOLO
        prod=YOLO(MODEL_PATH); person=YOLO('yolo11n.pt'); PRODUCTS[:]=list(prod.names.values())
        def loop():
            global latest_jpeg
            import cv2
            cap=cv2.VideoCapture(CAMERA)
            while True:
                ok,frame=cap.read()
                if not ok: time.sleep(.3); continue
                seen=set()
                for b in prod(frame,conf=.5,verbose=False)[0].boxes:
                    x1,y1,x2,y2=b.xyxy[0].tolist(); name=prod.names[int(b.cls)]; seen.add(name)
                    cv2.rectangle(frame,(int(x1),int(y1)),(int(x2),int(y2)),(38,220,150),2); cv2.putText(frame,name,(int(x1),int(y1)-5),cv2.FONT_HERSHEY_SIMPLEX,.6,(38,220,150),2)
                near=bool(person(frame,classes=[0],verbose=False)[0].boxes)
                with lock:
                    events=tracker.update(seen,near,time.time()); sid=store.active if store.active and store.exists(store.active) else None
                    if sid:
                        for name,event in events: store.add_event(sid,name,event)
                latest_jpeg=cv2.imencode('.jpg',frame)[1].tobytes()
        threading.Thread(target=loop,daemon=True).start()
    print(f'Sync Cart running on port {PORT} ({MODE} mode)'); yield
app=FastAPI(title='Sync Cart',lifespan=lifespan)

@app.get('/api/products')
def products(): return PRODUCTS or list(PRICES)
@app.post('/api/login')
def login(body:LoginBody,request:Request):
    with lock: sid=store.new_session(body.name); tracker.reset()
    return {'session_id':sid,'shop_url':f'{base(request)}/shop?session={sid}'}
@app.post('/api/scan/{sid}')
def scan(sid:str,body:ScanBody):
    sid=get_session(sid)
    if body.qty not in (-1,1): raise HTTPException(400,'Bad quantity')
    if PRODUCTS and body.product not in PRODUCTS: raise HTTPException(400,'Unknown product')
    with lock:
        if store.is_paid(sid): raise HTTPException(409,'Already paid')
        store.unscan(sid,body.product) if body.qty<0 else store.scan(sid,body.product,1)
    return {'ok':True}
@app.post('/api/scan_code/{sid}')
def scan_code(sid:str,body:CodeBody):
    sid=get_session(sid); product,unit=resolve_code(body.code)
    if not product: raise HTTPException(404,'Unknown product code')
    with lock:
        if store.is_paid(sid): raise HTTPException(409,'Already paid')
        if unit:
            result=store.scan_unit(sid,product,unit)
            if result=='dup': raise HTTPException(409,'You already scanned this exact item')
            if result=='taken': raise HTTPException(423,'This item is already in another cart')
        else: store.scan(sid,product,1)
    return {'product':product}
@app.post('/api/pay/{sid}')
def pay(sid:str):
    sid=get_session(sid)
    with lock: store.pay(sid)
    return {'ok':True,'demo':True,'message':'Demo checkout complete. No money was charged.'}
@app.get('/api/state/{sid}')
def state(sid:str,request:Request):
    sid=get_session(sid)
    with lock: d=store.state(sid); d['shelf']=tracker.statuses() if store.active==sid else {}
    d['shop_url']=f'{base(request)}/shop?session={sid}'; d['exit_url']=f'{base(request)}/exit?session={sid}'; d['exit_qr']=exit_payload(sid); return d
@app.get('/api/verify/{sid}')
def verify(sid:str):
    with lock: return store.verify(get_session(sid))
@app.post('/api/exit_scan')
def exit_scan(body:CodeBody):
    sid,signature_ok,items=parse_exit(body.code)
    if not store.exists(sid): raise HTTPException(404,'Unknown or expired code')
    with lock:
        v=store.verify(sid); v['closed']=False
        if v['paid']:
            v['closed']=True
            if store.active==sid: store.active=None; tracker.reset()
    v['signature_ok']=signature_ok; v['qr_items']=items; return v
@app.get('/api/active')
def active(pin:str=''):
    if not staff_ok(pin): raise HTTPException(401,'Staff PIN required')
    with lock:
        sid=store.active if store.active and store.exists(store.active) else None
        return {'session_id':sid,'name':store.sessions[sid]['name'] if sid else None,'recent':store.recent()}
@app.get('/api/qr')
def qr(data:str):
    if len(data)>300: raise HTTPException(400,'Too long')
    im=qrcode.make(data); b=io.BytesIO(); im.save(b,format='PNG'); return Response(b.getvalue(),media_type='image/png')
@app.get('/frame.jpg')
def frame(pin:str=''):
    if not staff_ok(pin): raise HTTPException(401,'Staff PIN required')
    return Response(latest_jpeg or placeholder(),media_type='image/jpeg',headers={'Cache-Control':'no-store'})
@app.post('/api/agent/products')
def agent_products(body:ProductsBody,request:Request):
    check_key(request); PRODUCTS[:]=body.products; return {'ok':True}
@app.post('/api/agent/frame')
async def agent_frame(request:Request):
    global latest_jpeg
    check_key(request); data=await request.body()
    if data: latest_jpeg=data
    return {'active':store.active}
@app.post('/api/agent/event')
def agent_event(body:SimBody,request:Request):
    check_key(request)
    with lock:
        sid=store.active if store.active and store.exists(store.active) else None
        if sid and body.event in ('PICKED','RETURNED'): store.add_event(sid,body.product,body.event); return {'active':sid,'recorded':True}
    return {'active':store.active,'recorded':False}
@app.post('/api/sim/{sid}')
def simulate(sid:str,body:SimBody):
    if not MOCK: raise HTTPException(403,'Simulation is only available in MOCK mode')
    with lock: store.add_event(get_session(sid),body.product,body.event)
    return {'ok':True}

CSS=r'''
:root{--bg:#080f1d;--panel:#111c2d;--panel2:#17263a;--line:#26384c;--text:#f3f7f6;--muted:#9aabba;--green:#31df9b;--green2:#0eaf78;--cyan:#63e7d0;--amber:#ffcb73;--red:#ff8585}*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:radial-gradient(ellipse at 78% 0%,#16352e 0,transparent 34%),var(--bg);color:var(--text);font:15px/1.5 Inter,ui-sans-serif,system-ui,-apple-system,Segoe UI,sans-serif}a{color:var(--green);text-decoration:none}header{height:74px;padding:0 clamp(18px,5vw,72px);display:flex;align-items:center;gap:14px;border-bottom:1px solid #ffffff10;background:#080f1ddf;backdrop-filter:blur(16px);position:sticky;top:0;z-index:10}header b,.brand{font-size:21px;font-weight:850;letter-spacing:-.8px}header span{color:var(--green);font-size:12px;font-weight:800;letter-spacing:1.2px;text-transform:uppercase}header small{margin-left:auto;color:var(--muted)}main{width:min(1160px,100%);margin:auto;padding:34px 22px 100px}.hero{display:grid;grid-template-columns:1.05fr .95fr;gap:35px;align-items:center;min-height:430px;padding:22px 0 38px}.eyebrow{display:inline-flex;align-items:center;gap:9px;color:var(--green);font-size:12px;font-weight:850;letter-spacing:1.8px;text-transform:uppercase}.dot{width:7px;height:7px;border-radius:50%;background:var(--green);box-shadow:0 0 16px var(--green)}h1{font-size:clamp(42px,6vw,72px);line-height:.99;letter-spacing:-3.5px;margin:18px 0 20px}h1 em{font-style:normal;color:var(--green)}h2{font-size:clamp(25px,3vw,36px);letter-spacing:-1.2px;margin:0 0 8px}h3{font-size:15px;margin:0 0 12px}.lead{color:#b9c7d4;font-size:17px;max-width:550px}.actions,.row{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-top:22px}.btn,button{display:inline-flex;align-items:center;justify-content:center;gap:8px;border:1px solid transparent;border-radius:13px;padding:12px 17px;background:var(--green);color:#062016;font:750 14px inherit;font-weight:800;cursor:pointer;transition:transform .18s,filter .18s}.btn:hover,button:hover{filter:brightness(1.08);transform:translateY(-1px)}.btn.secondary,button.ghost{background:#ffffff08;color:var(--text);border-color:var(--line)}button:disabled{opacity:.45;cursor:not-allowed;transform:none}.hero-art{min-height:330px;border:1px solid #ffffff16;border-radius:28px;overflow:hidden;position:relative;background:linear-gradient(140deg,#193c35,#101b2c 64%);padding:24px;display:flex;align-items:center;justify-content:center}.hero-art:before{content:'';position:absolute;width:300px;height:300px;border-radius:50%;background:#30dfa033;filter:blur(28px);right:-60px;top:-80px}.basket{position:relative;width:min(100%,400px);background:#0b1422dd;border:1px solid #ffffff22;border-radius:22px;padding:19px;box-shadow:0 30px 70px #0005;transform:rotate(-2deg)}.basket-top{display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid var(--line);padding-bottom:14px}.basket-top strong{font-size:15px}.pill{background:#103e30;color:var(--green);border:1px solid #1c7956;border-radius:99px;padding:5px 10px;font-size:11px;font-weight:800}.basket-item{display:flex;align-items:center;gap:12px;padding:13px 0;border-bottom:1px solid #ffffff0d}.emoji{width:47px;height:47px;display:grid;place-items:center;background:#ffffff09;border:1px solid #ffffff0b;border-radius:13px;font-size:25px}.basket-item strong{display:block;font-size:13px}.basket-item small{color:var(--muted)}.price{margin-left:auto;font-weight:800}.basket-total{display:flex;justify-content:space-between;padding-top:15px;font-weight:800}.features{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:18px 0 42px}.feature,.card{background:linear-gradient(145deg,#142237,#101a2a);border:1px solid var(--line);border-radius:20px;padding:20px}.feature .emoji{margin-bottom:13px}.feature strong{display:block}.feature p,.muted{color:var(--muted);font-size:13px}.grid{display:grid;grid-template-columns:minmax(0,1.35fr) minmax(280px,.8fr);gap:18px}.stack{display:grid;gap:16px;align-content:start}.section-head{display:flex;align-items:center;justify-content:space-between;gap:12px;margin:0 0 16px}.section-head small{color:var(--muted)}.input{width:100%;padding:13px 14px;border-radius:12px;border:1px solid var(--line);background:#081321;color:var(--text);font:inherit;margin:7px 0 14px;outline:none}.input:focus{border-color:var(--green)}.product-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}.product{background:#0c1726;border:1px solid var(--line);border-radius:16px;padding:13px;min-width:0}.product-visual{height:104px;border-radius:12px;background:linear-gradient(135deg,#1c3540,#132235);display:grid;place-items:center;font-size:45px;margin-bottom:12px}.product strong{display:block;overflow-wrap:anywhere}.product small{color:var(--muted)}.product-foot{display:flex;align-items:center;justify-content:space-between;gap:6px;margin-top:10px}.product-foot button{padding:8px 10px;border-radius:10px}.cart-row{display:flex;align-items:center;gap:10px;padding:12px 0;border-bottom:1px solid #ffffff10}.cart-row:last-child{border:0}.cart-row .emoji{width:40px;height:40px;font-size:21px}.cart-row strong{display:block}.cart-row small{color:var(--muted)}.qty{display:flex;align-items:center;gap:8px;margin-left:auto}.qty button{width:27px;height:27px;padding:0;border-radius:9px}.total-line{display:flex;justify-content:space-between;padding:7px 0;color:var(--muted)}.total-line.final{font-size:20px;color:var(--text);font-weight:850;border-top:1px solid var(--line);padding-top:14px;margin-top:6px}.full{width:100%;margin-top:13px}.qr-wrap{text-align:center;background:#fff;border-radius:18px;padding:15px;display:inline-block;margin:12px auto}.qr{width:180px;max-width:100%;display:block}.center{text-align:center}.badge{border:1px solid var(--line);border-radius:16px;padding:17px;font-size:20px;font-weight:850;background:#162338}.badge.ok{background:#10382d;border-color:#237957;color:#6df0b8}.badge.warn{background:#3b2a16;border-color:#8a6227;color:#ffd18a}.badge.wait{color:#c5d2df}.chip{display:inline-flex;align-items:center;background:#091421;border:1px solid var(--line);border-radius:99px;padding:6px 10px;margin:3px;font-size:12px}.reader{width:100%;overflow:hidden;border-radius:15px}.hide{display:none!important}#toast{position:fixed;left:50%;bottom:22px;transform:translateX(-50%);padding:12px 18px;border-radius:12px;font-weight:800;display:none;z-index:50;box-shadow:0 10px 30px #0005;max-width:90vw}.bottom-nav{display:none}.small-note{font-size:12px;color:var(--muted)}.statusline{display:flex;gap:8px;align-items:center;color:var(--muted);font-size:12px}.statusline i{display:inline-block;width:7px;height:7px;background:var(--green);border-radius:50%}.budgetbar{height:7px;background:#24364a;border-radius:99px;overflow:hidden;margin:12px 0}.budgetbar span{display:block;width:42%;height:100%;background:linear-gradient(90deg,var(--green2),var(--cyan));border-radius:99px}@media(max-width:820px){.hero{grid-template-columns:1fr;gap:18px;padding-top:32px}.hero-art{min-height:290px}.features{grid-template-columns:repeat(2,1fr)}.grid{grid-template-columns:1fr}.product-grid{grid-template-columns:repeat(2,minmax(0,1fr))}header{height:64px}.bottom-nav{display:flex;position:fixed;bottom:0;left:0;right:0;background:#08101fee;backdrop-filter:blur(15px);border-top:1px solid var(--line);padding:10px 8px calc(10px + env(safe-area-inset-bottom));justify-content:space-around;z-index:12}.bottom-nav a{color:var(--muted);font-size:11px;text-align:center}.bottom-nav b{display:block;color:var(--green);font-size:17px}.hero h1{letter-spacing:-2px}}@media(max-width:420px){main{padding:22px 14px 105px}.features{gap:8px}.feature,.card{padding:15px}.product-visual{height:82px;font-size:37px}.product{padding:10px}.hero-art{padding:14px}.basket{padding:14px}}
'''
COMMON=r'''const J={'Content-Type':'application/json'};function toast(m,ok=true){let t=document.getElementById('toast');t.textContent=m;t.style.background=ok?'#1b7656':'#8d333d';t.style.display='block';clearTimeout(window._tt);window._tt=setTimeout(()=>t.style.display='none',2600)}function chips(o){let k=Object.keys(o||{});return k.length?k.map(p=>'<span class="chip">'+p+' × '+o[p]+'</span>').join(''):'<span class="muted">None recorded</span>'}function badge(el,v){el.className='badge '+(v.result==='VERIFIED'?'ok':v.result==='POTENTIAL DISCREPANCY'?'warn':'wait');el.innerHTML=(v.result==='VERIFIED'?'✓ ':v.result==='POTENTIAL DISCREPANCY'?'⚠ ':'')+v.result+'<div style="font-size:13px;font-weight:500;margin-top:5px">'+v.message+'</div>'}const icons={'Chocolate':'🍫','Notebook':'📓','Bottle':'🧴','Milk':'🥛','Bread':'🍞','Apples':'🍎','Coffee':'☕','Biscuits':'🍪'};const prices={'Chocolate':40,'Notebook':80,'Bottle':120,'Milk':35,'Bread':45,'Apples':60,'Coffee':150,'Biscuits':30};'''
HEAD='<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><meta name="theme-color" content="#080f1d"><title>Sync Cart — See. Sync. Verify.</title><style>__CSS__</style></head><body><div id="toast"></div>'
LIB='<script src="https://cdn.jsdelivr.net/npm/html5-qrcode@2.3.8/html5-qrcode.min.js"></script>'
LOGIN=HEAD+r'''<header><b>◈ Sync Cart</b><span>See. Sync. Verify.</span><small>Smart shopping, made simple</small></header><main><section class="hero"><div><div class="eyebrow"><i class="dot"></i> A smarter way to shop</div><h1>Smart shopping.<br><em>In sync.</em></h1><p class="lead">Your basket, your budget, your checkout — all in one smooth experience. Scan products as you shop and get a simple verification at the exit.</p><div class="actions"><button onclick="document.getElementById('start').scrollIntoView({behavior:'smooth'})">Start shopping ↗</button><a class="btn secondary" href="#how">How it works</a></div><p class="small-note">Privacy-first demo · No facial recognition</p></div><div class="hero-art"><div class="basket"><div class="basket-top"><strong>✦ Your smart basket</strong><span class="pill">LIVE CART</span></div><div class="basket-item"><div class="emoji">🍫</div><div><strong>Chocolate bar</strong><small>Sweet little treat</small></div><span class="price">₹40</span></div><div class="basket-item"><div class="emoji">📓</div><div><strong>Notebook</strong><small>Everyday essentials</small></div><span class="price">₹80</span></div><div class="basket-item"><div class="emoji">🧴</div><div><strong>Water bottle</strong><small>Stay refreshed</small></div><span class="price">₹120</span></div><div class="basket-total"><span>Total</span><span>₹240.00</span></div><div class="statusline" style="margin-top:13px"><i></i> Ready for a smooth checkout</div></div></div></section><section class="features" id="how"><div class="feature"><div class="emoji">▦</div><strong>Scan & add</strong><p>Scan a product QR or add a demo item.</p></div><div class="feature"><div class="emoji">◈</div><strong>Know your total</strong><p>Keep your basket and spend in view.</p></div><div class="feature"><div class="emoji">⌁</div><strong>Quick checkout</strong><p>Review your items before demo payment.</p></div><div class="feature"><div class="emoji">✓</div><strong>Exit verification</strong><p>Show your session QR for a final check.</p></div></section><section class="grid" id="start"><div class="card"><div class="eyebrow">WELCOME IN</div><h2>Let's get your basket ready.</h2><p class="muted">Enter a nickname if you like. We’ll create a temporary shopping session for this demo.</p><label for="name">Your name (optional)</label><input class="input" id="name" placeholder="e.g. Alex" maxlength="20" autocomplete="nickname"><button class="full" onclick="start()">Start shopping <span>→</span></button><p class="small-note">Your session is temporary. The checkout is a prototype and does not charge money.</p></div><div class="card"><h3>How Sync Cart works</h3><p class="muted">01 — Scan item QR codes to build your digital cart.</p><p class="muted">02 — Review items and complete the demo checkout.</p><p class="muted">03 — Show your exit QR to the verification screen.</p><div class="pill" style="display:inline-block;margin-top:7px">BUILT FOR A BETTER EXIT</div></div></section></main><script>__COMMON__
let saved=null;try{saved=localStorage.getItem('synccart_session')}catch(e){}if(saved){fetch('/api/state/'+saved).then(r=>r.ok?r.json():null).then(d=>{if(d&&!d.paid){let n=document.createElement('div');n.className='card';n.innerHTML='<h3>Continue your basket?</h3><p class="muted">You have a shopping session in progress.</p><button onclick="location.href=\'/shop?session='+saved+'\'">Continue shopping →</button>';document.querySelector('#start').prepend(n)}else{localStorage.removeItem('synccart_session')}})}async function start(){let r=await fetch('/api/login',{method:'POST',headers:J,body:JSON.stringify({name:document.getElementById('name').value})});if(!r.ok){toast('Could not start a session',false);return}let d=await r.json();try{localStorage.setItem('synccart_session',d.session_id)}catch(e){}location.href='/shop?session='+d.session_id}</script></body></html>'''
SHOP=HEAD+LIB+r'''<header><b>◈ Sync Cart</b><span>Scan & Go</span><small id="hello">Your basket, in sync</small></header><main><div class="section-head"><div><div class="eyebrow">YOUR SHOPPING SESSION</div><h2>Build your basket.</h2></div><span class="pill" id="sessionpill">ACTIVE</span></div><div class="grid"><div class="stack"><div class="card"><div class="section-head"><h3>Find your favourites</h3><span class="muted">Demo catalogue</span></div><input class="input" id="search" placeholder="Search products..." oninput="renderProducts()"><div class="product-grid" id="prods"></div><div class="row"><button class="ghost" onclick="showScanner()">▦ Scan product QR</button></div><div id="readerbox" class="hide"><div id="reader" class="reader"></div><button class="ghost" onclick="stopScan()">Close scanner</button></div><p class="small-note">Camera scanning requires HTTPS and permission. You can also add items from this demo catalogue.</p></div><div class="card"><div class="section-head"><h3>Offers for you</h3><span class="pill">DEMO</span></div><div class="row" style="align-items:flex-start"><div class="emoji">✦</div><div><strong>Smart basket, smart spending</strong><p class="muted">Keep an eye on your total before checkout. Offers shown here are illustrative.</p></div></div><div class="card"><div class="section-head"><h3>Smart aisle route</h3><span class="pill">DEMO MAP</span></div><p class="muted">A suggested route based on example aisle locations. Update aisle data for your actual store.</p><div id="routebox" class="muted">Add items to build your route.</div></div></div><aside class="stack"><div class="card"><div class="section-head"><h3>Your cart</h3><span class="pill" id="count">0 ITEMS</span></div><div id="cart"><p class="muted">Your basket is waiting for its first item.</p></div><div class="total-line"><span>Subtotal</span><strong id="subtotal">₹0</strong></div><div class="total-line"><span>Demo discount</span><strong>₹0</strong></div><div class="total-line final"><span>Total</span><span id="total">₹0</span></div><div class="budgetbar"><span id="budgetbar"></span></div><div class="total-line"><span>Example budget ₹1,000</span><strong id="remaining">₹1,000 left</strong></div><button id="paybtn" class="full" onclick="pay()" disabled>Continue to checkout →</button><p class="small-note">Demo checkout only — no real payment is processed.</p></div><div class="card center"><h3>Exit pass</h3><p class="muted">Your session QR appears here after checkout.</p><div id="qrslot" class="muted">Complete checkout to generate your exit pass.</div><div class="code" id="code" style="font-size:18px;margin-top:8px"></div><a id="exitlink" class="btn secondary full hide" href="#">Open verification status</a></div></aside></div></main><nav class="bottom-nav"><a href="/"><b>⌂</b>Home</a><a href="#prods"><b>▦</b>Shop</a><a href="#cart"><b>◈</b>Basket</a><a href="#qrslot"><b>✓</b>Exit pass</a><a href="/profile"><b>◉</b>History</a></nav><script>__COMMON__
const sid=new URLSearchParams(location.search).get('session');let products=[],scanner=null,lastCode='',lastTime=0,current={};async function init(){if(!sid){location.href='/';return}let r=await fetch('/api/state/'+sid);if(!r.ok){location.href='/';return}let p=await fetch('/api/products');products=await p.json();renderProducts();refresh();setInterval(refresh,3000)}function renderProducts(){let q=(document.getElementById('search').value||'').toLowerCase(),box=document.getElementById('prods');box.innerHTML='';let list=products.filter(p=>p.toLowerCase().includes(q));if(!list.length){box.innerHTML='<p class="muted">No matching products.</p>';return}list.forEach(p=>{let d=document.createElement('div');d.className='product';d.innerHTML='<div class="product-visual">'+(icons[p]||'🛍️')+'</div><strong>'+p+'</strong><small>Everyday essential</small><div class="product-foot"><b>₹'+(prices[p]||50)+'</b><button onclick="scanProduct('+JSON.stringify(p)+',1)">＋ Add</button></div>';box.appendChild(d)})}async function scanProduct(p,q){let r=await fetch('/api/scan/'+sid,{method:'POST',headers:J,body:JSON.stringify({product:p,qty:q})});if(r.ok){toast((q>0?'Added ':'Removed ')+p);refresh()}else{let d=await r.json().catch(()=>({}));toast(d.detail||'Could not update cart',false)}}async function onCode(text){let now=Date.now();if(text===lastCode&&now-lastTime<2500)return;lastCode=text;lastTime=now;let r=await fetch('/api/scan_code/'+sid,{method:'POST',headers:J,body:JSON.stringify({code:text})});if(r.ok){let d=await r.json();toast('Added '+d.product);refresh()}else{let d=await r.json().catch(()=>({}));toast(d.detail||'Unknown product code',false)}}async function showScanner(){document.getElementById('readerbox').classList.remove('hide');if(scanner)return;scanner=new Html5Qrcode('reader');try{await scanner.start({facingMode:'environment'},{fps:10,qrbox:{width:230,height:230}},onCode,()=>{})}catch(e){toast('Allow camera access, or use Add buttons',false);scanner=null}}async function stopScan(){if(scanner){try{await scanner.stop()}catch(e){}scanner=null}document.getElementById('readerbox').classList.add('hide')}async function pay(){if(!Object.keys(current.cart||{}).length){toast('Add at least one item first',false);return}await stopScan();location.href='/checkout?session='+sid}function renderRoute(keys){const box=document.getElementById('routebox');if(!box)return;const aisle={'Apples':'Produce · Aisle 1','Milk':'Dairy · Aisle 2','Bread':'Bakery · Aisle 3','Chocolate':'Snacks · Aisle 4','Biscuits':'Snacks · Aisle 4','Coffee':'Beverages · Aisle 5','Bottle':'Household · Aisle 6','Notebook':'Stationery · Aisle 7'};const order=[];keys.forEach(p=>{const label=aisle[p]||'General · Ask store staff';if(!order.includes(label))order.push(label)});box.innerHTML=order.length?'<ol>'+order.map(x=>'<li style="padding:5px 0">'+x+'</li>').join('')+'</ol><p class="small-note">Illustrative aisle labels, not live indoor navigation.</p>':'Add items to build your route.'}async function refresh(){let r=await fetch('/api/state/'+sid);if(!r.ok)return;let s=await r.json();current=s;let c=document.getElementById('cart'),keys=Object.keys(s.cart||{});renderRoute(keys);c.innerHTML='';let count=0,total=0;if(!keys.length)c.innerHTML='<p class="muted">Your basket is waiting for its first item.</p>';keys.forEach(p=>{let q=s.cart[p];count+=q;total+=(prices[p]||50)*q;let row=document.createElement('div');row.className='cart-row';row.innerHTML='<div class="emoji">'+(icons[p]||'🛍️')+'</div><div><strong>'+p+'</strong><small>₹'+(prices[p]||50)+' each</small></div><div class="qty">'+(!s.paid?'<button class="ghost" onclick="scanProduct('+JSON.stringify(p)+',-1)">−</button>':'')+'<b>'+q+'</b>'+(!s.paid?'<button onclick="scanProduct('+JSON.stringify(p)+',1)">＋</button>':'')+'</div>';c.appendChild(row)});document.getElementById('count').textContent=count+' ITEM'+(count===1?'':'S');document.getElementById('subtotal').textContent='₹'+total;document.getElementById('total').textContent='₹'+total;document.getElementById('remaining').textContent='₹'+Math.max(0,1000-total)+' left';document.getElementById('budgetbar').style.width=Math.min(100,total/10)+'%';document.getElementById('paybtn').disabled=s.paid||count===0;document.getElementById('paybtn').textContent=s.paid?'Checkout complete':'Continue to checkout →';document.getElementById('hello').textContent=s.name?'Hi, '+s.name:'Your basket, in sync';if(s.paid){document.getElementById('sessionpill').textContent='CHECKOUT COMPLETE';document.getElementById('qrslot').innerHTML='<div class="qr-wrap"><img class="qr" src="/api/qr?data='+encodeURIComponent(s.exit_qr)+'"></div><p class="muted">Show this QR at the exit.</p>';document.getElementById('code').textContent=s.session_id;let a=document.getElementById('exitlink');a.href='/exit?session='+sid;a.classList.remove('hide')}}init();</script></body></html>'''
CHECKOUT=HEAD+r'''<header><b>◈ Sync Cart</b><span>Checkout</span></header><main style="max-width:720px"><div class="eyebrow">ALMOST THERE</div><h2>Review your basket.</h2><p class="muted">This prototype simulates checkout. It does not connect to a bank, UPI app, or payment gateway.</p><div class="card"><div id="summary"></div><div class="total-line final"><span>Total</span><span id="total">₹0</span></div><button id="confirm" class="full" onclick="confirmPay()">Confirm demo checkout →</button><a class="btn secondary full" id="back" href="#">Back to basket</a></div><div id="done" class="card center hide"><div class="emoji" style="margin:auto">✓</div><h2>You're checked out!</h2><p class="muted">Your exit pass is ready. Show the QR at the verification point.</p><div id="qr"></div><p class="code" id="code"></p><a id="verify" class="btn full" href="#">View verification status →</a></div></main><script>__COMMON__
const sid=new URLSearchParams(location.search).get('session');let state=null;async function init(){if(!sid){location.href='/';return}let r=await fetch('/api/state/'+sid);if(!r.ok){location.href='/';return}state=await r.json();document.getElementById('back').href='/shop?session='+sid;let total=0,box=document.getElementById('summary');box.innerHTML='';Object.keys(state.cart||{}).forEach(p=>{let q=state.cart[p],price=prices[p]||50;total+=q*price;let row=document.createElement('div');row.className='cart-row';row.innerHTML='<div class="emoji">'+(icons[p]||'🛍️')+'</div><div><strong>'+p+'</strong><small>'+q+' × ₹'+price+'</small></div><b style="margin-left:auto">₹'+(q*price)+'</b>';box.appendChild(row)});document.getElementById('total').textContent='₹'+total;if(state.paid)showDone()}async function confirmPay(){let r=await fetch('/api/pay/'+sid,{method:'POST'});if(!r.ok){toast('Checkout failed',false);return}state=await (await fetch('/api/state/'+sid)).json();showDone()}function showDone(){try{let h=JSON.parse(localStorage.getItem('synccart_history')||'[]');if(!h.some(x=>x.session_id===state.session_id)){let total=Object.keys(state.cart||{}).reduce((n,p)=>n+(prices[p]||50)*state.cart[p],0);h.unshift({session_id:state.session_id,date:new Date().toLocaleString(),items:state.cart,total:total});localStorage.setItem('synccart_history',JSON.stringify(h.slice(0,20)))}}catch(e){}document.getElementById('confirm').classList.add('hide');document.getElementById('done').classList.remove('hide');document.getElementById('qr').innerHTML='<div class="qr-wrap"><img class="qr" src="/api/qr?data='+encodeURIComponent(state.exit_qr)+'"></div>';document.getElementById('code').textContent=state.session_id;document.getElementById('verify').href='/exit?session='+sid}init();</script></body></html>'''
EXIT=HEAD+r'''<header><b>◈ Sync Cart</b><span>Exit verification</span></header><main style="max-width:700px"><div class="card center"><div class="eyebrow">FINAL STEP</div><div style="font-size:46px;margin:15px">⌁</div><h2>Exit verification</h2><p class="muted">Your shopping session is checked against the available physical-item record and paid cart.</p><div id="res" class="badge wait">Checking…</div><div id="detail" style="margin-top:18px;text-align:left"></div><button class="full" onclick="check()">Refresh status</button><a href="/" class="btn secondary full">Back to home</a></div></main><script>__COMMON__
const sid=new URLSearchParams(location.search).get('session');async function check(){if(!sid){document.getElementById('res').textContent='No session code';return}let r=await fetch('/api/verify/'+sid);if(!r.ok){document.getElementById('res').textContent='Unknown or expired code';return}let v=await r.json();badge(document.getElementById('res'),v);let d='<p><b>Session</b> · '+v.session_id+'</p><p><b>Paid cart</b><br>'+chips(v.digital)+'</p><p><b>Physical record</b><br>'+chips(v.physical)+'</p>';if(v.result==='POTENTIAL DISCREPANCY')d+='<p class="muted">A staff member may perform a quick secondary check. This is not an accusation; camera-based records can be imperfect.</p>';document.getElementById('detail').innerHTML=d}check();setInterval(check,5000)</script></body></html>'''
GATE=HEAD+LIB+r'''<header><b>◈ Sync Cart</b><span>Staff · Exit gate</span><small>Authorized staff only</small></header><main><div class="grid"><div class="card"><h2>Scan exit pass</h2><p class="muted">Use the customer’s exit QR or enter their session code.</p><div id="reader" class="reader"></div><button onclick="startScan()">Start camera</button><input class="input" id="manual" placeholder="Session code"><button class="ghost" onclick="lookup(document.getElementById('manual').value)">Check code</button></div><div class="card"><h3>Verification result</h3><div id="res" class="badge wait">Waiting for a scan</div><div id="detail"></div></div></div></main><script>__COMMON__
let scanner=null,last='',lastT=0;async function lookup(code){code=(code||'').trim();if(!code)return;let now=Date.now();if(code===last&&now-lastT<2000)return;last=code;lastT=now;let r=await fetch('/api/exit_scan',{method:'POST',headers:J,body:JSON.stringify({code})});if(!r.ok){toast('Unknown or expired code',false);return}let v=await r.json();badge(document.getElementById('res'),v);document.getElementById('detail').innerHTML='<p><b>Session</b> '+v.session_id+'</p><p><b>Physical</b><br>'+chips(v.physical)+'</p><p><b>Paid cart</b><br>'+chips(v.digital)+'</p>'+(v.result==='POTENTIAL DISCREPANCY'?'<p class="muted">Please perform a friendly secondary check.</p>':'')}async function startScan(){if(scanner)return;scanner=new Html5Qrcode('reader');try{await scanner.start({facingMode:'environment'},{fps:10,qrbox:{width:230,height:230}},lookup,()=>{})}catch(e){toast('Camera needs HTTPS and permission. Enter code instead.',false);scanner=null}}</script></body></html>'''
STAFF=HEAD+r'''<header><b>◈ Sync Cart</b><span>Staff dashboard</span><small><a id="gate" href="/gate">Exit scanner</a> · <a id="labels" href="/labels">QR labels</a></small></header><main><div class="grid"><div class="card"><h3>CAMERA SNAPSHOT · STAFF ONLY</h3><img id="live" style="width:100%;border-radius:14px" src="/frame.jpg"></div><div class="stack"><div class="card"><h3>ACTIVE SESSION</h3><div id="who" class="muted">Waiting for customer</div></div><div class="card"><h3>PHYSICAL ITEM RECORD</h3><div id="phys"></div><div id="events" class="muted"></div></div><div class="card"><h3>PAID CART</h3><div id="cart"></div></div><div class="card"><h3>VERIFICATION</h3><div id="res" class="badge wait">Waiting</div></div><div class="card"><h3>RECENT SESSIONS</h3><div id="recent" class="muted"></div></div></div></div></main><script>__COMMON__
const PIN=new URLSearchParams(location.search).get('pin')||'';document.getElementById('gate').href='/gate?pin='+encodeURIComponent(PIN);document.getElementById('labels').href='/labels?pin='+encodeURIComponent(PIN);async function poll(){let r=await fetch('/api/active?pin='+encodeURIComponent(PIN));if(!r.ok)return;let a=await r.json();document.getElementById('recent').innerHTML=(a.recent||[]).map(x=>x.session_id+' · '+(x.paid?'paid':'shopping')+' · '+x.items+' items').join('<br>')||'No recent sessions';if(!a.session_id)return;let s=await (await fetch('/api/state/'+a.session_id)).json(),v=await (await fetch('/api/verify/'+a.session_id)).json();document.getElementById('who').textContent=a.session_id+' · '+(s.name||'Customer');document.getElementById('phys').innerHTML=chips(s.physical);document.getElementById('events').innerHTML=(s.events||[]).slice().reverse().map(e=>e.time+' · '+e.product+' · '+e.event).join('<br>');document.getElementById('cart').innerHTML=chips(s.cart)+'<p class="muted">'+(s.paid?'Checkout complete':'Not paid yet')+'</p>';badge(document.getElementById('res'),v)}setInterval(poll,1500);setInterval(()=>document.getElementById('live').src='/frame.jpg?pin='+encodeURIComponent(PIN)+'&t='+Date.now(),1000);poll();</script></body></html>'''
PROFILE=HEAD+r'''<header><b>◈ Sync Cart</b><span>Shopping history</span><small><a href="/">Home</a></small></header><main style="max-width:850px"><div class="eyebrow">YOUR LOCAL HISTORY</div><h2>Past demo checkouts.</h2><p class="muted">This list is saved only in this browser on this device. It is not an account or server-backed order history.</p><div class="card"><div id="history"></div><button class="ghost" onclick="clearHistory()">Clear local history</button><a class="btn secondary" href="/">Back to shopping</a></div></main><script>__COMMON__
function loadHistory(){let h=[];try{h=JSON.parse(localStorage.getItem('synccart_history')||'[]')}catch(e){}const box=document.getElementById('history');if(!h.length){box.innerHTML='<p class="muted">No completed demo checkouts saved in this browser yet.</p>';return}box.innerHTML=h.map(x=>'<div class="cart-row"><div class="emoji">✓</div><div><strong>Session '+x.session_id+'</strong><small>'+x.date+' · '+Object.keys(x.items||{}).length+' product types</small></div><b style="margin-left:auto">₹'+x.total+'</b></div>').join('')}function clearHistory(){localStorage.removeItem('synccart_history');loadHistory()}loadHistory();</script></body></html>'''
LABELS=HEAD+r'''<header><b>◈ Sync Cart</b><span>Product QR labels</span></header><main><div class="card"><h2>Generate item labels</h2><p class="muted">Print a unique label for each physical item. Product code mapping must exist in your environment configuration.</p><label>Labels per product</label><input class="input" id="n" type="number" value="3" min="1" max="20"><button onclick="load()">Generate labels</button></div><div class="product-grid" id="grid"></div></main><script>__COMMON__
async function load(){let n=Math.max(1,Math.min(20,parseInt(document.getElementById('n').value)||1)),ps=await(await fetch('/api/products')).json(),g=document.getElementById('grid');g.innerHTML='';ps.forEach(p=>{for(let i=1;i<=n;i++){let code='SC:'+p+'#'+String(i).padStart(3,'0'),d=document.createElement('div');d.className='card center';d.innerHTML='<img class="qr" style="margin:auto" src="/api/qr?data='+encodeURIComponent(code)+'"><h3>'+p+'</h3><p class="muted">Unit '+i+'</p><button class="ghost" onclick="window.print()">Print</button>';g.appendChild(d)}})}load()</script></body></html>'''

def page(html): return HTMLResponse(html.replace('__COMMON__',COMMON).replace('__CSS__',CSS))
def staff_page(html,pin):
    if not staff_ok(pin): return HTMLResponse('Staff PIN required. Add ?pin=YOURPIN to the link.',status_code=401)
    return page(html)
@app.get('/')
def home(): return page(LOGIN)
@app.get('/shop')
def shop(): return page(SHOP)
@app.get('/checkout')
def checkout(): return page(CHECKOUT)
@app.get('/exit')
def exit_page(): return page(EXIT)
@app.get('/staff')
def staff(pin:str=''): return staff_page(STAFF,pin)
@app.get('/gate')
def gate(pin:str=''): return staff_page(GATE,pin)
@app.get('/profile')
def profile(): return page(PROFILE)
@app.get('/labels')
def labels(pin:str=''): return staff_page(LABELS,pin)
if __name__=='__main__':
    import uvicorn
    uvicorn.run(app,host='0.0.0.0',port=PORT)
