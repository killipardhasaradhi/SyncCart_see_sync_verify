# SyncCart: See. Sync. Verify.

A mobile web app for customers + a camera that verifies what they physically take against what they pay for.

## Links
Customers (send them this link):
- /          login with a nickname, get a temporary customer code
- /shop      scan product QR codes / barcodes with the phone camera, build the cart, pay, get an exit QR
- /exit      shows the result for one customer code
Staff:
- /staff     live camera snapshot, physical items vs paid cart, recent customers
- /gate      scan a customer's exit QR with a camera -> VERIFIED or POTENTIAL DISCREPANCY
- /labels    printable product QR codes (print them and stick one on each product)

Customer codes are random 6 characters and expire after 3 hours (SYNCCART_TTL_HOURS). No face recognition, no phone number.

## IMPORTANT: phone camera needs HTTPS
Browsers only allow camera access on https sites (or localhost). So scanning QR codes on a phone works on your
public website (Render gives https) or through a tunnel (ngrok / cloudflared). It will NOT work on a plain
http://192.168.x.x address. The tap-to-add buttons still work there.

---
# Option A: everything on your laptop
    pip install -r requirements.txt        (put best.pt in this folder)
    python app.py
Open http://localhost:8000/staff. For phones, run a tunnel (ngrok http 8000), then start the app with
SYNCCART_PUBLIC_URL=https://your-tunnel-link so QR codes contain the right address.

# Option B: public website + camera agent (recommended)
    Customer phone  --->  PUBLIC WEBSITE (Render or similar)  <---  agent.py on your laptop (webcam + YOLO)

Step 1: put the website online (for example Render)
1. Create a GitHub repository with app.py, core.py and requirements-server.txt.
2. On render.com: New > Web Service > connect the repository.
3. Build command:  pip install -r requirements-server.txt
4. Start command:  python app.py
5. Environment variables:
   - SYNCCART_MODE = server
   - SYNCCART_KEY  = a secret word shared with the agent (for example tiger-lamp-42)
   - SYNCCART_PIN  = a staff PIN (for example 4821). Protects /staff, /gate, /labels and the camera snapshot.
6. Deploy. You get a link like https://synccart.onrender.com. That is the link you send to customers.
Free hosting can sleep when idle and may take around a minute to wake. Open the link before the demo.
Data is kept in memory, so it resets when the site restarts. Free-tier limits change, so check the host's plan.

Step 2: run the camera agent on the laptop
    pip install -r requirements.txt        (put best.pt in this folder)
Windows PowerShell:
    $env:SYNCCART_SERVER="https://synccart.onrender.com"; $env:SYNCCART_KEY="tiger-lamp-42"; python agent.py
Mac / Linux:
    SYNCCART_SERVER=https://synccart.onrender.com SYNCCART_KEY=tiger-lamp-42 python agent.py

Step 3: product QR codes
1. Open https://synccart.onrender.com/labels?pin=4821 (after the agent has connected, so the product names load).
2. Print the page and stick one QR on each product. The QR contains SC:<product name>.
3. Real barcodes also work: set SYNCCART_CODES='{"8901234567890":"Notebook"}' on the website.

Step 4: demo
1. Customer opens the website link, types a nickname, presses Start shopping. They get a temporary code.
2. They scan the product QR codes with the phone camera and press Pay now. An exit QR appears.
3. The camera agent records what is physically picked, in the background.
4. Staff open /gate?pin=4821 and scan the customer's exit QR: VERIFIED or POTENTIAL DISCREPANCY.

Limits of this version: one customer is tracked by the camera at a time (the most recent login), as in the proof of concept.
The staff PIN is simple protection for a demo, not production security.

# Test without a camera
    Windows PowerShell:   $env:SYNCCART_MOCK="1"; python app.py
    Mac / Linux:          SYNCCART_MOCK=1 python app.py
Mock mode needs no camera or model. Open http://localhost:8000/docs and call POST /api/sim/{code} with
{"product":"Notebook","event":"PICKED"} to fake camera events.

# Settings (environment variables)
SYNCCART_MODE (local/server), SYNCCART_KEY, SYNCCART_PIN, SYNCCART_MODEL, SYNCCART_CAMERA, SYNCCART_PORT,
SYNCCART_PUBLIC_URL, SYNCCART_TTL_HOURS, SYNCCART_CODES, SYNCCART_MOCK.
Shelf zone: ZONE in app.py and agent.py. Timings: ABSENT_SECS / BACK_SECS in core.py.
