# Sync Cart — redesigned website

This is a responsive, dark navy + emerald redesign for the existing Sync Cart prototype. It includes a landing page, product catalogue/search, cart, demo checkout, exit QR, customer verification page, and separate staff-only camera/verification pages.

## Files
- `app.py` — redesigned FastAPI app and UI.
- `requirements.txt` — Python packages.
- `render.yaml` — optional Render build/start configuration.

## Important
1. Keep your existing `core.py` in the same repository folder. This app imports `PickTracker` and `Store` from it.
2. The checkout is a **demo only**. It does not connect to UPI, cards, or a payment provider and does not charge money.
3. Camera-based physical-item tracking only works when a compatible local YOLO model/camera agent is connected. The UI does not show live tracking to customers.
4. Set `SYNCCART_MODE=server` for a public Render deployment where your local camera agent sends data. Use `SYNCCART_MOCK=1` only for testing without a camera.
5. Set `SYNCCART_CODES` in Render as JSON if you want product QR IDs mapped to product names, for example `{"CHOC001":"Chocolate","NOTE001":"Notebook","BOTTLE001":"Bottle"}`. Keep product names consistent with the product names your backend exposes.
6. Set a strong `SYNCCART_PIN` for staff pages and rotate any previously exposed credentials. Do not commit API keys or PINs to GitHub.

## GitHub / Render update
Back up the current `app.py`, replace it with this `app.py`, keep `core.py` and your existing camera-agent files, then commit and push. Render will redeploy from the repository. Test `/`, `/shop`, `/checkout`, `/exit`, and the staff pages after deployment.
