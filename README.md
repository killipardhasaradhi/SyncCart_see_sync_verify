# Sync Cart — Holographic Future redesign

A responsive dark navy/cyan/emerald-inspired FastAPI UI for the Sync Cart prototype. It includes:

- Futuristic landing page and responsive customer shopping UI
- Product search, product QR scanning, add/remove cart quantities
- Demo checkout and signed exit QR flow
- Customer-facing verification result
- Budget meter (example budget ₹1,000)
- Illustrative aisle route (update aisle mapping for your real store)
- Browser-local checkout history page at `/profile`
- Staff-only dashboard, exit gate scanner, and printable QR labels
- Existing customer and agent API routes retained

## Important before deploying

1. **Keep your existing `core.py`** in the same repository directory. The app imports `PickTracker` and `Store` from it. Keep your existing camera-agent files too.
2. This ZIP intentionally does not replace `core.py`; back up your repository before replacing `app.py`.
3. Checkout is **demo only**. It does not charge money or connect to UPI, cards, or a payment provider.
4. The aisle route uses example aisle labels; it is not live indoor navigation. Edit the `aisle` map in the SHOP page JavaScript to match your store.
5. `/profile` stores demo history in the browser's `localStorage` only. It is not server-backed account history and may not appear on another device/browser.
6. On Render, `SYNCCART_MODE=server` means camera inference is not running on Render; the local `agent.py` must send frames/events if you want real physical-item records. `SYNCCART_MOCK=1` is for website testing only.
7. Set a strong `SYNCCART_PIN` and `SYNCCART_KEY` in Render Environment settings. Do not put secrets in GitHub. If prior values were exposed, rotate them.
8. Product QR codes are mapped by `SYNCCART_CODES`, e.g. `{"CHOC001":"Chocolate","NOTE001":"Notebook","BOTTLE001":"Bottle"}`. Product names must match backend product names.

## Local run

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

You must have the existing `core.py` beside `app.py`.

## Routes retained

Customer: `/`, `/shop`, `/checkout`, `/exit`, `/profile`
Staff: `/staff?pin=...`, `/gate?pin=...`, `/labels?pin=...`
API: `/api/products`, `/api/login`, `/api/scan/{sid}`, `/api/scan_code/{sid}`, `/api/pay/{sid}`, `/api/state/{sid}`, `/api/verify/{sid}`, `/api/exit_scan`, `/api/active`, `/api/qr`, `/frame.jpg`, `/api/agent/products`, `/api/agent/frame`, `/api/agent/event`, `/api/sim/{sid}`
