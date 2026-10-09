# GitHub + Render deployment steps

1. Download and extract `SyncCart_Final_Code.zip`.
2. Open your GitHub repository `killipardhasaradhi/SyncCart_see_sync_verify`.
3. Download/keep a backup of your current `app.py` and `core.py` before editing.
4. Upload the package's `app.py`, `requirements.txt`, `render.yaml`, and `README.md` to the repository root. Do not delete or overwrite your existing `core.py` or camera-agent files.
5. Commit the changes to the branch connected to Render.
6. In Render, open the existing web service and check **Settings → Build & Deploy**. The build command should be `pip install -r requirements.txt`; the start command should be `uvicorn app:app --host 0.0.0.0 --port $PORT`.
7. In Render **Environment**, set:
   - `SYNCCART_MODE` = `server`
   - `SYNCCART_MOCK` = `1` for a safe UI demo without camera inference
   - `SYNCCART_CODES` = `{"CHOC001":"Chocolate","NOTE001":"Notebook","BOTTLE001":"Bottle"}`
   - `SYNCCART_PIN` = a new strong staff PIN
   - `SYNCCART_KEY` = a new strong agent API key if you use the camera agent
8. Save and redeploy. Never commit the PIN or API key to GitHub.
9. Test `/`, `/shop`, `/checkout`, `/exit`, `/profile`, `/staff?pin=YOUR_PIN`, `/gate?pin=YOUR_PIN`, and `/labels?pin=YOUR_PIN`.
10. Test QR scanning on HTTPS and allow camera permission. If using a physical camera agent, configure its base URL and matching API key separately.

The staff PIN is shown as a placeholder in these instructions; use your own environment value in the URL. Do not share that URL publicly.
