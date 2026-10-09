"""SyncCart core logic: pick/return tracking, sessions, and cart verification.

No camera and no web code in here, so it can be tested on its own.
"""
import secrets
import time
from collections import Counter

ABSENT_SECS = 1.5   # product must be missing from the shelf zone this long (with a person near) to count as PICKED
BACK_SECS = 1.0     # product must be back in the shelf zone this long to count as RETURNED


class PickTracker:
    """Turns 'what is visible in the shelf zone' into PICKED / RETURNED events."""

    def __init__(self):
        self.items = {}  # product -> {"status": "ON_SHELF" | "HELD", "t": timer start or None}

    def reset(self):
        self.items = {}

    def update(self, seen, person_near, now):
        events = []
        for name in seen:
            self.items.setdefault(name, {"status": "ON_SHELF", "t": None})
        for name, s in self.items.items():
            if s["status"] == "ON_SHELF":
                if name in seen:
                    s["t"] = None
                else:
                    if s["t"] is None:
                        s["t"] = now
                    if now - s["t"] >= ABSENT_SECS and person_near:
                        s["status"], s["t"] = "HELD", None
                        events.append((name, "PICKED"))
            else:  # HELD
                if name in seen:
                    if s["t"] is None:
                        s["t"] = now
                    if now - s["t"] >= BACK_SECS:
                        s["status"], s["t"] = "ON_SHELF", None
                        events.append((name, "RETURNED"))
                else:
                    s["t"] = None
        return events

    def statuses(self):
        return {n: s["status"] for n, s in self.items.items()}


ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # no 0/O/1/I so codes are easy to read out


class Store:
    """Keeps sessions: the digital cart (what was scanned) and physical events (what the camera saw).

    Each customer gets a temporary code (6 characters) that expires after ttl_hours.
    """

    def __init__(self, ttl_hours=3.0):
        self.sessions = {}
        self.active = None  # session the camera is currently recording for
        self.ttl = ttl_hours * 3600
        self.units = {}     # (product, unit) -> session that scanned this exact physical item

    def new_session(self, name=""):
        while True:
            sid = "".join(secrets.choice(ALPHABET) for _ in range(6))
            if sid not in self.sessions:
                break
        self.sessions[sid] = {
            "name": (name or "").strip()[:20] or "Guest",
            "cart": Counter(), "paid": False, "events": [], "units": [], "created": time.time(),
        }
        self.active = sid
        return sid

    def exists(self, sid):
        s = self.sessions.get(sid)
        if not s:
            return False
        if time.time() - s["created"] > self.ttl:
            if self.active == sid:
                self.active = None
            return False
        return True

    def seconds_left(self, sid):
        return max(0, int(self.ttl - (time.time() - self.sessions[sid]["created"])))

    def is_paid(self, sid):
        return self.sessions[sid]["paid"]

    def add_event(self, sid, product, event):
        self.sessions[sid]["events"].append({"product": product, "event": event, "t": time.time()})

    def scan(self, sid, product, qty=1):
        self.sessions[sid]["cart"][product] += int(qty)

    def scan_unit(self, sid, product, unit):
        """Scan one uniquely labelled item. Returns "ok", "dup" (already in this cart) or "taken" (another customer has it)."""
        key = (product, unit)
        owner = self.units.get(key)
        if owner == sid:
            return "dup"
        if owner is not None and self.exists(owner):
            return "taken"
        self.units[key] = sid
        self.sessions[sid]["units"].append(key)
        self.sessions[sid]["cart"][product] += 1
        return "ok"

    def unscan(self, sid, product):
        """Remove one item of this product from the cart and free its unit label."""
        s = self.sessions[sid]
        if s["cart"][product] > 0:
            s["cart"][product] -= 1
        for key in reversed(s["units"]):
            if key[0] == product:
                s["units"].remove(key)
                self.units.pop(key, None)
                break

    def pay(self, sid):
        self.sessions[sid]["paid"] = True

    def physical(self, sid):
        c = Counter()
        for e in self.sessions[sid]["events"]:
            c[e["product"]] += 1 if e["event"] == "PICKED" else -1
        return +c  # drops zero and negative counts

    def verify(self, sid):
        s = self.sessions[sid]
        phys, dig = self.physical(sid), +s["cart"]
        base = {"session_id": sid, "physical": dict(phys), "digital": dict(dig), "paid": s["paid"]}
        if not s["paid"]:
            return {**base, "result": "PAYMENT PENDING", "message": "Payment has not been completed yet."}
        if phys == dig:
            return {**base, "result": "VERIFIED", "message": "Physical cart matches paid cart."}
        return {
            **base,
            "result": "POTENTIAL DISCREPANCY",
            "message": "Additional verification required.",
            "not_in_paid_cart": dict(phys - dig),
            "paid_but_not_seen": dict(dig - phys),
        }

    def state(self, sid):
        s = self.sessions[sid]
        return {
            "session_id": sid,
            "name": s["name"],
            "paid": s["paid"],
            "seconds_left": self.seconds_left(sid),
            "cart": dict(+s["cart"]),
            "physical": dict(self.physical(sid)),
            "events": [
                {"product": e["product"], "event": e["event"], "time": time.strftime("%H:%M:%S", time.localtime(e["t"]))}
                for e in s["events"][-10:]
            ],
        }

    def recent(self, n=8):
        live = [(sid, s) for sid, s in self.sessions.items() if self.exists(sid)]
        live.sort(key=lambda x: -x[1]["created"])
        return [{"session_id": sid, "name": s["name"], "paid": s["paid"], "items": sum((+s["cart"]).values())} for sid, s in live[:n]]
