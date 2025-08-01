# ---------------- Light piston ----------------
LIGHT_AIRCRAFT_CLIMB_PROFILE = [(5000, 85), (float("inf"), 95)]
LIGHT_AIRCRAFT_DESCENT_PROFILE = [(5000, 90), (10000, 100), (float("inf"), 100)]
LIGHT_AIRCRAFT_CLIMB_VS_PROFILE = [(5000, 700), (float("inf"), 500)]

# ---------------- Turboprop ----------------
TURBOPROP_CLIMB_PROFILE = [(10000, 190), (18000, 220), (float("inf"), 255)]
TURBOPROP_DESCENT_PROFILE = [(10000, 210), (25000, 250), (float("inf"), 260)]
TURBOPROP_CLIMB_VS_PROFILE = [(10000, 1500), (18000, 1200), (float("inf"), 800)]

# ---------------- Narrow‑body jet ------------
NARROW_BODY_JET_CLIMB_PROFILE = [
    (10000, 250),
    (20000, 330),
    (28000, 390),
    (float("inf"), 460),
]
NARROW_BODY_JET_DESCENT_PROFILE = [
    (10000, 250),
    (20000, 300),
    (28000, 360),
    (float("inf"), 450),
]
NARROW_BODY_JET_CLIMB_VS_PROFILE = [(10000, 3000), (28000, 2000), (float("inf"), 1200)]

# ---------------- Wide‑body jet --------------
WIDE_BODY_JET_CLIMB_PROFILE = [
    (10000, 250),
    (20000, 340),
    (30000, 400),
    (float("inf"), 490),
]
WIDE_BODY_JET_DESCENT_PROFILE = [
    (10000, 250),
    (30000, 320),
    (41000, 460),
    (float("inf"), 490),
]
WIDE_BODY_JET_CLIMB_VS_PROFILE = [(10000, 2500), (30000, 1600), (float("inf"), 1000)]

# ---------------- Business jet --------------
BUSINESS_JET_CLIMB_PROFILE = [(10000, 250), (25000, 340), (float("inf"), 480)]
BUSINESS_JET_DESCENT_PROFILE = [
    (10000, 250),
    (25000, 340),
    (45000, 480),
    (float("inf"), 480),
]
BUSINESS_JET_CLIMB_VS_PROFILE = [(10000, 4000), (25000, 3000), (float("inf"), 1800)]

# ================= DESCENT VERTICAL‑SPEED PROFILES ================

# 1 Light piston GA (C‑172/PA‑28)
LIGHT_AIRCRAFT_DESCENT_VS_PROFILE = [
    (5_000, 800),  # Below 5 000 ft
    (10_000, 600),  # 5 000 – <10 000 ft
    (float("inf"), 500),
]

# 2 Regional turboprop (ATR 72 / Dash‑8)
TURBOPROP_DESCENT_VS_PROFILE = [(10_000, 1_200), (25_000, 1_500), (float("inf"), 1_800)]

# 3 Narrow‑body jet (B737‑800 / A320)
NARROW_BODY_JET_DESCENT_VS_PROFILE = [
    (10_000, 1_500),
    (24_000, 3_000),
    (float("inf"), 1_000),
]

# 4 Wide‑body jet (B777 / A350)
WIDE_BODY_JET_DESCENT_VS_PROFILE = [
    (10_000, 1_500),
    (24_000, 3_000),
    (float("inf"), 1_000),
]

# 5 High‑performance business jet (G650, Global 7500)
BUSINESS_JET_DESCENT_VS_PROFILE = [
    (10_000, 2_000),
    (25_000, 2_500),
    (float("inf"), 3_000),
]

# 6 Generic conservative profile (unchanged)
CONSERVATIVE_DESCENT_VS_PROFILE = [
    (10_000, 1_200),
    (20_000, 1_500),
    (35_000, 2_100),
    (float("inf"), 2_100),
]
