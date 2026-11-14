"""Draws the Qwen Bot robot icon (512x512) and start screen (800x800) as PNGs, pure Python."""
import math, struct, sys, zlib

# ---- signed distance shapes (y points down) ----
def rrect(cx, cy, hw, hh, r):
    def d(x, y):
        qx, qy = abs(x-cx)-hw+r, abs(y-cy)-hh+r
        return math.hypot(max(qx, 0), max(qy, 0)) + min(max(qx, qy), 0) - r
    return d, (cx-hw, cy-hh, cx+hw, cy+hh)
def circle(cx, cy, r):
    return (lambda x, y: math.hypot(x-cx, y-cy) - r), (cx-r, cy-r, cx+r, cy+r)
def capsule(ax, ay, bx, by, r):
    def d(x, y):
        px, py, ex, ey = x-ax, y-ay, bx-ax, by-ay
        t = max(0, min(1, (px*ex+py*ey)/(ex*ex+ey*ey or 1)))
        return math.hypot(px-ex*t, py-ey*t) - r
    return d, (min(ax, bx)-r, min(ay, by)-r, max(ax, bx)+r, max(ay, by)+r)
def arc(cx, cy, R, a0, a1, r):  # degrees, clockwise on screen from a0 to a1 (0 = right, 90 = down)
    a0, a1 = math.radians(a0), math.radians(a1)
    ends = [(cx+R*math.cos(a), cy+R*math.sin(a)) for a in (a0, a1)]
    def d(x, y):
        a = math.atan2(y-cy, x-cx)
        if (a - a0) % (2*math.pi) <= (a1 - a0):
            return abs(math.hypot(x-cx, y-cy) - R) - r
        return min(math.hypot(x-ex, y-ey) for ex, ey in ends) - r
    return d, (cx-R-r, cy-R-r, cx+R+r, cy+R+r)

WHITE, SCREEN, GLOW, AMBER = (255, 255, 255), (6, 52, 47), (37, 211, 102), (255, 202, 40)

def robot(cx, cy, s):
    """Robot head; unit 1 = s pixels. Returns [(shape, color)] back to front."""
    P = lambda x, y: (cx + x*s, cy + y*s)
    L = []
    L.append((capsule(*P(0, -0.98), *P(0, -0.62), 0.05*s), WHITE))           # antenna
    L.append((circle(*P(0, -1.04), 0.12*s), AMBER))
    for side in (-1, 1):
        L.append((rrect(*P(side*0.70, -0.15), 0.08*s, 0.18*s, 0.07*s), WHITE))  # ears
    L.append((rrect(*P(0, -0.15), 0.64*s, 0.50*s, 0.24*s), WHITE))           # head
    L.append((rrect(*P(0, -0.15), 0.49*s, 0.34*s, 0.17*s), SCREEN))          # face screen
    for side in (-1, 1):
        L.append((circle(*P(side*0.21, -0.24), 0.095*s), GLOW))              # eyes
    L.append((arc(*P(0, -0.12), 0.17*s, 25, 155, 0.045*s), GLOW))            # smile
    return L

# Rounded stroke font, cap height 1, baseline 0, stroke radius W
W = 0.085
GLYPHS = {
    "Q": (1.0, lambda: [("c", .5, -.5, .5), ("l", .6, -.22, .98, .06)]),
    "w": (0.8, lambda: [("l", 0, -.72, .2, 0), ("l", .2, 0, .4, -.5), ("l", .4, -.5, .6, 0), ("l", .6, 0, .8, -.72)]),
    "e": (0.72, lambda: [("a", .36, -.36, .36, 45, 360), ("l", 0, -.36, .72, -.36)]),
    "n": (0.64, lambda: [("l", 0, -.72, 0, 0), ("a", .32, -.4, .32, 180, 360), ("l", .64, -.4, .64, 0)]),
    " ": (0.25, lambda: []),
    "B": (0.62, lambda: [("l", 0, -1, 0, 0), ("l", 0, -1, .32, -1), ("l", 0, -.53, .34, -.53), ("l", 0, 0, .36, 0),
                         ("a", .32, -.765, .235, -90, 90), ("a", .36, -.265, .265, -90, 90)]),
    "o": (0.72, lambda: [("c", .36, -.36, .36)]),
    "t": (0.5, lambda: [("l", .16, -.95, .16, -.2), ("a", .36, -.2, .2, 90, 180), ("l", .36, 0, .48, 0), ("l", 0, -.72, .44, -.72)]),
}
def text(s, cx, baseline, cap, colors):
    """Centered text; colors[i] for each word."""
    gap = 0.2
    width = sum(GLYPHS[ch][0] for ch in s) + gap*(len(s)-1) + 2*W
    x = cx - width*cap/2 + W*cap
    L, word = [], 0
    for ch in s:
        adv, strokes = GLYPHS[ch]
        if ch == " ":
            word += 1
        for st in strokes():
            k = st[0]; v = st[1:]
            X = lambda u: x + u*cap; Y = lambda u: baseline + u*cap
            if k == "l": shp = capsule(X(v[0]), Y(v[1]), X(v[2]), Y(v[3]), W*cap)
            elif k == "c": shp = arc(X(v[0]), Y(v[1]), v[2]*cap, 0, 359.999, W*cap)
            else: shp = arc(X(v[0]), Y(v[1]), v[2]*cap, v[3], v[4], W*cap)
            L.append((shp, colors[word]))
        x += (adv + gap)*cap
    return L

def render(path, size, bg, layers):
    rows = []
    for y in range(size):
        row = bytearray([0]); py = y + .5
        for x in range(size):
            px = x + .5
            c = bg(px, py)
            for (d, (x0, y0, x1, y1)), col in layers:
                if x0-1 <= px <= x1+1 and y0-1 <= py <= y1+1:
                    t = min(max(0.5 - d(px, py), 0), 1)
                    if t:
                        c = tuple(c[i] + (col[i]-c[i])*t for i in range(3))
            row += bytes(int(round(v)) for v in c)
        rows.append(bytes(row))
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t+d) & 0xffffffff)
    open(path, "wb").write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
                           + chunk(b"IDAT", zlib.compress(b"".join(rows), 9)) + chunk(b"IEND", b""))

out = sys.argv[1]
TOP, BOT = (18, 140, 126), (7, 94, 84)
grad = lambda S: (lambda x, y: tuple(TOP[i] + (BOT[i]-TOP[i])*y/S for i in range(3)))
render(f"{out}/icon.png", 512, grad(512), robot(256, 328, 180))
render(f"{out}/presplash.png", 800, lambda x, y: (7, 94, 84),
       robot(400, 360, 200) + text("Qwen Bot", 400, 650, 84, [WHITE, GLOW]))
