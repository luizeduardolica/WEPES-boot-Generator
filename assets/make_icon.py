"""Gera assets/icon.ico (grade de paleta 4x4 estilizada)."""
from PIL import Image, ImageDraw

S = 256
im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
d = ImageDraw.Draw(im)
d.rounded_rectangle((8, 8, S - 8, S - 8), 40, fill=(30, 30, 34, 255), outline=(0, 120, 215, 255), width=10)
cols = [(230, 60, 40), (250, 160, 30), (250, 230, 60), (60, 200, 90),
        (40, 200, 190), (40, 120, 230), (130, 70, 220), (230, 70, 160),
        (245, 245, 245), (180, 180, 180), (110, 110, 110), (20, 20, 20),
        (120, 220, 200), (200, 90, 60), (90, 60, 40), (71, 112, 76)]
c = 46
for i, col in enumerate(cols):
    x, y = 40 + (i % 4) * c, 40 + (i // 4) * c
    d.rounded_rectangle((x, y, x + c - 6, y + c - 6), 6, fill=col + (255,))
im.save(__import__("os").path.join(__import__("os").path.dirname(__file__), "icon.ico"), sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
