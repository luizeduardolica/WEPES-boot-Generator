"""Leitura e gravação de paletas (PNG/BMP indexado, .act, .pal JASC/RIFF, .gpl)."""
from __future__ import annotations

import os
import struct
from typing import Sequence

import numpy as np
from PIL import Image

from .quantize import Palette, palette_from_image, TRANSPARENT_INDEX

PALETTE_FILTER = ("Paletas (*.png *.bmp *.gif *.tga *.act *.pal *.gpl);;"
                  "Imagens indexadas (*.png *.bmp *.gif *.tga);;Todos (*.*)")


def _from_colors(colors: np.ndarray, source: str, transparent_index: int = TRANSPARENT_INDEX) -> Palette:
    pal = Palette()
    n = min(256, len(colors))
    pal.colors[:n] = colors[:n]
    pal.locked[:n] = True
    pal.transparent_index = transparent_index
    pal.locked[transparent_index] = True
    pal.source = source
    return pal


def load_palette(paths: Sequence[str], lock_mode: str = "all") -> Palette:
    """Carrega uma paleta. Com várias imagens indexadas (mesma paleta), os índices
    usados por todas são somados — útil para travar só as cores realmente em uso no jogo."""
    paths = list(paths)
    if not paths:
        raise ValueError("Nenhum arquivo informado.")
    ext = os.path.splitext(paths[0])[1].lower()
    name = ", ".join(os.path.basename(p) for p in paths)
    if ext == ".act":
        return _from_colors(_read_act(paths[0]), name)
    if ext == ".gpl":
        return _from_colors(_read_gpl(paths[0]), name)
    if ext == ".pal":
        return _from_colors(_read_pal(paths[0]), name)

    result = None
    for p in paths:
        with Image.open(p) as im:
            im.load()
            pal = palette_from_image(im, lock_mode)
        if result is None:
            result = pal
        else:
            if not np.array_equal(result.colors[result.locked & pal.locked],
                                  pal.colors[result.locked & pal.locked]):
                raise ValueError(f"'{os.path.basename(p)}' tem uma paleta diferente das outras.")
            result.colors[pal.locked] = pal.colors[pal.locked]
            result.locked |= pal.locked
    result.source = name
    return result


def _read_act(path: str) -> np.ndarray:
    data = open(path, "rb").read()
    n = 256
    if len(data) >= 772:
        n = struct.unpack(">H", data[768:770])[0] or 256
    return np.frombuffer(data[:768], np.uint8).reshape(256, 3)[:n].copy()


def _read_gpl(path: str) -> np.ndarray:
    cols = []
    for line in open(path, encoding="utf-8", errors="ignore"):
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("GIMP") or ":" in s:
            continue
        parts = s.split()
        if len(parts) >= 3 and all(x.isdigit() for x in parts[:3]):
            cols.append([int(x) for x in parts[:3]])
    return np.array(cols, np.uint8)


def _read_pal(path: str) -> np.ndarray:
    data = open(path, "rb").read()
    if data[:4] == b"RIFF":
        i = data.find(b"data")
        n = struct.unpack("<H", data[i + 10:i + 12])[0]
        raw = np.frombuffer(data[i + 12:i + 12 + n * 4], np.uint8).reshape(n, 4)
        return raw[:, :3].copy()
    lines = data.decode("latin-1").split()
    if lines[0] != "JASC-PAL":
        raise ValueError("Formato .pal desconhecido.")
    n = int(lines[2])
    vals = list(map(int, lines[3:3 + n * 3]))
    return np.array(vals, np.uint8).reshape(n, 3)


def save_palette(pal: Palette, path: str) -> None:
    ext = os.path.splitext(path)[1].lower()
    cols = pal.colors.astype(np.uint8)
    if ext == ".act":
        with open(path, "wb") as f:
            f.write(cols.tobytes() + struct.pack(">HH", 256, pal.transparent_index))
    elif ext == ".pal":
        with open(path, "w", newline="\r\n") as f:
            f.write("JASC-PAL\n0100\n256\n")
            for r, g, b in cols:
                f.write(f"{r} {g} {b}\n")
    elif ext == ".gpl":
        with open(path, "w", encoding="utf-8") as f:
            f.write("GIMP Palette\nName: WEPES Boots\nColumns: 16\n#\n")
            for i, (r, g, b) in enumerate(cols):
                f.write(f"{r:3d} {g:3d} {b:3d}\tIndex {i}\n")
    else:  # PNG 16x16 indexado: pode ser recarregado como paleta
        idx = np.arange(256, dtype=np.uint8).reshape(16, 16)
        im = Image.new("P", (16, 16))
        im.frombytes(idx.tobytes())
        im.putpalette(cols.reshape(-1).tolist())
        im.save(path, transparency=pal.transparent_index)
