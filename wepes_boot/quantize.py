"""Núcleo de conversão: chuteiras PES 6 PC -> PES PS2 (PNG indexado 8 bits, 256x256).

Abordagem de quantização
------------------------
* Todas as chuteiras compartilham UMA paleta de 256 entradas (índice 0 = transparente,
  igual às chuteiras de PS2 originais).
* As cores são agrupadas no espaço perceptual **Oklab** (distância próxima do que o
  olho percebe), usando um histograma ponderado de todas as imagens. Cada chuteira
  pode ter o mesmo peso ("balanceado"), assim uma chuteira com poucas cores não é
  engolida pelas outras.
* Inicialização por k-means++ ponderado e refinamento por k-means (Lloyd) em Oklab.
* Paletas existentes podem ser travadas: cores travadas nunca mudam de posição/valor,
  e apenas os slots livres recebem cores novas. Assim as chuteiras que já estão no
  jogo continuam corretas.
* Remapeamento com dithering Floyd-Steinberg serpentina opcional (força ajustável),
  que nunca espalha erro para pixels transparentes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional, Sequence

import numpy as np
from PIL import Image

PS2_SIZE = 256
TRANSPARENT_INDEX = 0
# Cor RGB usada pelos arquivos de PS2 originais na entrada transparente.
DEFAULT_TRANSPARENT_RGB = (71, 112, 76)

ProgressFn = Optional[Callable[[str, float], None]]


# --------------------------------------------------------------------------- cores
def srgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    """rgb: (..., 3) em 0..255 -> Oklab (..., 3) float32."""
    c = np.asarray(rgb, dtype=np.float32) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    m1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                   [0.2119034982, 0.6806995451, 0.1073969566],
                   [0.0883024619, 0.2817188376, 0.6299787005]], dtype=np.float32)
    lms = lin @ m1.T
    lms = np.cbrt(np.maximum(lms, 0))
    m2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                   [1.9779984951, -2.4285922050, 0.4505937099],
                   [0.0259040371, 0.7827717662, -0.8086757660]], dtype=np.float32)
    return (lms @ m2.T).astype(np.float32)


def oklab_to_srgb(lab: np.ndarray) -> np.ndarray:
    lab = np.asarray(lab, dtype=np.float32)
    m2i = np.array([[1.0, 0.3963377774, 0.2158037573],
                    [1.0, -0.1055613458, -0.0638541728],
                    [1.0, -0.0894841775, -1.2914855480]], dtype=np.float32)
    lms = (lab @ m2i.T) ** 3
    m1i = np.array([[4.0767416621, -3.3077115913, 0.2309699292],
                    [-1.2684380046, 2.6097574011, -0.3413193965],
                    [-0.0041960863, -0.7034186147, 1.7076147010]], dtype=np.float32)
    lin = np.clip(lms @ m1i.T, 0, 1)
    c = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * np.power(lin, 1 / 2.4) - 0.055)
    return np.clip(np.round(c * 255), 0, 255).astype(np.uint8)


def nearest(points: np.ndarray, centers: np.ndarray, chunk: int = 32768):
    """Índice do centro mais próximo (distância euclidiana) e a distância²."""
    idx = np.empty(len(points), dtype=np.int32)
    dist = np.empty(len(points), dtype=np.float32)
    cc = (centers ** 2).sum(1)
    for s in range(0, len(points), chunk):
        p = points[s:s + chunk]
        d = cc[None, :] - 2.0 * (p @ centers.T) + (p ** 2).sum(1)[:, None]
        i = d.argmin(1)
        idx[s:s + chunk] = i
        dist[s:s + chunk] = np.maximum(d[np.arange(len(p)), i], 0)
    return idx, dist


# --------------------------------------------------------------------- paleta
@dataclass
class Palette:
    """Paleta de 256 cores. `locked[i]` = cor que não pode mudar."""
    colors: np.ndarray = field(default_factory=lambda: np.zeros((256, 3), np.uint8))
    locked: np.ndarray = field(default_factory=lambda: np.zeros(256, bool))
    transparent_index: int = TRANSPARENT_INDEX
    source: str = ""

    def copy(self) -> "Palette":
        return Palette(self.colors.copy(), self.locked.copy(), self.transparent_index, self.source)

    @property
    def free_slots(self) -> int:
        usable = ~self.locked
        usable[self.transparent_index] = False
        return int(usable.sum())

    def flat(self) -> list[int]:
        return self.colors.astype(np.uint8).reshape(-1).tolist()


# --------------------------------------------------------------------- imagens
@dataclass
class ConvertOptions:
    resize_mode: str = "pad"          # "pad" (padrão PS2: arte à esquerda) | "stretch"
    alpha_threshold: int = 128
    dither: float = 0.0               # 0..1 (força Floyd-Steinberg)
    balanced: bool = True             # cada chuteira com o mesmo peso
    kmeans_iters: int = 16
    transparent_rgb: tuple = DEFAULT_TRANSPARENT_RGB


def prepare_image(img: Image.Image, opts: ConvertOptions) -> tuple[np.ndarray, np.ndarray]:
    """Normaliza qualquer chuteira para a tela 256x256 do PS2.

    Retorna (rgb uint8 HxWx3, máscara opaca bool HxW).
    PES 6 PC = 170x256 -> a arte fica em x=0..169 e o resto é transparente (é
    exatamente o layout das chuteiras de PS2). Outros tamanhos são escalados para
    altura 256 mantendo a proporção (ou esticados, se escolhido).
    """
    img = img.convert("RGBA")
    w, h = img.size
    if (w, h) != (PS2_SIZE, PS2_SIZE):
        if opts.resize_mode == "stretch":
            img = img.resize((PS2_SIZE, PS2_SIZE), Image.LANCZOS)
        else:
            scale = min(PS2_SIZE / h, PS2_SIZE / w)
            nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
            if (nw, nh) != (w, h):
                img = img.resize((nw, nh), Image.LANCZOS)
            canvas = Image.new("RGBA", (PS2_SIZE, PS2_SIZE), (0, 0, 0, 0))
            canvas.paste(img, (0, 0))
            img = canvas
    a = np.asarray(img)
    rgb = a[..., :3].copy()
    mask = a[..., 3] >= opts.alpha_threshold
    return rgb, mask


def palette_from_image(img: Image.Image, lock_mode: str = "all") -> Palette:
    """Lê a paleta de um PNG/BMP indexado.

    lock_mode: "all"  -> trava todas as entradas existentes no arquivo
               "used" -> trava somente os índices usados pelos pixels
    Entradas além do tamanho da paleta do arquivo ficam livres.
    """
    if img.mode != "P":
        raise ValueError("A imagem não é indexada (modo P). Use uma chuteira de PS2 em 8 bits.")
    raw = img.getpalette() or []
    n = len(raw) // 3
    pal = Palette()
    pal.colors[:n] = np.array(raw[:n * 3], np.uint8).reshape(-1, 3)
    t = img.info.get("transparency")
    ti = TRANSPARENT_INDEX
    if isinstance(t, int):
        ti = t
    elif isinstance(t, (bytes, bytearray)):
        zeros = [i for i, v in enumerate(t) if v == 0]
        if zeros:
            ti = zeros[0]
    pal.transparent_index = ti
    if lock_mode == "used":
        used = np.unique(np.asarray(img))
        pal.locked[used] = True
    else:
        pal.locked[:n] = True
    pal.locked[ti] = True
    return pal


# ------------------------------------------------------------------ quantização
def _weighted_histogram(images: Sequence[tuple[np.ndarray, np.ndarray]], balanced: bool):
    keys_all, w_all = [], []
    for rgb, mask in images:
        px = rgb[mask].astype(np.uint32)
        if len(px) == 0:
            continue
        keys = (px[:, 0] << 16) | (px[:, 1] << 8) | px[:, 2]
        weight = (1.0 / len(px)) if balanced else 1.0
        keys_all.append(keys)
        w_all.append(np.full(len(keys), weight, np.float64))
    if not keys_all:
        return np.zeros((0, 3), np.uint8), np.zeros(0)
    keys = np.concatenate(keys_all)
    ws = np.concatenate(w_all)
    uk, inv = np.unique(keys, return_inverse=True)
    weights = np.bincount(inv, weights=ws)
    cols = np.stack([(uk >> 16) & 255, (uk >> 8) & 255, uk & 255], 1).astype(np.uint8)
    return cols, weights


def build_palette(images: Sequence[tuple[np.ndarray, np.ndarray]],
                  base: Optional[Palette] = None,
                  opts: ConvertOptions = ConvertOptions(),
                  progress: ProgressFn = None,
                  seed: int = 1234) -> Palette:
    """Gera (ou completa) a paleta compartilhada para todas as imagens."""
    pal = base.copy() if base is not None else Palette()
    ti = pal.transparent_index
    if base is None:
        pal.colors[ti] = opts.transparent_rgb
        pal.locked[:] = False
    pal.locked[ti] = True

    free = np.where(~pal.locked)[0]
    fixed_idx = np.where(pal.locked)[0]
    fixed_idx = fixed_idx[fixed_idx != ti]
    k_free = len(free)
    if progress:
        progress("Montando histograma de cores…", 0.05)
    cols, weights = _weighted_histogram(images, opts.balanced)
    if k_free == 0 or len(cols) == 0:
        return pal

    labs = srgb_to_oklab(cols)
    fixed_lab = srgb_to_oklab(pal.colors[fixed_idx]) if len(fixed_idx) else np.zeros((0, 3), np.float32)

    # Poucas cores: usa as cores exatas (sem perdas).
    if len(fixed_idx):
        _, dfix = nearest(labs, fixed_lab)
        new_mask = dfix > 1e-7
    else:
        new_mask = np.ones(len(cols), bool)
    if new_mask.sum() <= k_free:
        newc = cols[new_mask]
        pal.colors[free[:len(newc)]] = newc
        # slots que sobraram continuam livres (marcados com a cor transparente)
        if base is None:
            pal.colors[free[len(newc):]] = opts.transparent_rgb
        pal.locked[free[:len(newc)]] = True
        return _finalize(pal, base, free[:len(newc)])

    rng = np.random.default_rng(seed)
    w = weights / weights.sum()

    # k-means++ ponderado, considerando as cores travadas como centros já existentes
    if progress:
        progress("Escolhendo cores iniciais…", 0.15)
    if len(fixed_lab):
        _, d2 = nearest(labs, fixed_lab)
    else:
        d2 = np.full(len(labs), np.inf, np.float32)
    centers = np.empty((k_free, 3), np.float32)
    for j in range(k_free):
        if np.isinf(d2).all():
            p = w
        else:
            p = w * d2
            s = p.sum()
            p = w if s <= 0 else p / s
        i = rng.choice(len(labs), p=p)
        centers[j] = labs[i]
        dd = ((labs - labs[i]) ** 2).sum(1)
        d2 = np.minimum(d2, dd)

    # Lloyd ponderado; centros travados não se movem
    all_centers = np.concatenate([fixed_lab, centers]) if len(fixed_lab) else centers
    nf = len(fixed_lab)
    for it in range(opts.kmeans_iters):
        if progress:
            progress(f"Otimizando paleta (k-means {it + 1}/{opts.kmeans_iters})…",
                     0.2 + 0.6 * it / opts.kmeans_iters)
        idx, dist = nearest(labs, all_centers)
        sums = np.zeros((len(all_centers), 3), np.float64)
        np.add.at(sums, idx, labs * w[:, None])
        cnt = np.bincount(idx, weights=w, minlength=len(all_centers))
        moved = 0.0
        for j in range(nf, len(all_centers)):
            if cnt[j] > 0:
                nc = (sums[j] / cnt[j]).astype(np.float32)
                moved = max(moved, float(((nc - all_centers[j]) ** 2).sum()))
                all_centers[j] = nc
            else:
                # centro morto -> realoca para a cor com maior erro ponderado
                worst = int(np.argmax(dist * w))
                all_centers[j] = labs[worst]
                dist[worst] = 0
                moved = 1.0
        if moved < 1e-9:
            break

    new_rgb = oklab_to_srgb(all_centers[nf:])
    pal.colors[free] = new_rgb
    pal.locked[free] = True
    return _finalize(pal, base, free)


def _finalize(pal: Palette, base: Optional[Palette], new_slots: np.ndarray) -> Palette:
    """Paleta nova: ordena as cores por luminosidade/matiz (fica organizada para edição)."""
    if base is None and len(new_slots):
        lab = srgb_to_oklab(pal.colors[new_slots])
        hue = np.arctan2(lab[:, 2], lab[:, 1])
        chroma = np.hypot(lab[:, 1], lab[:, 2])
        hue_bucket = np.where(chroma < 0.03, -10, np.floor((hue + np.pi) / (2 * np.pi) * 12))
        order = np.lexsort((lab[:, 0], hue_bucket))
        pal.colors[new_slots] = pal.colors[new_slots][order]
    return pal


# ------------------------------------------------------------------- remapeamento
def remap(rgb: np.ndarray, mask: np.ndarray, pal: Palette, dither: float = 0.0) -> np.ndarray:
    """Converte a imagem para índices da paleta (uint8 HxW)."""
    ti = pal.transparent_index
    valid = np.ones(256, bool)
    valid[ti] = False
    valid &= pal.locked  # só slots definidos (travados ou preenchidos pela quantização)
    cand = np.where(valid)[0]
    plab = srgb_to_oklab(pal.colors[cand])
    out = np.full(mask.shape, ti, np.uint8)

    if dither <= 0:
        lab = srgb_to_oklab(rgb[mask])
        idx, _ = nearest(lab, plab)
        out[mask] = cand[idx]
        return out

    # Floyd-Steinberg serpentina em RGB, busca do vizinho mais próximo em Oklab com cache
    h, w = mask.shape
    # Listas Python puras: bem mais rápido que indexar numpy pixel a pixel.
    R = rgb[..., 0].astype(np.float64).tolist()
    G = rgb[..., 1].astype(np.float64).tolist()
    B = rgb[..., 2].astype(np.float64).tolist()
    M = mask.tolist()
    pc = pal.colors.astype(np.float64).tolist()
    res = out.tolist()

    # cache pré-preenchido com o vizinho exato de todas as cores originais
    keys_src = (rgb[mask].astype(np.int64) * np.array([65536, 256, 1])).sum(1)
    uk, first = np.unique(keys_src, return_index=True)
    ucols = rgb[mask][first]
    uidx, _ = nearest(srgb_to_oklab(ucols), plab)
    cache: dict[int, int] = dict(zip(uk.tolist(), cand[uidx].tolist()))

    lut = _rgb_lut(pal.colors[cand].tobytes(), cand.tobytes())

    def lookup_slow(key, r, g, b):
        v = lut[((r >> 2) << 12) | ((g >> 2) << 6) | (b >> 2)]
        cache[key] = v
        return v

    k7, k3, k5, k1 = 7 / 16 * dither, 3 / 16 * dither, 5 / 16 * dither, 1 / 16 * dither
    LIM = 48.0  # limita o erro para evitar "manchas" em bordas fortes
    for y in range(h):
        my = M[y]
        if not any(my):
            continue
        my1 = M[y + 1] if y + 1 < h else None
        Ry, Gy, By = R[y], G[y], B[y]
        if my1 is not None:
            Ry1, Gy1, By1 = R[y + 1], G[y + 1], B[y + 1]
        if y % 2 == 0:
            xs, dx = range(w), 1
        else:
            xs, dx = range(w - 1, -1, -1), -1
        ry = res[y]
        for x in xs:
            if not my[x]:
                continue
            r = Ry[x]; g = Gy[x]; b = By[x]
            r = 0.0 if r < 0 else (255.0 if r > 255 else r)
            g = 0.0 if g < 0 else (255.0 if g > 255 else g)
            b = 0.0 if b < 0 else (255.0 if b > 255 else b)
            ir, ig, ib = int(r + 0.5), int(g + 0.5), int(b + 0.5)
            key = (ir << 16) | (ig << 8) | ib
            i = cache.get(key)
            if i is None:
                i = lookup_slow(key, ir, ig, ib)
            ry[x] = i
            p = pc[i]
            er = r - p[0]; eg = g - p[1]; eb = b - p[2]
            er = -LIM if er < -LIM else (LIM if er > LIM else er)
            eg = -LIM if eg < -LIM else (LIM if eg > LIM else eg)
            eb = -LIM if eb < -LIM else (LIM if eb > LIM else eb)
            nx = x + dx
            if 0 <= nx < w and my[nx]:
                Ry[nx] += er * k7; Gy[nx] += eg * k7; By[nx] += eb * k7
            if my1 is not None:
                px = x - dx
                if 0 <= px < w and my1[px]:
                    Ry1[px] += er * k3; Gy1[px] += eg * k3; By1[px] += eb * k3
                if my1[x]:
                    Ry1[x] += er * k5; Gy1[x] += eg * k5; By1[x] += eb * k5
                if 0 <= nx < w and my1[nx]:
                    Ry1[nx] += er * k1; Gy1[nx] += eg * k1; By1[nx] += eb * k1
    return np.array(res, np.uint8)


def save_png(indices: np.ndarray, pal: Palette, path: str) -> None:
    im = Image.new("P", (indices.shape[1], indices.shape[0]))
    im.frombytes(np.ascontiguousarray(indices, np.uint8).tobytes())
    im.putpalette(pal.flat())
    im.save(path, format="PNG", transparency=pal.transparent_index, optimize=False)


def indexed_preview(indices: np.ndarray, pal: Palette) -> Image.Image:
    rgba = np.zeros(indices.shape + (4,), np.uint8)
    rgba[..., :3] = pal.colors[indices]
    rgba[..., 3] = np.where(indices == pal.transparent_index, 0, 255)
    return Image.fromarray(rgba, "RGBA")


_LUT_MEMO: dict = {}


def _rgb_lut(colors_bytes: bytes, cand_bytes: bytes) -> list:
    """Tabela 64x64x64 (RGB 6 bits) -> índice mais próximo em Oklab. Memorizada por paleta."""
    key = (colors_bytes, cand_bytes)
    lut = _LUT_MEMO.get(key)
    if lut is None:
        cols = np.frombuffer(colors_bytes, np.uint8).reshape(-1, 3)
        cand = np.frombuffer(cand_bytes, np.intp)
        g = (np.arange(64) * 4 + 2).clip(0, 255)
        grid = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
        idx, _ = nearest(srgb_to_oklab(grid), srgb_to_oklab(cols))
        lut = cand[idx].tolist()
        _LUT_MEMO.clear()
        _LUT_MEMO[key] = lut
    return lut


def quality(rgb: np.ndarray, mask: np.ndarray, indices: np.ndarray, pal: Palette) -> dict:
    """Métricas: ΔE Oklab médio (x100) e PSNR em RGB, só nos pixels opacos."""
    if not mask.any():
        return {"delta_e": 0.0, "psnr": float("inf"), "colors": 0}
    a = rgb[mask].astype(np.float32)
    b = pal.colors[indices[mask]].astype(np.float32)
    mse = float(((a - b) ** 2).mean())
    psnr = float("inf") if mse == 0 else 10 * np.log10(255 ** 2 / mse)
    de = float(np.sqrt(((srgb_to_oklab(a) - srgb_to_oklab(b)) ** 2).sum(1)).mean() * 100)
    return {"delta_e": de, "psnr": psnr, "colors": int(len(np.unique(indices[mask])))}


# ---------------------------------------------------------------- pipeline simples
def convert_batch(images: Iterable[Image.Image], base: Optional[Palette] = None,
                  opts: ConvertOptions = ConvertOptions(), progress: ProgressFn = None):
    prepared = [prepare_image(im, opts) for im in images]
    pal = build_palette(prepared, base, opts, progress)
    results = []
    for n, (rgb, mask) in enumerate(prepared):
        if progress:
            progress(f"Indexando chuteira {n + 1}/{len(prepared)}…", 0.8 + 0.2 * n / max(1, len(prepared)))
        idx = remap(rgb, mask, pal, opts.dither)
        results.append((idx, quality(rgb, mask, idx, pal)))
    return pal, prepared, results
