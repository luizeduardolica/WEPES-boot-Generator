"""Interface gráfica do WEPES Boot Generator (PySide6)."""
from __future__ import annotations

import os
import sys
import traceback

import numpy as np
from PIL import Image
from PySide6.QtCore import Qt, QThread, Signal, QSize, QSettings, QRect, QPoint
from PySide6.QtGui import (QAction, QColor, QIcon, QImage, QPainter, QPixmap, QPen, QBrush,
                           QDragEnterEvent, QDropEvent)
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QListWidget, QListWidgetItem,
                               QLabel, QPushButton, QHBoxLayout, QVBoxLayout, QGroupBox,
                               QRadioButton, QSlider, QCheckBox, QComboBox, QFileDialog,
                               QMessageBox, QProgressBar, QSplitter, QLineEdit, QScrollArea,
                               QButtonGroup, QToolBar, QInputDialog, QStyle, QSizePolicy)

from . import __version__
from .quantize import (ConvertOptions, Palette, prepare_image, build_palette, remap, quality,
                       save_png, indexed_preview)
from .palette_io import load_palette, save_palette, PALETTE_FILTER

IMAGE_EXTS = (".png", ".bmp", ".tga", ".dds", ".gif", ".jpg", ".jpeg", ".tif", ".tiff", ".webp")


# ------------------------------------------------------------------ utilidades Qt
def pil_to_qimage(img: Image.Image) -> QImage:
    img = img.convert("RGBA")
    data = img.tobytes("raw", "RGBA")
    q = QImage(data, img.width, img.height, img.width * 4, QImage.Format_RGBA8888)
    return q.copy()


def checker_pixmap(w: int, h: int, cell: int = 8) -> QPixmap:
    pm = QPixmap(w, h)
    pm.fill(QColor(200, 200, 200))
    p = QPainter(pm)
    for y in range(0, h, cell):
        for x in range(0, w, cell):
            if (x // cell + y // cell) % 2:
                p.fillRect(x, y, cell, cell, QColor(150, 150, 150))
    p.end()
    return pm


class BootItem:
    def __init__(self, path: str, root: str | None = None):
        self.path = path
        # root = pasta principal aberta; rel_dir = subpasta (ex.: marca) relativa a ela
        self.root = os.path.normpath(root or os.path.dirname(path))
        rel = os.path.relpath(os.path.dirname(path), self.root)
        self.rel_dir = "" if rel == "." else rel
        self.name = os.path.splitext(os.path.basename(path))[0]
        with Image.open(path) as im:
            im.load()
            self.source = im.copy()
        self.prepared = None      # (rgb, mask)
        self.indices = None       # resultado indexado
        self.metrics = None


# ------------------------------------------------------------------ trabalhador
class ConvertWorker(QThread):
    progress = Signal(str, float)
    done = Signal(object, object)   # palette, list[(indices, metrics)]
    failed = Signal(str)

    def __init__(self, items, base, mode, opts):
        super().__init__()
        self.items, self.base, self.mode, self.opts = items, base, mode, opts

    def run(self):
        try:
            prepared = []
            for i, it in enumerate(self.items):
                self.progress.emit(f"Preparando {it.name}…", 0.05 * i / len(self.items))
                prepared.append(prepare_image(it.source, self.opts))
            if self.mode == "existing" and self.base is not None:
                pal = self.base.copy()
            else:
                base = self.base if self.mode == "extend" else None
                pal = build_palette(prepared, base, self.opts, lambda m, f: self.progress.emit(m, f))
            results = []
            for i, (rgb, mask) in enumerate(prepared):
                self.progress.emit(f"Indexando {self.items[i].name} ({i + 1}/{len(prepared)})…",
                                   0.8 + 0.2 * i / len(prepared))
                idx = remap(rgb, mask, pal, self.opts.dither)
                results.append(((rgb, mask), idx, quality(rgb, mask, idx, pal)))
            self.done.emit(pal, results)
        except Exception:
            self.failed.emit(traceback.format_exc())


# ------------------------------------------------------------------ widgets
class DropList(QListWidget):
    files_dropped = Signal(list)

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setIconSize(QSize(48, 72))
        self.setSelectionMode(QListWidget.ExtendedSelection)
        self.setMinimumWidth(260)

    def dragEnterEvent(self, e: QDragEnterEvent):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dragMoveEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e: QDropEvent):
        self.files_dropped.emit([u.toLocalFile() for u in e.mimeData().urls()])

    def paintEvent(self, e):
        super().paintEvent(e)
        if self.count() == 0:
            p = QPainter(self.viewport())
            p.setPen(QColor(130, 130, 130))
            p.drawText(self.viewport().rect(), Qt.AlignCenter,
                       "Arraste aqui as chuteiras\ndo PES 6 PC (PNG)\n\nou arquivos/pastas inteiras")


class PaletteView(QWidget):
    """Grade 16x16 da paleta. Borda clara = cor travada (já existia); ponto = usada pela chuteira."""
    index_clicked = Signal(int)

    def __init__(self):
        super().__init__()
        self.pal: Palette | None = None
        self.base_locked = np.zeros(256, bool)
        self.used = np.zeros(256, bool)
        self.selected = -1
        self.setFixedSize(16 * 17 + 1, 16 * 17 + 1)
        self.setMouseTracking(True)

    def set_palette(self, pal, base_locked=None, used=None):
        self.pal = pal
        self.base_locked = base_locked if base_locked is not None else np.zeros(256, bool)
        self.used = used if used is not None else np.zeros(256, bool)
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(40, 40, 40))
        if self.pal is None:
            p.setPen(QColor(160, 160, 160))
            p.drawText(self.rect(), Qt.AlignCenter, "Sem paleta")
            return
        for i in range(256):
            x, y = (i % 16) * 17 + 1, (i // 16) * 17 + 1
            r = QRect(x, y, 16, 16)
            defined = self.pal.locked[i]
            if i == self.pal.transparent_index:
                p.drawPixmap(r, checker_pixmap(16, 16, 4))
            elif defined:
                c = self.pal.colors[i]
                p.fillRect(r, QColor(int(c[0]), int(c[1]), int(c[2])))
            else:
                p.fillRect(r, QColor(60, 60, 60))
                p.setPen(QColor(90, 90, 90))
                p.drawLine(r.topLeft(), r.bottomRight())
            if self.base_locked[i] and i != self.pal.transparent_index:
                p.setBrush(Qt.NoBrush)
                p.setPen(QPen(QColor(255, 255, 255, 140), 1))
                p.drawRect(r.adjusted(0, 0, -1, -1))
            if self.used[i]:
                c = self.pal.colors[i]
                lum = 0.3 * c[0] + 0.59 * c[1] + 0.11 * c[2]
                p.setBrush(QColor(0, 0, 0) if lum > 128 else QColor(255, 255, 255))
                p.setPen(Qt.NoPen)
                p.drawEllipse(QPoint(x + 8, y + 8), 2, 2)
            if i == self.selected:
                p.setBrush(Qt.NoBrush)
                p.setPen(QPen(QColor(255, 0, 255), 2))
                p.drawRect(r.adjusted(-1, -1, 0, 0))

    def _index_at(self, pos):
        i = (pos.y() - 1) // 17 * 16 + (pos.x() - 1) // 17
        return i if 0 <= i < 256 else -1

    def mouseMoveEvent(self, e):
        i = self._index_at(e.position().toPoint())
        if self.pal is not None and i >= 0:
            c = self.pal.colors[i]
            status = "transparente" if i == self.pal.transparent_index else (
                "definida" if self.pal.locked[i] else "livre")
            self.setToolTip(f"Índice {i}: RGB({c[0]}, {c[1]}, {c[2]}) — {status}"
                            + (" — travada (paleta carregada)" if self.base_locked[i] else ""))

    def mousePressEvent(self, e):
        i = self._index_at(e.position().toPoint())
        self.selected = -1 if i == self.selected else i
        self.update()
        self.index_clicked.emit(self.selected)


class PreviewPane(QLabel):
    def __init__(self, title):
        super().__init__()
        self.title = title
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(256, 256)
        self.setStyleSheet("QLabel{background:#2b2b2b;border:1px solid #444;color:#999}")
        self.setText(title)

    def show_image(self, img: Image.Image | None, zoom: int):
        if img is None:
            self.setPixmap(QPixmap())
            self.setText(self.title)
            return
        q = pil_to_qimage(img)
        w, h = q.width() * zoom, q.height() * zoom
        pm = checker_pixmap(w, h, 8)
        p = QPainter(pm)
        p.drawImage(QRect(0, 0, w, h), q)
        p.setPen(QColor(255, 255, 255, 60))
        p.end()
        self.setPixmap(pm)
        self.setMinimumSize(w + 2, h + 2)


# ------------------------------------------------------------------ janela
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"WEPES Boot Generator v{__version__} — PES 6 PC ➜ PES PS2")
        self.resize(1320, 820)
        self.settings = QSettings("WEPES", "BootGenerator")
        self.items: list[BootItem] = []
        self.loaded_palette: Palette | None = None
        self.result_palette: Palette | None = None
        self.worker = None
        self.highlight = -1
        self._build_ui()
        self._refresh_palette_view()

    # ---------------------------------------------------------------- UI
    def _build_ui(self):
        tb = QToolBar()
        tb.setMovable(False)
        tb.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.addToolBar(tb)
        st = self.style()

        def act(text, icon, slot, tip=""):
            a = QAction(st.standardIcon(icon), text, self)
            a.triggered.connect(slot)
            a.setToolTip(tip or text)
            tb.addAction(a)
            return a

        act("Adicionar chuteiras", QStyle.SP_FileIcon, self.add_dialog)
        act("Abrir pasta (com subpastas)", QStyle.SP_DirOpenIcon, self.add_folder_dialog,
            "Lê a pasta principal e TODAS as subpastas (ex.: uma pasta por marca) procurando chuteiras")
        act("Remover", QStyle.SP_TrashIcon, self.remove_selected, "Remove as chuteiras selecionadas")
        act("Limpar", QStyle.SP_DialogResetButton, self.clear_all)
        tb.addSeparator()
        act("Carregar paleta…", QStyle.SP_DirOpenIcon, self.load_palette_dialog,
            "Carrega a paleta de chuteiras PS2 já existentes (PNG/BMP indexado, .act, .pal, .gpl)")
        act("Salvar paleta…", QStyle.SP_DialogSaveButton, self.save_palette_dialog)
        tb.addSeparator()
        self.act_convert = act("Converter", QStyle.SP_MediaPlay, self.convert,
                               "Gera a paleta compartilhada e indexa todas as chuteiras")
        self.act_save = act("Salvar PNGs PS2", QStyle.SP_DialogSaveButton, self.save_outputs)

        # ----- lista
        self.list = DropList()
        self.list.files_dropped.connect(self.add_paths)
        self.list.currentRowChanged.connect(self.show_current)

        # ----- pré-visualização
        self.prev_src = PreviewPane("Original (PES 6 PC)")
        self.prev_dst = PreviewPane("Resultado (PES PS2, 256 cores)")
        self.zoom = QComboBox()
        self.zoom.addItems(["1x", "2x", "3x", "4x"])
        self.zoom.setCurrentIndex(1)
        self.zoom.currentIndexChanged.connect(lambda _: self.show_current(self.list.currentRow()))
        self.lbl_info = QLabel("")
        self.lbl_info.setWordWrap(True)
        prev_row = QHBoxLayout()
        holder = QWidget()
        hl = QHBoxLayout(holder)
        hl.addWidget(self.prev_src)
        hl.addWidget(self.prev_dst)
        sa = QScrollArea()
        sa.setWidget(holder)
        sa.setWidgetResizable(True)
        prev_row.addWidget(sa)
        zrow = QHBoxLayout()
        zrow.addWidget(QLabel("Zoom:"))
        zrow.addWidget(self.zoom)
        zrow.addSpacing(16)
        zrow.addWidget(self.lbl_info, 1)
        center = QWidget()
        cl = QVBoxLayout(center)
        cl.addLayout(prev_row, 1)
        cl.addLayout(zrow)

        # ----- painel direito
        right = QWidget()
        right.setMinimumWidth(330)
        right.setMaximumWidth(400)
        rl = QVBoxLayout(right)

        g_pal = QGroupBox("Paleta")
        gl = QVBoxLayout(g_pal)
        self.rb_new = QRadioButton("Gerar paleta nova (compartilhada por todas)")
        self.rb_extend = QRadioButton("Usar paleta carregada + preencher slots livres")
        self.rb_existing = QRadioButton("Usar SOMENTE as cores da paleta carregada")
        self.rb_new.setChecked(True)
        self.grp_mode = QButtonGroup(self)
        for rb in (self.rb_new, self.rb_extend, self.rb_existing):
            self.grp_mode.addButton(rb)
            gl.addWidget(rb)
        self.rb_extend.setToolTip("As cores existentes NÃO mudam de posição: as chuteiras que já estão no jogo\n"
                                  "continuam iguais. Apenas os índices livres recebem as cores novas.")
        self.rb_existing.setToolTip("Não altera nada na paleta: a nova chuteira é indexada apenas\n"
                                    "com as cores que já existem (ideal para trocar só uma chuteira).")
        self.cb_lock_used = QCheckBox("Ao carregar PNG: travar só os índices usados")
        self.cb_lock_used.setToolTip("Marque se a paleta tem entradas sobrando que nenhuma chuteira do jogo usa.\n"
                                     "Dica: selecione TODAS as chuteiras do jogo que usam essa paleta ao carregar,\n"
                                     "assim os índices usados por qualquer uma delas ficam protegidos.")
        gl.addWidget(self.cb_lock_used)
        self.lbl_pal = QLabel("Paleta carregada: nenhuma")
        self.lbl_pal.setWordWrap(True)
        gl.addWidget(self.lbl_pal)
        self.pal_view = PaletteView()
        self.pal_view.index_clicked.connect(self._on_index_clicked)
        gl.addWidget(self.pal_view, 0, Qt.AlignHCenter)
        hint = QLabel("<small>Clique numa cor para destacar onde ela é usada. "
                      "Borda branca = cor travada; ponto = usada pela chuteira selecionada.</small>")
        hint.setWordWrap(True)
        gl.addWidget(hint)
        rl.addWidget(g_pal)

        g_q = QGroupBox("Qualidade")
        ql = QVBoxLayout(g_q)
        drow = QHBoxLayout()
        drow.addWidget(QLabel("Dithering:"))
        self.sl_dither = QSlider(Qt.Horizontal)
        self.sl_dither.setRange(0, 100)
        self.sl_dither.setValue(int(self.settings.value("dither", 0)))
        self.lbl_dither = QLabel()
        self.sl_dither.valueChanged.connect(lambda v: self.lbl_dither.setText(f"{v}%"))
        self.lbl_dither.setText(f"{self.sl_dither.value()}%")
        drow.addWidget(self.sl_dither, 1)
        drow.addWidget(self.lbl_dither)
        ql.addLayout(drow)
        self.sl_dither.setToolTip("0% = cores sólidas (texturas mais limpas).\n"
                                  "30–60% = degradês mais suaves quando há muitas chuteiras na mesma paleta.")
        self.cb_balanced = QCheckBox("Peso igual para cada chuteira")
        self.cb_balanced.setChecked(True)
        self.cb_balanced.setToolTip("Evita que uma chuteira com poucas cores fique mal representada na paleta.")
        ql.addWidget(self.cb_balanced)
        rrow = QHBoxLayout()
        rrow.addWidget(QLabel("Tamanho ≠ 170x256:"))
        self.cmb_resize = QComboBox()
        self.cmb_resize.addItems(["Escalar p/ altura 256 + alinhar à esquerda", "Esticar para 256x256"])
        rrow.addWidget(self.cmb_resize, 1)
        ql.addLayout(rrow)
        rl.addWidget(g_q)

        g_out = QGroupBox("Saída")
        ol = QVBoxLayout(g_out)
        out_hint = QLabel("<small>Ao salvar, você escolhe onde criar a nova pasta. Ela terá a mesma "
                          "estrutura de subpastas (marcas) da pasta aberta.</small>")
        out_hint.setWordWrap(True)
        ol.addWidget(out_hint)
        srow = QHBoxLayout()
        srow.addWidget(QLabel("Sufixo do nome:"))
        self.ed_suffix = QLineEdit(self.settings.value("suffix", ""))
        self.ed_suffix.setPlaceholderText("ex.: _ps2")
        srow.addWidget(self.ed_suffix, 1)
        ol.addLayout(srow)
        self.cb_save_pal = QCheckBox("Salvar também a paleta (.act + .pal + PNG)")
        self.cb_save_pal.setChecked(True)
        ol.addWidget(self.cb_save_pal)
        rl.addWidget(g_out)
        rl.addStretch(1)

        split = QSplitter()
        split.addWidget(self.list)
        split.addWidget(center)
        right_sa = QScrollArea()
        right_sa.setWidget(right)
        right_sa.setWidgetResizable(True)
        right_sa.setMinimumWidth(350)
        right_sa.setMaximumWidth(420)
        split.addWidget(right_sa)
        split.setStretchFactor(1, 1)
        split.setSizes([280, 660, 380])
        self.setCentralWidget(split)

        self.progress = QProgressBar()
        self.progress.setMaximumWidth(260)
        self.progress.setVisible(False)
        self.statusBar().addPermanentWidget(self.progress)
        self.statusBar().showMessage("Arraste as chuteiras do PES 6 PC para a lista.")

    # ---------------------------------------------------------------- lista
    def add_dialog(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Adicionar chuteiras", self.settings.value("indir", ""),
                                                "Imagens (*.png *.bmp *.tga *.dds *.gif *.jpg *.tif *.webp)")
        if files:
            self.settings.setValue("indir", os.path.dirname(files[0]))
            self.add_paths(files)

    def add_folder_dialog(self):
        d = QFileDialog.getExistingDirectory(self, "Abrir pasta principal (todas as subpastas serão lidas)",
                                             self.settings.value("indir", ""))
        if d:
            self.settings.setValue("indir", d)
            self.add_paths([d])

    def add_paths(self, paths):
        files = []  # (arquivo, pasta raiz)
        for p in paths:
            p = os.path.normpath(p)
            if os.path.isdir(p):
                # busca recursiva: pasta principal -> pastas das marcas -> imagens
                for dirpath, dirnames, filenames in os.walk(p):
                    dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
                    for f in sorted(filenames):
                        if f.lower().endswith(IMAGE_EXTS):
                            files.append((os.path.join(dirpath, f), p))
            elif p.lower().endswith(IMAGE_EXTS):
                files.append((p, None))
        known = {os.path.normcase(it.path) for it in self.items}
        errors = []
        self.progress.setVisible(True)
        self.progress.setRange(0, max(1, len(files)))
        for n, (f, root) in enumerate(files):
            if os.path.normcase(f) in known:
                continue
            if n % 10 == 0:
                self.progress.setValue(n)
                self.statusBar().showMessage(f"Carregando {n + 1}/{len(files)}: {os.path.basename(f)}")
                QApplication.processEvents()
            try:
                it = BootItem(f, root)
            except Exception as ex:
                errors.append(f"{os.path.relpath(f, root) if root else os.path.basename(f)}: {ex}")
                continue
            self.items.append(it)
            known.add(os.path.normcase(f))
            li = QListWidgetItem(self._icon_for(it.source), self._item_text(it))
            li.setToolTip(it.path)
            self.list.addItem(li)
        self.progress.setVisible(False)
        folders = len({(it.root, it.rel_dir) for it in self.items})
        self.statusBar().showMessage(f"{len(self.items)} chuteiras em {folders} pasta(s).")
        if errors:
            QMessageBox.warning(self, "Arquivos ignorados", "\n".join(errors[:40]))
        self._invalidate()
        if self.list.currentRow() < 0 and self.items:
            self.list.setCurrentRow(0)

    def _icon_for(self, img):
        t = img.convert("RGBA").copy()
        t.thumbnail((48, 72))
        return QIcon(QPixmap.fromImage(pil_to_qimage(t)))

    def _item_text(self, it: BootItem):
        w, h = it.source.size
        s = f"{it.name}\n"
        if it.rel_dir:
            s += f"Pasta: {it.rel_dir}\n"
        s += f"{w}x{h}"
        if (w, h) == (170, 256):
            s += " (PES 6 PC)"
        elif (w, h) == (256, 256):
            s += " (já 256x256)"
        else:
            s += " (tamanho diferente — será ajustado)"
        if it.metrics:
            s += f"\nΔE {it.metrics['delta_e']:.2f} · {it.metrics['colors']} cores"
        return s

    def remove_selected(self):
        rows = sorted((self.list.row(i) for i in self.list.selectedItems()), reverse=True)
        for r in rows:
            self.list.takeItem(r)
            del self.items[r]
        self._invalidate()

    def clear_all(self):
        self.list.clear()
        self.items.clear()
        self._invalidate()

    def _invalidate(self):
        self.result_palette = None
        for i, it in enumerate(self.items):
            it.indices, it.metrics = None, None
            self.list.item(i).setText(self._item_text(it))
        self._refresh_palette_view()
        self.show_current(self.list.currentRow())

    # ---------------------------------------------------------------- paleta
    def load_palette_dialog(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Carregar paleta (selecione uma ou TODAS as chuteiras PS2 que usam a paleta)",
            self.settings.value("paldir", ""), PALETTE_FILTER)
        if not files:
            return
        self.settings.setValue("paldir", os.path.dirname(files[0]))
        try:
            pal = load_palette(files, "used" if self.cb_lock_used.isChecked() else "all")
        except Exception as ex:
            QMessageBox.critical(self, "Erro ao carregar paleta", str(ex))
            return
        self.loaded_palette = pal
        self.rb_extend.setChecked(True)
        self._invalidate()

    def save_palette_dialog(self):
        pal = self.result_palette or self.loaded_palette
        if pal is None:
            QMessageBox.information(self, "Salvar paleta", "Converta as chuteiras ou carregue uma paleta primeiro.")
            return
        f, _ = QFileDialog.getSaveFileName(self, "Salvar paleta", "paleta_chuteiras.act",
                                           "Photoshop ACT (*.act);;JASC PAL (*.pal);;GIMP (*.gpl);;PNG 16x16 (*.png)")
        if f:
            save_palette(pal, f)
            self.statusBar().showMessage(f"Paleta salva em {f}", 5000)

    def _refresh_palette_view(self):
        lp = self.loaded_palette
        if lp is not None:
            n_locked = int(lp.locked.sum()) - 1
            self.lbl_pal.setText(f"Paleta carregada: {lp.source}\n{n_locked} cores travadas · "
                                 f"{lp.free_slots} slots livres · transparente = índice {lp.transparent_index}")
        else:
            self.lbl_pal.setText("Paleta carregada: nenhuma")
        self.rb_extend.setEnabled(lp is not None)
        self.rb_existing.setEnabled(lp is not None)
        if lp is None and not self.rb_new.isChecked():
            self.rb_new.setChecked(True)
        pal = self.result_palette or lp
        base_locked = lp.locked if (lp is not None and not self.rb_new.isChecked()) else None
        used = None
        cur = self.list.currentRow()
        if self.result_palette is not None and 0 <= cur < len(self.items) and self.items[cur].indices is not None:
            used = np.zeros(256, bool)
            idx = self.items[cur].indices
            used[np.unique(idx)] = True
            used[self.result_palette.transparent_index] = False
        self.pal_view.set_palette(pal, base_locked, used)

    def _on_index_clicked(self, i):
        self.highlight = i
        self.show_current(self.list.currentRow())

    # ---------------------------------------------------------------- preview
    def show_current(self, row):
        z = self.zoom.currentIndex() + 1
        if row is None or row < 0 or row >= len(self.items):
            self.prev_src.show_image(None, z)
            self.prev_dst.show_image(None, z)
            self.lbl_info.setText("")
            return
        it = self.items[row]
        self.prev_src.show_image(it.source, z)
        if it.indices is not None and self.result_palette is not None:
            img = indexed_preview(it.indices, self.result_palette)
            if self.highlight >= 0:
                a = np.array(img)
                m = it.indices == self.highlight
                a[m] = (255, 0, 255, 255)
                img = Image.fromarray(a, "RGBA")
            self.prev_dst.show_image(img, z)
            m = it.metrics
            self.lbl_info.setText(f"<b>{it.name}</b> — {m['colors']} cores da paleta usadas · "
                                  f"ΔE médio {m['delta_e']:.2f} (abaixo de ~2 é imperceptível) · "
                                  f"PSNR {m['psnr']:.1f} dB")
        else:
            self.prev_dst.show_image(None, z)
            self.lbl_info.setText(f"<b>{it.name}</b> — clique em <i>Converter</i> para gerar a versão PS2.")
        self._refresh_palette_view()

    # ---------------------------------------------------------------- conversão
    def _mode(self):
        if self.rb_existing.isChecked():
            return "existing"
        if self.rb_extend.isChecked():
            return "extend"
        return "new"

    def _options(self):
        return ConvertOptions(resize_mode="stretch" if self.cmb_resize.currentIndex() == 1 else "pad",
                              dither=self.sl_dither.value() / 100.0,
                              balanced=self.cb_balanced.isChecked())

    def convert(self):
        if not self.items:
            QMessageBox.information(self, "Converter", "Adicione pelo menos uma chuteira.")
            return
        if self.worker is not None:
            return
        mode = self._mode()
        if mode != "new" and self.loaded_palette is None:
            mode = "new"
        if mode == "extend" and self.loaded_palette.free_slots == 0:
            r = QMessageBox.question(self, "Paleta cheia",
                                     "A paleta carregada não tem slots livres. As chuteiras serão indexadas "
                                     "somente com as cores existentes. Continuar?")
            if r != QMessageBox.Yes:
                return
            mode = "existing"
        self.settings.setValue("dither", self.sl_dither.value())
        self.worker = ConvertWorker(list(self.items), self.loaded_palette, mode, self._options())
        self.worker.progress.connect(self._on_progress)
        self.worker.done.connect(self._on_done)
        self.worker.failed.connect(self._on_failed)
        self.worker.finished.connect(self._on_finished)
        self.act_convert.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, 1000)
        self.worker.start()

    def _on_progress(self, msg, frac):
        self.statusBar().showMessage(msg)
        self.progress.setValue(int(frac * 1000))

    def _on_done(self, pal, results):
        self.result_palette = pal
        for i, (prep, idx, met) in enumerate(results):
            it = self.items[i]
            it.prepared, it.indices, it.metrics = prep, idx, met
            self.list.item(i).setText(self._item_text(it))
        used_total = int(pal.locked.sum()) - 1
        wi = max(range(len(results)), key=lambda k: results[k][2]["delta_e"])
        worst = results[wi]
        msg = (f"Pronto: {len(results)} chuteiras com a mesma paleta ({used_total} cores). "
               f"Pior caso: {self.items[wi].name} (ΔE {worst[2]['delta_e']:.2f}).")
        if self._mode() == "extend" and self.loaded_palette is not None:
            added = int((pal.locked & ~self.loaded_palette.locked).sum())
            msg += f" {added} cores novas adicionadas nos slots livres — importe a paleta nova no jogo."
        self.statusBar().showMessage(msg)
        self.show_current(self.list.currentRow())

    def _on_failed(self, tb):
        QMessageBox.critical(self, "Erro na conversão", tb)

    def _on_finished(self):
        self.worker = None
        self.act_convert.setEnabled(True)
        self.progress.setVisible(False)

    # ---------------------------------------------------------------- saída
    def save_outputs(self):
        if self.result_palette is None or any(it.indices is None for it in self.items):
            if not self.items:
                return
            QMessageBox.information(self, "Salvar", "Clique em Converter antes de salvar.")
            return
        dest = QFileDialog.getExistingDirectory(
            self, "Onde criar a nova pasta com as chuteiras PS2?",
            self.settings.value("savedir", os.path.dirname(self.items[0].root)))
        if not dest:
            return
        self.settings.setValue("savedir", dest)
        suffix = self.ed_suffix.text()
        self.settings.setValue("suffix", suffix)

        # uma pasta nova para cada pasta principal aberta, com a mesma estrutura de subpastas
        roots = list(dict.fromkeys(it.root for it in self.items))
        out_for_root = {}
        for root in roots:
            default = (os.path.basename(root) or "Chuteiras") + " PS2"
            if len(roots) == 1:
                name, ok = QInputDialog.getText(self, "Nome da nova pasta",
                                                f"A nova pasta será criada em:\n{dest}\n\nNome:",
                                                text=default)
                if not ok or not name.strip():
                    return
                default = name.strip()
            out_for_root[root] = os.path.join(dest, default)

        targets = [(it, os.path.join(out_for_root[it.root], it.rel_dir, it.name + suffix + ".png"))
                   for it in self.items]
        existing = [t for _, t in targets if os.path.exists(t)]
        if existing:
            r = QMessageBox.question(self, "Sobrescrever?",
                                     f"{len(existing)} arquivo(s) já existem no destino. Sobrescrever?")
            if r != QMessageBox.Yes:
                return
        try:
            for it, t in targets:
                os.makedirs(os.path.dirname(t), exist_ok=True)
                save_png(it.indices, self.result_palette, t)
            if self.cb_save_pal.isChecked():
                for out in out_for_root.values():
                    for ext in (".act", ".pal", ".png"):
                        save_palette(self.result_palette, os.path.join(out, "_paleta_chuteiras" + ext))
        except OSError as ex:
            QMessageBox.critical(self, "Erro ao salvar", str(ex))
            return
        outs = list(out_for_root.values())
        n_dirs = len({os.path.dirname(t) for _, t in targets})
        self.statusBar().showMessage(f"{len(targets)} chuteiras salvas em {n_dirs} pasta(s).", 8000)
        r = QMessageBox.information(self, "Concluído",
                                    f"{len(targets)} chuteiras PS2 (256x256, 8 bits) salvas em {n_dirs} pasta(s):\n"
                                    + "\n".join(outs), QMessageBox.Open | QMessageBox.Ok)
        if r == QMessageBox.Open and sys.platform == "win32":
            os.startfile(outs[0])


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    from PySide6.QtGui import QPalette
    p = QPalette()
    p.setColor(QPalette.Window, QColor(45, 45, 48))
    p.setColor(QPalette.WindowText, QColor(230, 230, 230))
    p.setColor(QPalette.Base, QColor(30, 30, 32))
    p.setColor(QPalette.AlternateBase, QColor(45, 45, 48))
    p.setColor(QPalette.Text, QColor(230, 230, 230))
    p.setColor(QPalette.Button, QColor(55, 55, 60))
    p.setColor(QPalette.ButtonText, QColor(230, 230, 230))
    p.setColor(QPalette.Highlight, QColor(0, 120, 215))
    p.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    p.setColor(QPalette.ToolTipBase, QColor(50, 50, 50))
    p.setColor(QPalette.ToolTipText, QColor(230, 230, 230))
    app.setPalette(p)
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ico = os.path.join(base, "assets", "icon.ico")
    if os.path.exists(ico):
        app.setWindowIcon(QIcon(ico))
    w = MainWindow()
    w.show()
    if len(sys.argv) > 1:
        w.add_paths(sys.argv[1:])
    sys.exit(app.exec())
