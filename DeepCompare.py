import sys
import os
import torch
from sentence_transformers import SentenceTransformer, util

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QFileDialog, QTableWidget, QTableWidgetItem,
    QMessageBox, QHeaderView, QSplashScreen
)
from PyQt6.QtGui import QColor, QPixmap, QPainter, QFont, QPalette
from PyQt6.QtCore import Qt, QThread, pyqtSignal

# グローバル変数としてモデルを保持
model = None
MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L6-v2"

# -------------------------------------------------
# ModelLoader: MINI LM モデルの読み込みを非同期に行うためのスレッド
# -------------------------------------------------
class ModelLoader(QThread):
    loaded = pyqtSignal(object)  # モデルが読み込まれた際にモデルを渡す信号

    def run(self):
        global model
        model = SentenceTransformer(MODEL_NAME)
        self.loaded.emit(model)

# -------------------------------------------------
# FileDropLineEdit: ドラッグ＆ドロップ対応の QLineEdit
# -------------------------------------------------
class FileDropLineEdit(QLineEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls:
                file_path = urls[0].toLocalFile()
                self.setText(file_path)
                event.acceptProposedAction()
        else:
            super().dropEvent(event)

# -------------------------------------------------
# 以下、CUI 版と同様の関数群
# -------------------------------------------------
def get_line_embeddings(code_text: str):
    """
    コードテキストを行ごとに分割し、各行の埋め込みを MINI LM で計算する
    """
    lines = code_text.splitlines()
    embeddings = model.encode(lines, convert_to_tensor=True)
    return lines, embeddings

def compute_similarity_matrix(embeddings1, embeddings2):
    similarity_matrix = util.pytorch_cos_sim(embeddings1, embeddings2)
    return similarity_matrix.tolist()

def align_lines(lines1, lines2, sim_matrix, gap_penalty=-0.5):
    n = len(lines1)
    m = len(lines2)
    dp = [[0.0]*(m+1) for _ in range(n+1)]
    backtrack = [[None]*(m+1) for _ in range(n+1)]

    for i in range(1, n+1):
        dp[i][0] = dp[i-1][0] + gap_penalty
        backtrack[i][0] = "up"
    for j in range(1, m+1):
        dp[0][j] = dp[0][j-1] + gap_penalty
        backtrack[0][j] = "left"

    for i in range(1, n+1):
        for j in range(1, m+1):
            score_diag = dp[i-1][j-1] + sim_matrix[i-1][j-1]
            score_up = dp[i-1][j] + gap_penalty
            score_left = dp[i][j-1] + gap_penalty
            max_score = max(score_diag, score_up, score_left)
            dp[i][j] = max_score
            if max_score == score_diag:
                backtrack[i][j] = "diag"
            elif max_score == score_up:
                backtrack[i][j] = "up"
            else:
                backtrack[i][j] = "left"

    aligned = []
    i, j = n, m
    while i > 0 or j > 0:
        direction = backtrack[i][j]
        if direction == "diag":
            aligned.append((i-1, j-1, sim_matrix[i-1][j-1]))
            i -= 1
            j -= 1
        elif direction == "up":
            aligned.append((i-1, None, None))
            i -= 1
        elif direction == "left":
            aligned.append((None, j-1, None))
            j -= 1
        else:
            break
    aligned.reverse()
    return aligned

# -------------------------------------------------
# DiffWindow: PyQt6 GUI クラス
# -------------------------------------------------
class DiffWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MiniLM ベース コード比較ツール")
        self.resize(1200, 600)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        # --- ファイル選択用ウィジェット ---
        file_layout = QHBoxLayout()
        self.file1_edit = FileDropLineEdit()
        self.file1_edit.setPlaceholderText("ファイル1のパスを入力またはドラッグ＆ドロップ")
        file1_button = QPushButton("参照")
        file1_button.clicked.connect(self.select_file1)
        self.file2_edit = FileDropLineEdit()
        self.file2_edit.setPlaceholderText("ファイル2のパスを入力またはドラッグ＆ドロップ")
        file2_button = QPushButton("参照")
        file2_button.clicked.connect(self.select_file2)

        file_layout.addWidget(QLabel("ファイル1:"))
        file_layout.addWidget(self.file1_edit)
        file_layout.addWidget(file1_button)
        file_layout.addWidget(QLabel("ファイル2:"))
        file_layout.addWidget(self.file2_edit)
        file_layout.addWidget(file2_button)
        main_layout.addLayout(file_layout)

        # --- 比較開始ボタン ---
        compare_button = QPushButton("比較開始")
        compare_button.clicked.connect(self.compare_files)
        main_layout.addWidget(compare_button)

        # --- 結果表示用テーブル ---
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["File1", "File2", "Score"])
        self.table.setShowGrid(False)  # セパレーター（グリッド）非表示
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(2, 60)
        # 行高さ（行間）を小さく設定
        self.table.verticalHeader().setDefaultSectionSize(20)
        main_layout.addWidget(self.table)

    def select_file1(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "ファイル1を選択", "", "Python Files (*.py);;All Files (*)"
        )
        if file_path:
            self.file1_edit.setText(file_path)

    def select_file2(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "ファイル2を選択", "", "Python Files (*.py);;All Files (*)"
        )
        if file_path:
            self.file2_edit.setText(file_path)

    def compare_files(self):
        file1_path = self.file1_edit.text().strip()
        file2_path = self.file2_edit.text().strip()

        if not file1_path or not file2_path:
            QMessageBox.warning(self, "エラー", "両方のファイルパスを指定してください。")
            return

        try:
            with open(file1_path, "r", encoding="utf-8") as f:
                code1 = f.read()
            with open(file2_path, "r", encoding="utf-8") as f:
                code2 = f.read()
        except Exception as e:
            QMessageBox.critical(self, "エラー", f"ファイル読み込み時にエラーが発生しました:\n{e}")
            return

        # --- 各行ごとの埋め込み計算 ---
        lines1, embeddings1 = get_line_embeddings(code1)
        lines2, embeddings2 = get_line_embeddings(code2)
        # --- コサイン類似度マトリックス計算 ---
        sim_matrix = compute_similarity_matrix(embeddings1, embeddings2)
        # --- アライメント計算 ---
        aligned = align_lines(lines1, lines2, sim_matrix, gap_penalty=-0.5)

        # --- テーブルに結果を表示 ---
        self.table.setRowCount(len(aligned))
        for row, (left_idx, right_idx, score) in enumerate(aligned):
            left_text = lines1[left_idx] if left_idx is not None else "---"
            right_text = lines2[right_idx] if right_idx is not None else "---"
            score_text = f"{score:.2f}" if score is not None else ""

            left_item = QTableWidgetItem(left_text)
            right_item = QTableWidgetItem(right_text)
            score_item = QTableWidgetItem(score_text)

            # 左右の行が異なる場合、薄い黄色でハイライト
            if left_text != right_text:
                highlight_color = QColor(255, 255, 150)
                left_item.setBackground(highlight_color)
                right_item.setBackground(highlight_color)

            self.table.setItem(row, 0, left_item)
            self.table.setItem(row, 1, right_item)
            self.table.setItem(row, 2, score_item)

        QMessageBox.information(self, "完了", "比較が完了しました。")

# -------------------------------------------------
# メイン処理
# -------------------------------------------------
if __name__ == '__main__':
    app = QApplication(sys.argv)

    # ダークデザインのためのパレット設定（Fusion スタイル）
    app.setStyle("Fusion")
    dark_palette = QPalette()
    dark_palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ColorRole.WindowText, QColor(255, 255, 255))
    dark_palette.setColor(QPalette.ColorRole.Base, QColor(35, 35, 35))
    dark_palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(255, 255, 255))
    dark_palette.setColor(QPalette.ColorRole.ToolTipText, QColor(255, 255, 255))
    dark_palette.setColor(QPalette.ColorRole.Text, QColor(255, 255, 255))
    dark_palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ColorRole.ButtonText, QColor(255, 255, 255))
    dark_palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 0, 0))
    dark_palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.ColorRole.HighlightedText, QColor(0, 0, 0))
    app.setPalette(dark_palette)

    # --- スプラッシュスクリーンの表示 ---
    splash_pix = QPixmap(300, 100)
    splash_pix.fill(QColor(53, 53, 53))
    painter = QPainter(splash_pix)
    painter.setPen(QColor(255, 255, 255))
    painter.setFont(QFont("Arial", 16))
    painter.drawText(splash_pix.rect(), Qt.AlignmentFlag.AlignCenter, "MINI LM 読み込み中...")
    painter.end()
    splash = QSplashScreen(splash_pix)
    splash.show()
    app.processEvents()  # スプラッシュスクリーンを更新

    # --- モデルの読み込みを非同期に開始 ---
    loader = ModelLoader()
    # メインウィンドウの準備（まだ表示はしない）
    main_window = DiffWindow()

    def on_model_loaded(m):
        splash.finish(main_window)
        main_window.show()

    loader.loaded.connect(on_model_loaded)
    loader.start()

    sys.exit(app.exec())
