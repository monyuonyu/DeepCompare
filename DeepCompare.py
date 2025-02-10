import sys
import os
import difflib
import torch
from sentence_transformers import SentenceTransformer, util

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QFileDialog, QTableWidget, QTableWidgetItem,
    QMessageBox, QHeaderView, QSplashScreen
)
from PyQt6.QtGui import QColor, QPixmap, QPainter, QFont, QPalette, QCursor
from PyQt6.QtCore import Qt, QThread, pyqtSignal

# --- グローバル変数とモデル名 ---
model = None
MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L6-v2"

# =============================================================================
# ModelLoader: MINI LM モデルの非同期読み込み用スレッド
# =============================================================================
class ModelLoader(QThread):
    loaded = pyqtSignal(object)  # 読み込み完了時にモデルを送出

    def run(self):
        global model
        model = SentenceTransformer(MODEL_NAME)
        self.loaded.emit(model)

# =============================================================================
# FileDropLineEdit: ドラッグ＆ドロップ対応の QLineEdit（ダークテーマ用）
# =============================================================================
class FileDropLineEdit(QLineEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        # ダークテーマ用の文字色・背景色設定
        self.setStyleSheet("color: white; background-color: rgb(53,53,53);")

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

# =============================================================================
# 文字レベルの差分抽出関数（SequenceMatcher 使用）
# － 差分部分はフォント色をオレンジ (RGB(255,165,0)) に変更
# =============================================================================
def diff_characters(left, right):
    s = difflib.SequenceMatcher(None, left, right)
    left_html = ""
    right_html = ""
    for tag, i1, i2, j1, j2 in s.get_opcodes():
        if tag == "equal":
            left_html += left[i1:i2]
            right_html += right[j1:j2]
        elif tag == "replace":
            left_html += f"<span style='color: rgb(255,165,0);'>{left[i1:i2]}</span>"
            right_html += f"<span style='color: rgb(255,165,0);'>{right[j1:j2]}</span>"
        elif tag == "delete":
            left_html += f"<span style='color: rgb(255,165,0);'>{left[i1:i2]}</span>"
        elif tag == "insert":
            right_html += f"<span style='color: rgb(255,165,0);'>{right[j1:j2]}</span>"
    return left_html, right_html

# =============================================================================
# ファイル内容の行ごとの埋め込み・アライメント関数
# =============================================================================
def get_line_embeddings(code_text: str):
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

# =============================================================================
# DiffWindow: PyQt6 GUI クラス
# OS標準のウィンドウ枠・タイトルバーを使用するように修正
# =============================================================================
class DiffWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # OS標準のウィンドウ枠を使用するため、FramelessWindowHint を設定しない
        self.setWindowTitle("DeepCompare")
        self.setMinimumSize(800, 600)
        self.full_lines1 = []
        self.full_lines2 = []
        self.alignment = []  # 現在のアライメント結果

        # 中央ウィジェットとレイアウトの設定
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # ファイル選択用ウィジェット
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

        # 比較開始ボタン
        compare_button = QPushButton("比較開始")
        compare_button.clicked.connect(self.compare_files)
        main_layout.addWidget(compare_button)

        # 結果表示用テーブル（5列）
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["行番号", "File1", "行番号", "File2", "Score"])
        self.table.setShowGrid(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(0, 50)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(2, 50)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(4, 60)
        self.table.verticalHeader().setDefaultSectionSize(20)
        main_layout.addWidget(self.table)

    def select_file1(self):
        # ファイル1選択ダイアログの表示
        file_path, _ = QFileDialog.getOpenFileName(self, "ファイル1を選択", "", "Python Files (*.py);;All Files (*)")
        if file_path:
            self.file1_edit.setText(file_path)

    def select_file2(self):
        # ファイル2選択ダイアログの表示
        file_path, _ = QFileDialog.getOpenFileName(self, "ファイル2を選択", "", "Python Files (*.py);;All Files (*)")
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

        self.full_lines1 = code1.splitlines()
        self.full_lines2 = code2.splitlines()

        lines1, emb1 = get_line_embeddings(code1)
        lines2, emb2 = get_line_embeddings(code2)
        sim_matrix = compute_similarity_matrix(emb1, emb2)
        self.alignment = align_lines(lines1, lines2, sim_matrix, gap_penalty=-0.5)
        self.update_table()

    def update_table(self):
        self.table.setRowCount(len(self.alignment))
        # 差分行の背景色（ダークテーマに合わせた青みのある色）
        diff_bg = QColor(80, 100, 130)
        for row, (l_idx, r_idx, score) in enumerate(self.alignment):
            if l_idx is not None:
                left_num = str(l_idx + 1)
                left_text = self.full_lines1[l_idx]
            else:
                left_num = ""
                left_text = "---"
            if r_idx is not None:
                right_num = str(r_idx + 1)
                right_text = self.full_lines2[r_idx]
            else:
                right_num = ""
                right_text = "---"
            score_text = f"{score:.2f}" if score is not None else ""

            # 左ファイル行番号
            item0 = QTableWidgetItem(left_num)
            # 右ファイル行番号
            item2 = QTableWidgetItem(right_num)
            # スコア
            item4 = QTableWidgetItem(score_text)

            # 左右の内容
            # 差分がある場合は、文字レベル差分を HTML 形式で表示
            if left_text != right_text:
                l_html, r_html = diff_characters(left_text, right_text)
                lbl_left = QLabel()
                lbl_left.setTextFormat(Qt.TextFormat.RichText)
                lbl_left.setText(l_html)
                # 背景色を差分色(diff_bg)に、文字は白（基本）を指定
                lbl_left.setStyleSheet("background-color: rgb(80,100,130); color: white;")
                lbl_right = QLabel()
                lbl_right.setTextFormat(Qt.TextFormat.RichText)
                lbl_right.setText(r_html)
                lbl_right.setStyleSheet("background-color: rgb(80,100,130); color: white;")
                self.table.setCellWidget(row, 1, lbl_left)
                self.table.setCellWidget(row, 3, lbl_right)
                item0.setBackground(diff_bg)
                item2.setBackground(diff_bg)
                item4.setBackground(diff_bg)
            else:
                self.table.setItem(row, 1, QTableWidgetItem(left_text))
                self.table.setItem(row, 3, QTableWidgetItem(right_text))
            self.table.setItem(row, 0, item0)
            self.table.setItem(row, 2, item2)
            self.table.setItem(row, 4, item4)

# =============================================================================
# メイン処理
# =============================================================================
if __name__ == '__main__':
    app = QApplication(sys.argv)
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

    # スプラッシュ画面の表示（モデル読み込み中）
    splash_pix = QPixmap(300, 100)
    splash_pix.fill(QColor(53, 53, 53))
    painter = QPainter(splash_pix)
    painter.setPen(QColor(255, 255, 255))
    painter.setFont(QFont("Arial", 16))
    painter.drawText(splash_pix.rect(), Qt.AlignmentFlag.AlignCenter, "MINI LM 読み込み中...")
    painter.end()
    splash = QSplashScreen(splash_pix)
    splash.show()
    app.processEvents()

    loader = ModelLoader()
    main_window = DiffWindow()
    loader.loaded.connect(lambda m: (splash.finish(main_window), main_window.show()))
    loader.start()

    sys.exit(app.exec())
