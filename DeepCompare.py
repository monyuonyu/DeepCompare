import sys
import os
import torch
from sentence_transformers import SentenceTransformer, util
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QFileDialog, QTableWidget, QTableWidgetItem,
    QMessageBox, QHeaderView
)
from PyQt6.QtGui import QColor  # 背景色設定用

# --- MiniLM モデルの読み込み ---
MODEL_NAME = "sentence-transformers/paraphrase-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)

# ============================================================
# 以下、CUI 版と同様の関数群
# ============================================================

def get_line_embeddings(code_text: str):
    """
    コードテキストを行ごとに分割し、各行の埋め込みを MiniLM で計算する関数
    """
    lines = code_text.splitlines()
    embeddings = model.encode(lines, convert_to_tensor=True)
    return lines, embeddings

def compute_similarity_matrix(embeddings1, embeddings2):
    """
    2 つの埋め込み集合間のコサイン類似度マトリックスを計算する関数
    """
    similarity_matrix = util.pytorch_cos_sim(embeddings1, embeddings2)
    return similarity_matrix.tolist()

def align_lines(lines1, lines2, sim_matrix, gap_penalty=-0.5):
    """
    動的計画法（Needleman–Wunsch アルゴリズム）を用いて、
    2 つの行リストの最適なアライメントを求める関数

    【定式化】
      dp[i][j] = max{
        dp[i-1][j-1] + sim_matrix[i-1][j-1],   （対角移動：両ファイルの行を対応付ける）
        dp[i-1][j] + gap_penalty,              （上方向：ファイル1側の行をギャップ）
        dp[i][j-1] + gap_penalty               （左方向：ファイル2側の行をギャップ）
      }
    """
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

# ============================================================
# PyQt6 を用いた GUI クラスの定義
# ============================================================

class DiffWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MiniLM ベース コード比較ツール")
        self.resize(1200, 600)
        
        # メインウィジェットとレイアウトの設定
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        
        # --- ファイル選択用ウィジェット ---
        file_layout = QHBoxLayout()
        self.file1_edit = QLineEdit()
        self.file1_edit.setPlaceholderText("ファイル1のパスを入力")
        file1_button = QPushButton("参照")
        file1_button.clicked.connect(self.select_file1)
        self.file2_edit = QLineEdit()
        self.file2_edit.setPlaceholderText("ファイル2のパスを入力")
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
        
        # --- 結果表示用のテーブル ---
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["File1", "File2", "Score"])
        # グリッド（セパレーター）を非表示にする
        self.table.setShowGrid(False)
        # 列ごとのサイズ設定
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        # 3列目（スコア）のサイズは固定
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(2, 60)
        main_layout.addWidget(self.table)
    
    def select_file1(self):
        """
        ファイル1を選択するためのダイアログを表示する
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self, "ファイル1を選択", "", "Python Files (*.py);;All Files (*)"
        )
        if file_path:
            self.file1_edit.setText(file_path)
    
    def select_file2(self):
        """
        ファイル2を選択するためのダイアログを表示する
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self, "ファイル2を選択", "", "Python Files (*.py);;All Files (*)"
        )
        if file_path:
            self.file2_edit.setText(file_path)
    
    def compare_files(self):
        """
        入力された 2 つのファイルを読み込み、行単位で比較を実行し、
        結果をテーブルに表示する
        """
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
        # --- アライメント計算（ギャップペナルティ：-0.5） ---
        aligned = align_lines(lines1, lines2, sim_matrix, gap_penalty=-0.5)
        
        # --- 結果をテーブルに表示 ---
        self.table.setRowCount(len(aligned))
        for row, (left_idx, right_idx, score) in enumerate(aligned):
            if left_idx is not None:
                left_text = lines1[left_idx]
            else:
                left_text = "---"  # ギャップの場合
            if right_idx is not None:
                right_text = lines2[right_idx]
            else:
                right_text = "---"
            score_text = f"{score:.2f}" if score is not None else ""
            
            # QTableWidgetItem の作成
            left_item = QTableWidgetItem(left_text)
            right_item = QTableWidgetItem(right_text)
            score_item = QTableWidgetItem(score_text)
            
            # 左右のテキストが異なる場合、背景色でハイライト
            if left_text != right_text:
                highlight_color = QColor(255, 255, 150)  # 薄い黄色
                left_item.setBackground(highlight_color)
                right_item.setBackground(highlight_color)
            
            self.table.setItem(row, 0, left_item)
            self.table.setItem(row, 1, right_item)
            self.table.setItem(row, 2, score_item)
        
        QMessageBox.information(self, "完了", "比較が完了しました。")

# ============================================================
# アプリケーションの起動
# ============================================================
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = DiffWindow()
    window.show()
    sys.exit(app.exec())
