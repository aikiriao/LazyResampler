"""
単純なサンプリングレート変換
"""
import argparse
from fractions import Fraction
import numpy as np
from scipy.signal import butter, sosfilt
from scipy.io import wavfile

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str, help="specify input wav file")
    parser.add_argument("output_file", type=str, help="specify output wav file")
    parser.add_argument("output_sampling_rate", type=int, help="specify output sampling rate")
    parser.add_argument("--filter_order", default=5, help="specify butterworth filter order")

    args = parser.parse_args()
    OUTSR = args.output_sampling_rate

    # 入力ファイル取得
    INSR, inwav = wavfile.read(args.input_file)
    # レート変換比
    RATIO = Fraction(OUTSR, INSR)
    # 変換比を元にフィルタ設計
    SOS = butter(args.filter_order, 0.5 / max(RATIO.numerator, RATIO.denominator), output='sos')

    # モノラルwavでも後の処理が共通化するように成形
    if len(inwav.shape) == 1:
        inwav = inwav.reshape((len(inwav), 1))

    nsmp, nchan = inwav.shape
    out = np.zeros((round(nsmp * RATIO), nchan), dtype=inwav.dtype)
    for ch in range(nchan):
        # ゼロ埋め
        y = np.zeros(RATIO.numerator * nsmp, dtype=inwav.dtype)
        y[::RATIO.numerator] = inwav[:, ch]
        # フィルタ適用
        y = sosfilt(SOS.copy(), y)
        # 反転入力でフィルタリングすることで位相歪みがキャンセルされる
        y = sosfilt(SOS.copy(), y[::-1])[::-1]
        # 間引き
        y = y[::RATIO.denominator]
        y *= RATIO.numerator
        y = np.round(y).astype(inwav.dtype)
        out[:, ch] = y

    # 結果出力
    wavfile.write(args.output_file, OUTSR, out)
