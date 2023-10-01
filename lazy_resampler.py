"""
単純なサンプリングレート変換
"""
import argparse
from fractions import Fraction
import numpy as np
from scipy.signal import butter, sosfilt
from scipy.io import wavfile

def _anti_phasedistortion_filtering(sos, data):
    """
    位相歪みをキャンセルしたフィルタリング
    """
    data = sosfilt(sos.copy(), data)
    # 反転入力で再度フィルタリングすることで位相歪みがキャンセルされる
    return sosfilt(sos.copy(), data[::-1])[::-1]

def _resampling(data, interp, desim, filter_order):
    """
    interp / desim の比でリサンプリング
    """
    # 変換比を元にフィルタ設計
    sos = butter(filter_order, 1.0 / max(interp, desim), output='sos')
    # ゼロ埋め
    out = np.zeros(interp * len(data))
    out[::interp] = data
    # LPF適用
    out = _anti_phasedistortion_filtering(sos, out)
    # 間引き
    out = out[::desim]
    # ゲイン補償
    out *= interp
    return out

def _prime_factors(n):
    """
    素因数を列挙
    """
    i = 2
    factors = []
    while i * i <= n:
        if n % i == 0:
            n //= i
            factors.append(i)
        else:
            i += 1
    if n > 1:
        factors.append(n)
    return factors

def _factored_resampling(data, in_rate, out_rate, filter_order):
    """
    レート変換比の素因数分解によるリサンプリング
    """
    # 変換レートを有理数比に変換
    ratio = Fraction(out_rate, in_rate)
    # 素因数分解
    interp_factors = _prime_factors(ratio.numerator)
    desim_factors = _prime_factors(ratio.denominator)
    # 素因数によるリサンプリング
    while len(desim_factors) > 0 and len(interp_factors) > 0:
        interp = interp_factors.pop(0)
        desim = desim_factors.pop(0)
        # 間引き比が大きいと情報損失するため、補間比が大きくなるまで
        # 素因数を乗じる
        while interp < desim and len(interp_factors) > 0:
            interp *= interp_factors.pop(0)
        # リサンプリング
        data = _resampling(data, interp, desim, filter_order)
    # 残った素数を使ってリサンプリング
    if len(desim_factors) > 0 or len(interp_factors) > 0:
        data = _resampling(data,
                int(np.prod(interp_factors)), int(np.prod(desim_factors)), filter_order)
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str, help="specify input wav file")
    parser.add_argument("output_file", type=str, help="specify output wav file")
    parser.add_argument("output_sampling_rate", type=int, help="specify output sampling rate")
    parser.add_argument("--filter_order", type=int, default=30, help="specify butterworth filter order")

    args = parser.parse_args()
    OUTSR = args.output_sampling_rate

    # 入力ファイル取得
    INSR, inwav = wavfile.read(args.input_file)

    # 最小最大値の取得
    if 'float' in str(inwav.dtype):
        min_val, max_val = -1.0, 1.0
    else:
        info = np.iinfo(inwav.dtype)
        min_val, max_val = info.min, info.max

    # モノラルwavでも後の処理が共通化するように成形
    if len(inwav.shape) == 1:
        inwav = inwav.reshape((len(inwav), 1))

    # レート変換
    _, num_channels = inwav.shape
    outwav = np.array([], dtype=inwav.dtype)
    for ch in range(num_channels):
        y = _factored_resampling(inwav[:, ch], INSR, OUTSR, args.filter_order)
        y = np.clip(y, min_val, max_val).astype(inwav.dtype)
        outwav = np.vstack([outwav, y]) if outwav.size else y

    # 結果出力
    wavfile.write(args.output_file, OUTSR, outwav.T)
