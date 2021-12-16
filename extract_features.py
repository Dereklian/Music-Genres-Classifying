import numpy as np
import contextlib
import wave
from librosa import resample


def load_file(path):
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        sample_rate = wf.getframerate()
        pcm_data = wf.readframes(wf.getnframes())
        wave_data = np.fromstring(pcm_data, dtype=np.int16)
        wave_data = wave_data[:int(30 * sample_rate)]
        wave_data = wave_data / max(abs(wave_data))
        wave_data = resample(wave_data, sample_rate, 22050)
    return wave_data, 22050


def spectrum(frames, frame_size):
    f_energy = np.square(abs(np.fft.rfft(frames, frame_size))) + 1
    return f_energy


def dct_transform(cep_unm):
    dctcoef = np.zeros((cep_unm, 2 * cep_unm))
    for k in np.arange(cep_unm):
        n = np.arange(2 * cep_unm)
        dctcoef[k, :] = np.cos((2 * n + 1) * (k + 1) * np.pi / (4 * cep_unm))
    return dctcoef


def calc_mel_filter_banks(cep_num, frame_size, sample_rate):
    fl = 0
    fh = sample_rate / 2
    bl = 1125 * np.log(1 + fl / 700)
    bh = 1125 * np.log(1 + fh / 700)
    bandwidth = bh - bl
    mel_points = np.linspace(0, bandwidth, cep_num + 2)
    binf = 700 * (np.exp(mel_points / 1125) - 1)
    w2 = int(frame_size / 2 + 1)
    bank = np.zeros([cep_num, w2])
    df = sample_rate / frame_size
    freq = []
    for count in range(0, w2):
        freqs = int(count * df)
        freq.append(freqs)
    for i in range(1, cep_num + 1):
        mid = binf[i]
        left = binf[i - 1]
        right = binf[i + 1]
        mid = np.floor(mid / df)
        left = np.floor(left / df)
        right = np.floor(right / df)
        for j in range(1, w2):
            if left <= j <= mid:
                bank[i - 1, j] = (j - left) / (mid - left)
            elif mid < j <= right:
                bank[i - 1, j] = (right - j) / (right - mid)
    return bank


def lift_window(win_len):
    lift = 1 + (win_len / 2) * np.sin(np.pi * np.arange(1, win_len + 1) / win_len)
    lift = lift / max(lift)
    lift = np.reshape(np.array(lift), (1, win_len))
    return lift


def cal_mfcc(frames, frame_size, sample_rate, cep_num):
    mel_bank = calc_mel_filter_banks(cep_num * 2, frame_size, sample_rate)
    f_energy = spectrum(frames, frame_size)  # nf*129
    melX = np.log(np.dot(f_energy, mel_bank.T))  # nf*26
    dctcoef = dct_transform(cep_num)  # 13*26
    feat = np.dot(dctcoef, melX.T)  # 13*nf
    lift = lift_window(cep_num)
    mfcc = np.multiply(feat.T, np.tile(lift, (feat.shape[1], 1)))
    return mfcc  # nf*13


def derivate(feat):
    nf, ndim = feat.shape
    result = np.zeros([nf, ndim])
    for frame in range(2, nf - 2):
        result[frame, :] = -2 * feat[frame - 2, :] - feat[frame - 1, :] + feat[frame + 1, :] + 2 * feat[frame + 2, :]
    result = result / 3.0
    return result


def delta_delta_mfcc(frames, frame_size, sample_rate):
    nf = frames.shape[0]
    mfcc = cal_mfcc(frames, frame_size, sample_rate, 13)  # nf*13
    mfcc_delta = derivate(mfcc)
    mfcc_delta_delta = derivate(mfcc_delta)
    feat = np.concatenate((mfcc, mfcc_delta, mfcc_delta_delta), axis=1)
    feat = feat[2:nf - 2, :]
    return feat


def pre_emphasis(audio, e):
    audio[1:] = [x - e * y for (x, y) in zip(audio[1:], audio[:-1])]
    return audio


def enframe(audio, frame_size, frame_inc, preE):
    audio = pre_emphasis(audio, preE)
    if len(audio) < frame_size:
        nf = 1
    else:
        nf = int((len(audio) - frame_size) / frame_inc) + 1

    pad_length = (nf - 1) * frame_inc + frame_size
    pad_signal = audio[0:pad_length]
    indices = np.tile(np.arange(0, frame_size), (nf, 1)) \
              + np.tile(np.arange(0, nf * frame_inc, frame_inc), (frame_size, 1)).T
    indices = np.array(indices, dtype=np.int16)
    frames = np.array(pad_signal)[indices]
    ham = np.hamming(frame_size)
    win = np.tile(ham, (nf, 1))
    frames = np.multiply(frames, win)
    return frames

