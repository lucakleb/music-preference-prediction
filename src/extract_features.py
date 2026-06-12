from pathlib import Path
import sqlite3

import librosa
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "songs.db"
AUDIO_DIR = PROJECT_ROOT / "data" / "songs_mp3"

SAMPLE_RATE = 44_100
DV_TIME_WINDOW = 0.05

BAND_DEFINITIONS = {
    "sub_bass": (20, 60),
    "bass": (60, 150),
    "low_mid": (150, 400),
    "mid": (400, 1000),
    "high_mid": (1000, 4000),
    "presence": (4000, 6000),
    "brilliance": (6000, 20000),
}

COLUMNS = [
    "Track",
    "bpm",
    "dynamic_variation_score",
    "sub_bass",
    "bass",
    "low_mid",
    "mid",
    "high_mid",
    "presence",
    "brilliance",
    "spectral_spread",
    "spectral_balance",
    "timing_deviation",
    "asymmetry_bias",
    "asymmetry_index",
    "rating",
]


def get_bpm(signal: np.ndarray, sr: int) -> float:
    onset_env = librosa.onset.onset_strength(y=signal, sr=sr)
    tempo = librosa.feature.tempo(
        onset_envelope=onset_env,
        sr=sr,
        aggregate=None,
    )
    return round(float(np.mean(tempo)), 2)


def get_dynamic_variation(signal: np.ndarray, sr: int, time_window: float = DV_TIME_WINDOW) -> float:
    max_abs = np.max(np.abs(signal))

    if max_abs == 0:
        return 0.0

    signal_normed = signal / max_abs

    sample_size = int(time_window * sr)
    hop_size = sample_size // 2

    rms_values = []

    for i in range(0, len(signal_normed) - sample_size, hop_size):
        window = signal_normed[i:i + sample_size]
        rms_values.append(np.sqrt(np.mean(window**2)))

    rms_values = np.array(rms_values)

    if len(rms_values) == 0 or np.mean(rms_values) == 0:
        return 0.0

    return float(np.std(rms_values) / np.mean(rms_values))


def get_band_profile(
    signal: np.ndarray,
    sr: int,
    bands: dict[str, tuple[int, int]] = BAND_DEFINITIONS,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> dict[str, float]:
    spectrum = np.abs(librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)) ** 2
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    band_power = {}

    for name, (f_low, f_high) in bands.items():
        idx = (frequencies >= f_low) & (frequencies < f_high)
        band_power[name] = spectrum[idx].sum(axis=0)

    total_power = np.sum(list(band_power.values()), axis=0)
    mask = total_power > 1e-12

    band_profile = {}

    for name, power in band_power.items():
        fractions = power[mask] / (total_power[mask] + 1e-10)
        band_profile[name] = float(np.mean(fractions))

    total = sum(band_profile.values())

    if total == 0:
        return {name: 0.0 for name in bands}

    return {
        name: round(value / total, 6)
        for name, value in band_profile.items()
    }


def get_spectral_metrics(band_profile: dict[str, float]) -> tuple[float, float]:
    representative_frequencies = {}

    for band, value in band_profile.items():
        f_low, f_high = BAND_DEFINITIONS[band]
        log_frequency = np.log(np.sqrt(f_low * f_high))
        representative_frequencies[log_frequency] = value

    frequencies = np.array(list(representative_frequencies.keys()))
    weights = np.array(list(representative_frequencies.values()))

    spectral_center = np.sum(frequencies * weights)

    normed_balance = (
        (spectral_center - frequencies.min())
        / (frequencies.max() - frequencies.min())
    )

    spectral_spread = np.sqrt(
        np.sum(weights * (frequencies - spectral_center) ** 2)
    ) / (frequencies.max() - frequencies.min())

    return float(normed_balance), float(spectral_spread)


def get_timing_asymmetry(
    signal: np.ndarray,
    sr: int,
    hop_length: int = 512,
) -> tuple[float | None, float | None, float | None]:
    _, beat_frames = librosa.beat.beat_track(
        y=signal,
        sr=sr,
        hop_length=hop_length,
        units="frames",
    )

    onset_env = librosa.onset.onset_strength(
        y=signal,
        sr=sr,
        hop_length=hop_length,
    )

    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=hop_length,
    )

    onset_times = librosa.frames_to_time(
        onset_frames,
        sr=sr,
        hop_length=hop_length,
    )

    beat_times = librosa.frames_to_time(
        beat_frames,
        sr=sr,
        hop_length=hop_length,
    )

    strengths = onset_env[onset_frames]

    alpha = 0.2
    beta = 0.3
    decay = 8
    asymmetries = []

    for i, beat_time in enumerate(beat_times[:-1]):
        next_beat_time = beat_times[i + 1]
        midpoint = (beat_time + next_beat_time) / 2
        beat_duration = next_beat_time - beat_time

        window_start = beat_time + alpha * beat_duration
        window_end = next_beat_time - alpha * beat_duration

        mask = (onset_times > window_start) & (onset_times < window_end)

        candidate_times = onset_times[mask]
        candidate_strengths = strengths[mask]

        if len(candidate_times) == 0:
            continue

        normed_distances = np.abs(candidate_times - midpoint) / beat_duration

        if len(candidate_times) > 1:
            min_strength = np.min(candidate_strengths)
            max_strength = np.max(candidate_strengths)

            if max_strength == min_strength:
                normed_strengths = np.ones_like(candidate_strengths)
            else:
                normed_strengths = (
                    (candidate_strengths - min_strength)
                    / (max_strength - min_strength)
                )

            scores = 0.85 * (1 - normed_distances) + 0.15 * normed_strengths
            best_onset = candidate_times[np.argmax(scores)]
        else:
            best_onset = candidate_times[0]

        normed_deviation = abs(best_onset - midpoint) / beat_duration

        if normed_deviation > beta:
            continue

        asymmetry = (best_onset - midpoint) / beat_duration
        asymmetries.append(asymmetry)

    if not asymmetries:
        return None, None, None

    asymmetries = np.array(asymmetries)

    asymmetry_bias = float(np.median(asymmetries))
    timing_deviation = float(np.median(np.abs(asymmetries - asymmetry_bias)))

    asymmetry_index = (
        min(max(0, asymmetry_bias) / 0.15, 1)
        * np.exp(-decay * timing_deviation)
    )

    return asymmetry_bias, timing_deviation, float(asymmetry_index)


def extract_features(file_path: Path) -> dict:
    signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    band_profile = get_band_profile(signal, sr)
    spectral_balance, spectral_spread = get_spectral_metrics(band_profile)
    asymmetry_bias, timing_deviation, asymmetry_index = get_timing_asymmetry(signal, sr)

    song = {
        "Track": file_path.stem,
        "bpm": get_bpm(signal, sr),
        "dynamic_variation_score": get_dynamic_variation(signal, sr),
        **band_profile,
        "spectral_spread": spectral_spread,
        "spectral_balance": spectral_balance,
        "timing_deviation": timing_deviation,
        "asymmetry_bias": asymmetry_bias,
        "asymmetry_index": asymmetry_index,
        "rating": None,
    }

    return song


def upsert_song(conn: sqlite3.Connection, song: dict) -> None:
    placeholders = ", ".join(["?"] * len(COLUMNS))
    columns = ", ".join(COLUMNS)

    update_assignments = ", ".join(
        f"{column} = excluded.{column}"
        for column in COLUMNS
        if column not in ["Track", "rating"]
    )

    values = [song[column] for column in COLUMNS]

    query = f"""
        INSERT INTO songs ({columns})
        VALUES ({placeholders})
        ON CONFLICT(Track) DO UPDATE SET
            {update_assignments}
    """

    conn.execute(query, values)


def main() -> None:
    audio_files = list(AUDIO_DIR.glob("*.mp3"))

    if not audio_files:
        print(f"No MP3 files found in {AUDIO_DIR}")
        return

    with sqlite3.connect(DB_PATH) as conn:
        for file_path in audio_files:
            print(f"Processing: {file_path.name}")
            song = extract_features(file_path)
            upsert_song(conn, song)

        conn.commit()

    print(f"Processed {len(audio_files)} files.")


if __name__ == "__main__":
    main()