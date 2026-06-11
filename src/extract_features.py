import librosa
import numpy as np
import glob
import sqlite3
from pathlib import Path


sample_rate = 44100
dv_time_window = 0.05
band_def = {
    'sub_bass': (20, 60),
    'bass': (60, 150),
    'low_mid': (150, 400),
    'mid': (400, 1000),
    'high_mid': (1000, 4000),
    'presence': (4000, 6000),
    'brilliance': (6000, 20000)
}

#setup DB connection
DB_PATH = Path(__file__).parent.parent / "data" / "songs.db"
con = sqlite3.connect(DB_PATH)
cur = con.cursor()


def get_bpm(signal, sr):
    onset_env = librosa.onset.onset_strength(y=signal, sr=sr)
    dtempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr,
                               aggregate=None)
    return np.round(np.mean(dtempo),2)

def get_dynamic_variance(signal_raw, sr, tw = dv_time_window):
    signal_normed = signal_raw / np.max(np.abs(signal_raw))
    sample_size = int(tw*sr)
    hop_size = sample_size // 2
    RMS = []
    for i in range(0, len(signal_normed) - sample_size, hop_size):
        window = signal_normed[i:i+sample_size]
        RMS.append(np.sqrt(np.mean(window**2)))
    RMS = np.array(RMS)
    dynamic_variance = np.std(RMS) / np.mean(RMS)
    return dynamic_variance

def get_band_profile(signal, sr, bands = band_def, n_fft=2048, hop_length=512):

    S = np.abs(librosa.stft(signal, n_fft=n_fft, hop_length=hop_length))**2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    band_power = {}
    for name, (f_lo, f_hi) in bands.items():
        idx = (freqs >= f_lo) & (freqs < f_hi)
        band_power[name] = S[idx].sum(axis=0)  # per frame

    total_power = np.sum(list(band_power.values()), axis=0)

    # remove silent / near-silent frames
    mask = total_power > 1e-12

    band_fraction = {}
    for k in band_power:
        band_fraction[k] = band_power[k][mask] / (total_power[mask] + 1e-10)

    # average over time
    band_profile = {k: np.mean(v) for k, v in band_fraction.items()}

    # final normalization for numerical safety
    total = sum(band_profile.values())
    band_profile = {k: np.round( v / (total + 1e-10), 6) for k, v in band_profile.items()}
    return band_profile

def get_spectral_metrics(band_profile):

    repr_log_band_frequencies = {}

    #define logarithmic representative frequencies for each band
    for band in band_profile:
        (f1, f2) = band_def[band]
        f = np.log(np.sqrt(f1*f2))
        repr_log_band_frequencies[f] = band_profile[band]

    spectral_center = 0
    for frequency in repr_log_band_frequencies:
        spectral_center += frequency*repr_log_band_frequencies[frequency]
    frequencies = np.array(list(repr_log_band_frequencies.keys()))

    normed_balance = (spectral_center - frequencies.min()) / (frequencies.max() - frequencies.min())
 
    spectral_spread = 0
    for frequency in repr_log_band_frequencies:
        spectral_spread += repr_log_band_frequencies[frequency]*(frequency-spectral_center)**2
    spectral_spread = np.sqrt(spectral_spread) / (frequencies.max() - frequencies.min())

    return (normed_balance, spectral_spread)

def get_timing_asymmetry(signal, sr, hop_length = 512):
    tempo, beat_frames = librosa.beat.beat_track(y=signal, sr=sr, hop_length=hop_length, units='frames')
    onset_env = librosa.onset.onset_strength(y=signal, sr=sr, hop_length=hop_length)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env,sr=sr, hop_length=hop_length)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)
    strengths = onset_env[onset_frames]

    alpha = 0.2
    beta = 0.3
    c = 8
    asymmetries = []

    for i, times in enumerate(beat_times[0:-1]):
        t1 = times
        t2 = beat_times[i+1]
        midpoint = (t1 + t2) / 2
        dt = t2 - t1

        # define window around midpoint of onbeats
        window_start = t1 + alpha*dt
        window_end = t2 - alpha*dt

        # find possible offbeat candidates in said window 
        mask = (onset_times > window_start) & (onset_times < window_end)
        candidates_times = onset_times[mask]
        candidates_strengths = strengths[mask]
        if len(candidates_times) == 0:
            continue
        
        # define best candidate as nearest/strongest onset
        normed_distances = np.abs(candidates_times - midpoint) / dt
        if len(candidates_times) > 1:
            scores = []
            for j, candidate in enumerate(candidates_strengths):
                if np.max(candidates_strengths) == np.min(candidates_strengths):
                    normed_strength = 1
                else:
                    normed_strength = (candidate - np.min(candidates_strengths)) / (np.max(candidates_strengths) - np.min(candidates_strengths))
                scores.append(0.85 * (1 - normed_distances[j]) + 0.15 * normed_strength)
            best_idx = np.argmax(scores)
            best_onset = candidates_times[best_idx]
        else:
            best_onset = candidates_times[0]

        # plausability check
        norm_dev = np.abs(best_onset - midpoint) / dt
        if norm_dev > beta:
            continue

        asymmetry_value = (best_onset - midpoint ) / dt
        asymmetries.append(asymmetry_value)

    if len(asymmetries) == 0:
        return (None, None, None)
    
    asymmetry_bias = np.median(asymmetries)
    timing_deviation = np.median(np.abs(asymmetries - asymmetry_bias))
    asymmetry_index = np.min([np.max([0, asymmetry_bias])/0.15, 1])*np.e**(-c*timing_deviation)

    return (asymmetry_bias, timing_deviation, asymmetry_index)

def UpdateDB(song):
    update_metric = ""
    metrics = dict(list(song.items())[1:])
    for metric in metrics:
        if song[metric] != None:
            update_metric += f" {metric} = {song[metric]} ,"
    update_metric = update_metric[0:-1]
    update_statement = f"UPDATE SONGS SET {update_metric} WHERE TRACK = '{song['title']}'"
    cur.execute(update_statement)

def InsertDB(song):
    insert_metric = "("
    for metric in song:
        if song[metric] != None:
            insert_metric += f"'{song[metric]}' ,"
        else: 
            insert_metric += "NULL," 
    insert_metric = insert_metric[0:-1] + ")"
    insert_statement = f"INSERT INTO Songs VALUES {insert_metric}"
    cur.execute(insert_statement)



directory = 'data/songs_mp3'

for track in glob.iglob(f'{directory}/*'):  

    title = Path(track).stem
    song = {
        'title': title,
        'bpm': None,
        'dynamic_variation_score': None,
        'sub_bass': None,
        'bass': None,
        'low_mid': None,
        'mid': None,
        'high_mid': None,
        'presence': None,
        'brilliance': None,
        'spectral_spread': None,
        'spectral_balance': None,
        'timing_deviation': None,
        'asymmetry_bias': None,
        'asymmetry_index': None,
        'rating': None
    }

    #get raw signal of audio file with samplerate sr
    signal, sample_rate = librosa.load(track, sr = sample_rate)

    #get BPM
    song['bpm'] = get_bpm(signal, sample_rate)

    # Dynamic Variance
    song['dynamic_variation_score'] = get_dynamic_variance(signal, sample_rate)
    
    # Band Profile
    band_profile = get_band_profile(signal, sample_rate)
    for band in band_profile:
        song[band] = band_profile[band]

    # Spectral Balance
    (song['spectral_balance'], song['spectral_spread']) = get_spectral_metrics(band_profile)

    #swing index
    (song['asymmetry_bias'], song['timing_deviation'], song['asymmetry_index']) = get_timing_asymmetry(signal, sample_rate)

    #Update DB
    try:
        InsertDB(song)
    except:
        UpdateDB(song)

con.commit()