from flask import Flask, request, render_template, redirect, url_for
import os
import subprocess
import shutil
import glob
import librosa
import soundfile as sf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pyworld as pw

app = Flask(__name__)

# グローバル変数として再生履歴を管理
play_history = []

# 音源ファイルの保存ディレクトリ
UPLOAD_FOLDER = 'uploads'
AUDIO_FOLDER = 'static/audio'
PROCESSED_FOLDER = 'processed'

# 必要なディレクトリが存在しない場合は作成
for folder in [UPLOAD_FOLDER, AUDIO_FOLDER, PROCESSED_FOLDER]:
    os.makedirs(folder, exist_ok=True)

def clear_audio_folder():
    files = glob.glob(os.path.join(AUDIO_FOLDER, '*'))
    for f in files:
        os.remove(f)

def demucs_separate(file_path):
    output_dir = PROCESSED_FOLDER
    track_name = os.path.splitext(os.path.basename(file_path))[0]
    track_folder = os.path.join(output_dir, 'htdemucs', track_name)
    os.makedirs(track_folder, exist_ok=True)
    shutil.copy(file_path, os.path.join(track_folder, os.path.basename(file_path)))
    command = ['python', '-m', 'demucs.separate', '-o', output_dir, file_path]
    subprocess.run(command, check=True)
    return [os.path.join(track_folder, f) for f in os.listdir(track_folder) if f.endswith('.wav')]

def save_stems(selected_file):
    source_folder = os.path.join(PROCESSED_FOLDER, 'htdemucs', os.path.splitext(selected_file)[0])
    if not os.path.exists(source_folder):
        return
    for file_name in os.listdir(source_folder):
        shutil.copy(os.path.join(source_folder, file_name), os.path.join(AUDIO_FOLDER, file_name))

def load_audio(file_path):
    y, sr = librosa.load(file_path, sr=None)
    return y, sr

def save_audio(file_path, y, sr):
    sf.write(f'static/{file_path}', y, sr)


def frequency_to_note_index(frequency):
    """周波数をノート（音階のインデックス）に変換する関数"""
    if frequency <= 0:
        return None  # 雑音や無効な値は無視
    
    # A4の基準を440Hzとする
    A4_frequency = 440.0
    # A4のノート番号
    A4_note_number = 69

    # ノート番号の計算
    note_number = A4_note_number + 12 * np.log2(frequency / A4_frequency)
    return round(note_number)

def pitch_estimation(audio_file, f0_threshold=50):
    try:
        y, sr = load_audio(audio_file)
        
        # pyworldでの基本周波数（音高）推定
        _f0, t = pw.dio(y.astype(np.float64), sr)  # 初期のF0推定
        f0 = pw.stonemask(y.astype(np.float64), _f0, t, sr)  # 精密化したF0推定

        # 雑音を無効にするためのしきい値設定
        # 周波数が一定以下や振幅が低い部分を除外
        filtered_f0 = np.array([freq if freq > f0_threshold else 0 for freq in f0])

        # 有効な音高のみを対象にノートインデックスに変換
        note_indices = [frequency_to_note_index(p) for p in filtered_f0 if p > 0]
        corresponding_time = t[:len(note_indices)]

        # デバッグ用に表示
        print("音高インデックス:", note_indices)
        print("対応する時間ラベル:", corresponding_time)


        # グラフの横幅を動的に設定しつつ、縦幅をさらに小さく
        graph_width = max(8, int(len(corresponding_time) / 100))  # 横幅はそのまま
        plt.figure(figsize=(graph_width, 10))  # 縦幅を4に設定

        # Pitchのプロット
        plt.plot(corresponding_time, note_indices, label='Pitch', color='orange', linewidth=2)

        # グリッドの設定
        plt.grid(True, which='both', color='gray', linestyle='--', linewidth=0.7)

        # X軸の間隔調整とフォントサイズ
        plt.xticks(np.arange(min(corresponding_time), max(corresponding_time), step=1), rotation=45, fontsize=16)

        # Y軸の設定
        note_names = ['C3', 'C#3', 'D3', 'D#3', 'E3', 'F3', 'F#3', 'G3', 'G#3', 'A3', 'A#3', 'B3', 
                    'C4', 'C#4', 'D4', 'D#4', 'E4', 'F4', 'F#4', 'G4', 'G#4', 'A4', 'A#4', 'B4', 
                    'C5', 'C#5', 'D5', 'D#5', 'E5', 'F5', 'F#5', 'G5', 'G#5', 'A5', 'A#5', 'B5', 
                    'C6']
        plt.yticks(ticks=range(48, 85), labels=note_names, fontsize=12)
        plt.ylim(47, 85)  # 音域をC3（ノート番号48）からC6（ノート番号84）までに拡大


        # グラフ保存
        graph_path = os.path.join('static', 'pitch_estimation.png')
        plt.savefig(graph_path)
        plt.close()


        return graph_path
    except Exception as e:
        print(f"音高推定中にエラーが発生しました: {e}")
        return None

@app.route('/')
def index():
    clear_audio_folder()
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['audio_file']
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    stems = demucs_separate(file_path)
    for stem in stems:
        shutil.copy(stem, os.path.join(AUDIO_FOLDER, os.path.basename(stem)))
    play_history.append(file.filename)
    return render_template('select_track.html', stems=[os.path.basename(stem) for stem in stems])

@app.route('/select_track', methods=['GET', 'POST'])
def select_track():
    if request.method == 'POST':
        stem_name = request.form.get('stem_name')
        if stem_name:
            stem_path = os.path.join(AUDIO_FOLDER, stem_name)
            graph_path = pitch_estimation(stem_path)
            return render_template('playback.html', audio_file=stem_name, graph=graph_path)
    stems = [f for f in os.listdir(AUDIO_FOLDER) if f.endswith('.wav')]
    return render_template('select_track.html', stems=stems)

@app.route('/play', methods=['POST'])
def play_audio():
    stem_file = request.form['stem']
    full_path = os.path.join(AUDIO_FOLDER, stem_file)
    if not os.path.exists(full_path):
        return '音源ファイルが見つかりませんでした。'
    play_history.append(stem_file)
    return render_template('playback.html', audio_file=stem_file)

@app.route('/playlist', methods=['GET', 'POST'])
def playlist():
    clear_audio_folder()
    playlist_files = [f for f in os.listdir(UPLOAD_FOLDER) if f.endswith(('.mp3', '.wav'))]
    if request.method == 'POST':
        delete_file = request.form.get('delete_file')
        if delete_file:
            os.remove(os.path.join(UPLOAD_FOLDER, delete_file))
            shutil.rmtree(os.path.join(PROCESSED_FOLDER, 'htdemucs', os.path.splitext(delete_file)[0]))
            return redirect(url_for('playlist'))
        selected_file = request.form.get('selected_file')
        if selected_file:
            save_stems(selected_file)
            return redirect(url_for('select_from_playlist', filename=selected_file))
    return render_template('playlist.html', playlist=playlist_files)

@app.route('/select_from_playlist/<filename>')
def select_from_playlist(filename):
    stems = [f for f in os.listdir(AUDIO_FOLDER) if f.endswith('.wav')]
    return render_template('select_track.html', stems=stems)

@app.route('/play_from_select', methods=['POST'])
def play_from_select():
    selected_track = request.form.get('selected_track')
    return render_template('playback.html', track=selected_track)

@app.route('/explanation')
def explanation():
    return render_template('explanation.html')


if __name__ == '__main__':
    app.run(debug=True)
