from django.shortcuts import render
from django.conf import settings
import os
import numpy as np
import cv2
import joblib
import librosa
import librosa.display
from scipy import signal
import community as community_louvain
import networkx as nx
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
import matplotlib.cm as cm

model = joblib.load('/home/pavan-kumar/pavan/audio-copy-forgery-detection/notebooks/audio_forgery_detection_model.joblib')
print("Model loaded successfully.")

def generate_super_resolution_spectrogram(audio_input, sr=16000, 
                                        short_n_fft=256, short_hop=64, 
                                        long_n_fft=2048, long_hop=512, 
                                        high_tf_n_fft=1024, high_tf_hop=256):
    if isinstance(audio_input, str):
        y, sr = librosa.load(audio_input, sr=sr)
    else:
        y = audio_input
    print(f"Sample Rate: {sr}, Duration: {len(y)/sr:.2f}s")

    S_short = librosa.stft(y, n_fft=short_n_fft, hop_length=short_hop)
    S_short_db = librosa.amplitude_to_db(np.abs(S_short), ref=np.max)
    print(f"Short Window STFT Shape: {S_short.shape}")

    S_long = librosa.stft(y, n_fft=long_n_fft, hop_length=long_hop)
    S_long_db = librosa.amplitude_to_db(np.abs(S_long), ref=np.max)
    print(f"Long Window STFT Shape: {S_long.shape}")

    S_high_tf = librosa.stft(y, n_fft=high_tf_n_fft, hop_length=high_tf_hop)
    S_high_tf_db = librosa.amplitude_to_db(np.abs(S_high_tf), ref=np.max)
    print(f"High Time-Frequency STFT Shape: {S_high_tf.shape}")

    freq_long = np.linspace(0, sr/2, S_long.shape[0])
    time_long = np.linspace(0, len(y)/sr, S_long.shape[1])
    freq_short = np.linspace(0, sr/2, S_short.shape[0])
    time_short = np.linspace(0, len(y)/sr, S_short.shape[1])

    interpolator = RegularGridInterpolator((freq_long, time_long), np.abs(S_long), 
                                         method='linear', bounds_error=False, fill_value=0)

    F_short, T_short = np.meshgrid(time_short, freq_short, indexing='ij')
    coords = np.stack([F_short.ravel(), T_short.ravel()], axis=-1)

    S_long_resized = interpolator(coords).reshape(S_short.shape)
    S_super_res = np.maximum(np.abs(S_short), S_long_resized)
    S_super_res_db = librosa.amplitude_to_db(S_super_res, ref=np.max)
    print(f"Super-Resolution Spectrogram Shape: {S_super_res_db.shape}")

    return S_high_tf_db, S_super_res_db

def plot_spectrogram(S, sr, title, hop_length, output_path):
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(S, sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Spectrogram saved to: {output_path}")

def load_audio(file_path, sr=22050, duration=10):
    try:
        audio, sample_rate = librosa.load(file_path, sr=sr, duration=duration)
        return audio, sample_rate
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return np.zeros(sr), sr
    
def extract_and_match_keypoints(spectrogram):
    max_dim = 1000
    if spectrogram.shape[0] > max_dim or spectrogram.shape[1] > max_dim:
        scale = max_dim / max(spectrogram.shape[0], spectrogram.shape[1])
        new_shape = (int(spectrogram.shape[1] * scale), int(spectrogram.shape[0] * scale))
        spectrogram_db = librosa.amplitude_to_db(np.abs(spectrogram), ref=np.max)
        img = np.uint8(255 * (spectrogram_db - spectrogram_db.min()) /
                      (spectrogram_db.max() - spectrogram_db.min() + 1e-10))
        img = cv2.resize(img, new_shape)
    else:
        spectrogram_db = librosa.amplitude_to_db(np.abs(spectrogram), ref=np.max)
        img = np.uint8(255 * (spectrogram_db - spectrogram_db.min()) /
                      (spectrogram_db.max() - spectrogram_db.min() + 1e-10))

    sift = cv2.SIFT_create(nfeatures=500)
    keypoints, descriptors = sift.detectAndCompute(img, None)

    if descriptors is None or len(keypoints) < 2:
        return [], [], img

    if len(keypoints) > 200:
        keypoints = keypoints[:200]
        descriptors = descriptors[:200]

    matches = []
    for i in range(len(keypoints)):
        for j in range(i+1, len(keypoints)):
            dist = np.linalg.norm(descriptors[i] - descriptors[j]) 
            if dist < 200:
                match = cv2.DMatch()
                match.queryIdx = i
                match.trainIdx = j
                match.distance = dist
                matches.append(match)
                if len(matches) >= 200:
                    break
        if len(matches) >= 200:
            break

    good_matches = matches[:200]
    return keypoints, good_matches, img

def determine_frequency_range(spectrogram, keypoints, matches, max_iter=10):
    if not matches or len(matches) < 2:
        freq_bins = np.linspace(0, 22050 // 2, spectrogram.shape[0])
        return freq_bins[0], freq_bins[-1]

    src_pts = np.float32([keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    if len(src_pts) > 0:
        hist, bin_edges = np.histogram(src_pts[:, 1], bins=10)
        max_bin = np.argmax(hist)

        total_freq = 22050 // 2
        bin_size = spectrogram.shape[0] / 10
        f_start = (bin_edges[max_bin] / spectrogram.shape[0]) * total_freq
        f_end = (bin_edges[max_bin+1] / spectrogram.shape[0]) * total_freq

        f_start = max(0, f_start - 500)
        f_end = min(total_freq, f_end + 500)
        return f_start, f_end

    freq_bins = np.linspace(0, 22050 // 2, spectrogram.shape[0])
    return freq_bins[0], freq_bins[-1]

def bandpass_filter_audio(audio, sr, f_start, f_end, order=2):
    nyquist = sr / 2
    f_start = max(20, min(f_start, nyquist - 100))
    f_end = max(f_start + 100, min(f_end, nyquist - 20))

    low = f_start / nyquist
    high = f_end / nyquist

    try:
        b, a = signal.butter(order, [low, high], btype='band')
        filtered_audio = signal.lfilter(b, a, audio)
    except Exception as e:
        print(f"Bandpass filter error: {e}")
        filtered_audio = audio
    return filtered_audio

def spiral_pattern_extraction(audio, frame_size=8, max_nodes=200):
    hop_length = max(frame_size * 8, len(audio) // max_nodes)
    if len(audio) < frame_size:
        padded_audio = np.pad(audio, (0, frame_size - len(audio)))
        frames = np.array([padded_audio])
    else:
        frames = librosa.util.frame(audio, frame_length=frame_size, hop_length=hop_length).T

    frames = frames[:max_nodes]
    spiral_values = [np.mean(frame) + np.std(frame) for frame in frames]
    return np.array(spiral_values)

def generate_visibility_graph(spiral_values, max_connections=10):
    G = nx.Graph()
    n = min(len(spiral_values), 200)
    
    if n < 2:
        G.add_node(0, value=0)
        return G, {0: 0}
    
    spiral_values = spiral_values[:n].astype(float)
    min_val = np.min(spiral_values)
    max_val = np.max(spiral_values)
    if max_val > min_val:
        normalized_values = (spiral_values - min_val) / (max_val - min_val)
    else:
        normalized_values = np.zeros_like(spiral_values)
    
    for i, (v, nv) in enumerate(zip(spiral_values, normalized_values)):
        G.add_node(i, value=float(v), normalized_value=float(nv))
    
    for a in range(n):
        for b in range(a + 1, min(a + 3, n)):
            weight = 1.0 - 0.1 * abs(b - a)
            G.add_edge(a, b, weight=weight, type='temporal')
        
        value_a = normalized_values[a]
        for b in range(n):
            if a != b and abs(a - b) > 2: 
                value_b = normalized_values[b]
                similarity = 1.0 - abs(value_a - value_b)
                if similarity > 0.9:  
                    G.add_edge(a, b, weight=similarity, type='similarity')
    
    for a in range(0, n, 10):
        b = (a + n//3) % n
        if not G.has_edge(a, b):
            G.add_edge(a, b, weight=0.5, type='long-range')
    
    try:
        partition = community_louvain.best_partition(G, resolution=1.2, randomize=False)
    except Exception as e:
        print(f"Community detection error: {e}")
        partition = {i: i % 5 for i in G.nodes()}
    
    return G, partition

def graph_to_image(G, partition, size=(480, 640), min_node_size=4, max_node_size=12):
    img = np.ones((*size, 3), dtype=np.uint8) * 255
    
    if len(G.nodes()) == 0:
        print("Graph has no nodes")
        return img

    try:
        pos = nx.kamada_kawai_layout(G)
    except Exception:
        try:
            pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)
        except Exception as e:
            print(f"Layout error: {e}")
            pos = {node: (np.random.random(), np.random.random()) for node in G.nodes()}

    degrees = dict(G.degree())
    max_degree = max(degrees.values()) if degrees else 1
    min_degree = min(degrees.values()) if degrees else 0
    range_degree = max_degree - min_degree or 1
    
    node_values = nx.get_node_attributes(G, 'value')
    
    for u, v in G.edges():
        if u in pos and v in pos:
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            x1 = int((x1 + 1) * size[1] / 2)
            y1 = int((y1 + 1) * size[0] / 2)
            x2 = int((x2 + 1) * size[1] / 2)
            y2 = int((y2 + 1) * size[0] / 2)
            if all(0 <= coord < dim for coord, dim in [(x1, size[1]), (y1, size[0]), (x2, size[1]), (y2, size[0])]):
                weight = G[u][v].get('weight', 1.0)
                thickness = max(1, int(weight * 2))
                color = (220, 220, 220)
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    
    colormap = cm.viridis
    
    for node in G.nodes():
        if node in pos:
            x, y = pos[node]
            x = int((x + 1) * size[1] / 2)
            y = int((y + 1) * size[0] / 2)

            if 0 <= x < size[1] and 0 <= y < size[0]:
                node_size = int(min_node_size + (degrees.get(node, 0) - min_degree) / range_degree * (max_node_size - min_node_size))
                
                community_id = partition[node]
                color_val = (community_id % 10) / 10.0
                rgba_color = colormap(color_val)
                color = (int(rgba_color[2]*255), int(rgba_color[1]*255), int(rgba_color[0]*255))
                cv2.circle(img, (x, y), node_size, color, -1)
                cv2.circle(img, (x, y), node_size, (50, 50, 50), 1)
    
    unique_communities = set(partition.values())
    legend_y = 20
    for comm in sorted(unique_communities)[:10]:  
        color_val = (comm % 10) / 10.0
        rgba_color = colormap(color_val)
        color = (int(rgba_color[2]*255), int(rgba_color[1]*255), int(rgba_color[0]*255))
        
        cv2.circle(img, (size[1] - 30, legend_y), 6, color, -1)
        cv2.circle(img, (size[1] - 30, legend_y), 6, (50, 50, 50), 1)
        cv2.putText(img, f"C{comm}", (size[1] - 20, legend_y + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        legend_y += 20
    
    cv2.putText(img, "Audio Pattern Graph", (10, 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    cv2.putText(img, f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}", 
               (10, size[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
    
    return img

def test(file_path, max_pso_iter=10, debug=True):
    try:
        audio, sr = load_audio(file_path, duration=10)
        _, V_super = generate_super_resolution_spectrogram(audio, sr)  
        keypoints, matches, img = extract_and_match_keypoints(V_super)
        if debug:
            print(f"Keypoints: {len(keypoints)}, Matches: {len(matches)}")
        if len(matches) > 5:
            f_start, f_end = determine_frequency_range(V_super, keypoints, matches, max_iter=max_pso_iter)
            filtered_audio = bandpass_filter_audio(audio, sr, f_start, f_end)
        else:
            filtered_audio = audio
        spiral_values = spiral_pattern_extraction(filtered_audio, max_nodes=200)
        G, partition = generate_visibility_graph(spiral_values, max_connections=10)
        if debug:
            print(f"Graph nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}")
        graph_img = graph_to_image(G, partition, size=(480, 640))
        return graph_img
    except Exception as e:
        if debug:
            print(f"Error processing {file_path}: {e}")
        return np.zeros((480, 640, 3), dtype=np.uint8)

def audio_to_mfcc(file_path):
    y, sr = librosa.load(file_path, sr=22050, duration=10)
    y = librosa.util.fix_length(y, size=int(22050 * 10))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return mfcc.flatten()

def index(request):
    if request.method == "POST" and 'audio_file' in request.FILES:
        audio_file = request.FILES['audio_file']
        
        upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
        os.makedirs(upload_dir, exist_ok=True)
        audio_path = os.path.join(upload_dir, audio_file.name)
        with open(audio_path, 'wb+') as destination:
            for chunk in audio_file.chunks():
                destination.write(chunk)

        output_dir = os.path.join(settings.STATICFILES_DIRS[0], 'images')
        os.makedirs(output_dir, exist_ok=True)
        
        spectrogram_path = os.path.join(output_dir, f"spectrogram.png")
        _, S_super = generate_super_resolution_spectrogram(audio_path) 
        plot_spectrogram(S_super, 16000, "Super-Resolution Spectrogram", 64, spectrogram_path)

        graph_img = test(audio_path, debug=True)
        graph_path = os.path.join(output_dir, f"graph.png")
        cv2.imwrite(graph_path, graph_img)
        print(f"Graph image saved to: {graph_path}, non-zero pixels: {np.count_nonzero(graph_img)}")

        mfcc_flat = audio_to_mfcc(audio_path)
        X_test = np.array([mfcc_flat])
        prediction = model.predict(X_test)[0]
        class_names = {0: "Genuine", 1: "Forged"}
        result = class_names[prediction]

        os.remove(audio_path)

        context = {
            'spectrogram_image': os.path.join('images', f"spectrogram.png"),
            'graph_image': os.path.join('images', f"graph.png"),
            'prediction': result
        }
        return render(request, "api/index.html", context)
    
    return render(request, "api/index.html")