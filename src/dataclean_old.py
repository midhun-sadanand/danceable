import kagglehub

# Download latest version
path = kagglehub.dataset_download("tomigelo/spotify-audio-features")

print("Path to dataset files:", path)