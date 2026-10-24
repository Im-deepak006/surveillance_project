import urllib.request, os, sys
url = 'https://raw.githubusercontent.com/openai/CLIP/main/clip/bpe_simple_vocab_16e6.txt.gz'
out_path = os.path.join('run_on_video', 'clip', 'bpe_simple_vocab_16e6.txt.gz')
os.makedirs(os.path.dirname(out_path), exist_ok=True)
print('Downloading to', out_path)
urllib.request.urlretrieve(url, out_path)
print('Done')
