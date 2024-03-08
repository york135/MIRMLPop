import yt_dlp
import os, sys
import json
from tqdm import tqdm

if __name__ == "__main__":
    json_path = sys.argv[1]
    output_dir = sys.argv[2]

    with open(json_path) as json_data:
        data = json.load(json_data)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for i in tqdm(range(len(data))):
        wav_path = data[i]['song_id']
        cur_url = data[i]['youtube_link']
        outtmpl = os.path.join(output_dir, wav_path)

        if os.path.isfile(outtmpl + '.wav'):
            continue

        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': outtmpl,
            'postprocessors': [{  # Extract audio using ffmpeg
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
            }],
            "quiet": True,
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                error_code = ydl.download(cur_url)
        except:
            print ('Song #', data[i]['song_id'], 'not available.')