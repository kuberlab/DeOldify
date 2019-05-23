import argparse
import os
import sys
if os.environ['SRC_DIR'] not in sys.path:
    sys.path.append(os.environ['SRC_DIR'])
from fasterai.generators import gen_inference_deep, gen_inference_wide
import pathlib
import PIL.Image as Image
from fasterai.filters import MasterFilter, ColorizerFilter
import cv2
import numpy as np
from mlboardclient.api import client
import logging

logging.getLogger().setLevel('INFO')

render_factor = 35
os.environ['CUDA_VISIBLE_DEVICES']='0'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input',
        required=True,
    )
    parser.add_argument(
        '--yt',
        default=False,
        action='store_true'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    learn = gen_inference_deep(root_folder=pathlib.Path(os.environ['CODE_DIR']), weights_name='ColorizeArtistic_gen')
    filtr = MasterFilter([ColorizerFilter(learn=learn)], render_factor=render_factor)
    output_dir = os.path.join(os.environ['TRAINING_DIR'],os.environ['BUILD_ID'])
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_file = os.path.join(output_dir,'result.mp4')
    try:
        client.update_task_info({'input':args.input,'output':output_file,'elsapsed(sec)':0})
    except:
        logging.info('Unexpected error during submit state: {}'.format(sys.exc_info()[0]))
    if args.yt:
        import youtube_dl
        input_file = os.path.join(output_dir,'test.mp4')
        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',
            'outtmpl': input_file
        }
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download([args.input])
    else:
        input_file = args.input
    cap = cv2.VideoCapture(input_file)
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(output_file, fourcc,fps, (w,h))
    frames = 0


    while True:
        ret,frame = cap.read()
        if not ret:
            break
        frame = Image.fromarray(frame[:,:,::-1])
        frame = filtr.filter(frame, frame, render_factor=render_factor)
        out.write(np.array(frame)[:,:,::-1])
        frames+=1
        if frames%fps==0:
            logging.info('Processing frame: {}'.format(frames))
            try:
                client.update_task_info({'elsapsed(sec)':int(frames/fps)})
            except:
                logging.info('Unexpected error during submit state: {}'.format(sys.exc_info()[0]))

    out.release()

if __name__ == '__main__':
    main()
