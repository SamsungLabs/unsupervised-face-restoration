import numpy as np
import torch
import os
import cv2
import lpips
import pyiqa
import pickle
import argparse
from pathlib import Path

from basicsr.utils import img2tensor
from basicsr.metrics.nr_metrics import calculate_maniqa, calculate_musiq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--pred_dir', type=str, required=True, help='Path to the testing dir')
    parser.add_argument(
            "--out_path",
            type=str,
            default='out.txt',
            help='text file summarizing results',
            )



    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    musiq_fn = pyiqa.create_metric('musiq', device='cuda')
    maniqa_fn = pyiqa.create_metric('maniqa', device='cuda')

    maniqa_list, musiq_list = [], []
    filename_list = []

    pred_dir = args.pred_dir
    
    pred_file_list = sorted(os.listdir(pred_dir))

    
    for idx in range(len(pred_file_list)):
        pred_img_path = os.path.join(pred_dir, pred_file_list[idx])
        # input_noisy_img_path = os.path.join(files_dir, output_list[idx])
        
        input_img = cv2.imread(pred_img_path)
        input_img = input_img.astype(np.float32) / 255.
        pred = img2tensor(input_img, bgr2rgb=True, float32=True)
    
        
        with torch.no_grad():
            musiq_result = calculate_musiq(pred.unsqueeze(0), musiq_fn=musiq_fn)
            maniqa_result = calculate_maniqa(pred.unsqueeze(0), maniqa_fn=maniqa_fn)

        
        maniqa_list.append(maniqa_result)
        musiq_list.append(musiq_result)
        filename_list.append(Path(pred_img_path).stem)
        print(pred_img_path, maniqa_result, musiq_result, Path(pred_img_path).stem)

    maniqa_mean = np.mean(np.array(maniqa_list))
    musiq_mean = np.mean(np.array(musiq_list))
    
    print('avg maniqa, musiq', maniqa_mean, musiq_mean)
    
    output_text_file = Path(args.out_path) / 'output_real.txt'
    with open(output_text_file, 'a') as f:
        f.write('Evaluation\n')
        f.write(f'Average MANIQA {maniqa_mean}\n')
        f.write(f'Average MUSIQ {musiq_mean}\n')
        f.write('filename | MANIQA | MUSIQ \n')
        for file_idx in range(len(filename_list)):
            f.write(f'{filename_list[file_idx]} | {maniqa_list[file_idx]} | {musiq_list[file_idx]}\n')
    
    output_pickle_file = Path(args.out_path) / 'output_real.pkl'
    with open(output_pickle_file, 'wb') as pkl_file:
        pickle.dump({'filenames': filename_list, 
                    'maniqa': np.array(maniqa_list),
                    'musiq': np.array(musiq_list)}, pkl_file)
            
        

        

if __name__ == '__main__':
    main()
