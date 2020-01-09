import re
import os
import shutil
import fileinput
import subprocess

args = {
    'job_name':
    ['siam_tw', 'siam_co', 'siam_sl', 'siam_br', 'siam_spleen', 'siam_liver'],
    'exp_name':
    ['deeplab_nn']*6,
    'test_dirs': [
        '00 01 02 03 04 05', '10 11 12 13', '20 21 22 23 24 25',
        '30 31 32 33 34 35', '40 41 42 43 44 45', '50 51 52 53 54'
    ],
    'train_dir': ['00', '10', '20', '30', '40', '50'],
    'train_frames': ['15', '52', '15', '52', '102', '59']
}

template = 'mysubmit_tmpl.sh'
file_ = 'mysubmit_tmp.sh'

job_mask = [False, False, False, False, True, True]

for j in range(len(args['job_name'])):
    shutil.copyfile(template, file_)
    if (job_mask[j]):
        for k, v in args.items():
            for line in fileinput.input(file_, inplace=True):
                line = re.sub('###{}###'.format(k), v[j], line.rstrip())
                print(line)

        print('-----------------------------------')
        print('starting job {}'.format(j))
        print('-----------------------------------')
        os.system('cat {}'.format(file_))
        print('-----------------------------------')
        subprocess.call(["sbatch", file_])
