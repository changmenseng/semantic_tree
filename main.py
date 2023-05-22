import os
import json
import argparse
import tempfile

CUDA_VISIBLE_DEVICES = os.getenv('CUDA_VISIBLE_DEVICES')
if CUDA_VISIBLE_DEVICES is not None:
    os.system(f'export CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES}')


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', type=str, required=True, choices=['sst2/phrase', 'sst2/sent', 'MR'])
parser.add_argument('-m', '--model', type=str, required=True, choices=['lstm', 'bert'])
args = parser.parse_args()

config_file = os.path.join('configs', args.dataset, f'{args.model}.json')
if args.dataset.startswith('sst2'):
    os.system(f'python train.py -c {config_file}')
    
else: # for MR, conduct 10-fold cross-validation
    with open(config_file, 'r', encoding='utf8') as f:
        config = json.load(f)
    
    cmd = []; config_fnames = []
    for i in range(10):
        config['data']['train']['sent_fnames'] = []
        config['data']['train']['label_fnames'] = []
        for j in range(10):
            if j != i:
                config['data']['train']['sent_fnames'].append(f'resources/dataset/MR/{j}.sents.parse.txt')
                config['data']['train']['label_fnames'].append(f'resources/dataset/MR/{j}.labels.txt')
        config['data']['val']['sent_fnames'] = [f'resources/dataset/MR/{i}.sents.parse.txt']
        config['data']['val']['label_fnames'] = [f'resources/dataset/MR/{i}.labels.txt']
        config['data']['test'] = [{
            "sent_fnames": [f'resources/dataset/MR/{i}.sents.parse.txt'],
            "label_fnames": [f'resources/dataset/MR/{i}.labels.txt'],
            "batch_size": 60
        }]
        config['version'] = i
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', encoding='utf8', delete=False) as f:
            json.dump(config, f, indent=2)

        cmd.append(f'python train.py -c {f.name}')
        config_fnames.append(f.name)

    cmd = ';'.join(cmd)

    try:
        os.system(cmd)
    except:
        pass
    finally:
        for config_fname in config_fnames:
            os.remove(config_fname)

