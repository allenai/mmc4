"""
Adapted from: 
    https://github.com/igorbrigadir/DownloadConceptualCaptions/blob/master/download_data.py

Requirements:
    - ImageMagick
    - See requirements.txt for python dependencies

Example Usage:
    python download_images.py --input_jsonl ./data_core/docs_no_face_shard_0_v3.jsonl
"""

import pandas as pd
import requests
import os
import shelve
import magic
from multiprocessing import Pool
import tqdm
import argparse
import json
import subprocess
import time
import glob
from pathlib import Path


headers = {
    'User-Agent':'Googlebot-Image/1.0', # Pretend to be googlebot
    'X-Forwarded-For': '64.18.15.200'
}


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_jsonl', type=str, required=True, help='Local path to the input jsonl file')
    parser.add_argument('--output_image_dir', type=str, default=None, help='Local path to the directory that stores the downloaded images')
    parser.add_argument('--num_process', type=int, default=16, help='Number of processes in the pool can be larger than cores')
    parser.add_argument('--chunk_size', type=int, default=100, help='Number of images per chunk per process')
    parser.add_argument('--shard_name', type=str, default=None)
    parser.add_argument('--report_dir', type=str, default='./status_report/', help='Local path to the directory that stores the downloading status')
    
    args = parser.parse_args()

    assert args.input_jsonl.endswith('.jsonl')

    if args.shard_name is None:
        args.shard_name = Path(args.input_jsonl).stem
    
    if args.output_image_dir is None:
        args.output_image_dir = f'./{args.shard_name}_images/'

    return args


def call(cmd):
    subprocess.call(cmd, shell=True)


def _df_split_apply(tup_arg):
    split_ind, subset, func = tup_arg
    r = subset.apply(func, axis=1)
    return (split_ind, r)


def download_images_multiprocess(args, df, func):
    """Download images with multiprocessing"""

    chunk_size = args.chunk_size
    num_process = args.num_process

    print('Generating parts...')

    shelve_filename = '%s_%s_%s_results.tmp' % (args.shard_name, func.__name__, chunk_size)
    with shelve.open(shelve_filename) as results:

        pbar = tqdm.tqdm(total=len(df), position=0)
        # Resume:
        finished_chunks = set([int(k) for k in results.keys()])
        pbar.desc = "Resuming"
        for k in results.keys():
            pbar.update(len(results[str(k)][1]))

        pool_data = ((index, df[i:i + chunk_size], func) for index, i in enumerate(range(0, len(df), chunk_size)) if index not in finished_chunks)
        pbar.write(f'\t{int(len(df) / chunk_size)} parts. Using {num_process} processes.')

        pbar.desc = "Downloading"
        with Pool(num_process) as pool:
            for i, result in enumerate(pool.imap_unordered(_df_split_apply, pool_data, 2)):
                results[str(result[0])] = result
                pbar.update(len(result[1]))
        pbar.close()

    print(f'Finished downloading images for {args.input_jsonl}\nImages saved at {args.output_image_dir}')

    return shelve_filename


def _get_local_image_filename(row):
    return row['folder'] + '/' + row['local_identifier']


def download_image(row):
    fname = _get_local_image_filename(row)

    # Skip already downloaded images, retry others later
    if os.path.isfile(fname):
        row['status'] = 200
        row['file'] = fname
        row['mimetype'] = magic.from_file(row['file'], mime=True)
        row['size'] = os.stat(row['file']).st_size
        return row

    try:
        # Use smaller timeout to skip errors, but can result in failed downloads
        response = requests.get(row['url'], stream=False, timeout=10, allow_redirects=True, headers=headers)
        row['status'] = response.status_code
        rate_limit_idx = 0
        while response.status_code == 429:
            print(f'RATE LIMIT {rate_limit_idx} for {row["local_identifier"]}, will try again in 2s')
            response = requests.get(row['url'], stream=False, timeout=10, allow_redirects=True, headers=headers)
            row['status'] = response.status_code
            rate_limit_idx += 1
            time.sleep(2)
            if rate_limit_idx == 5:
                print(f'Reached rate limit for {row["local_identifier"]} ({row["url"]}). Will skip this image for now.')
                row['status'] = 429
                return row

    except Exception as e:
        # log errors later, set error as 408 timeout
        row['status'] = 408
        return row

    if response.ok:
        try:
            with open(fname, 'wb') as out_file:
                # some sites respond with gzip transport encoding
                response.raw.decode_content = True
                out_file.write(response.content)

            # Resize image if it is too big
            call('mogrify -resize "800x800>" {}'.format(fname))

            row['mimetype'] = magic.from_file(fname, mime=True)
            row['size'] = os.stat(fname).st_size
        except:
            # This is if it times out during a download or decode
            row['status'] = 408
            return row
        row['file'] = fname
    return row


def save_status(args, shelve_filename):
    print(f'Generating Dataframe from results...')
    with shelve.open(shelve_filename) as results:
        keylist = sorted([int(k) for k in results.keys()])
        df = pd.concat([results[str(k)][1] for k in keylist], sort=True)
    
    report_filename = os.path.join(args.report_dir, f'{args.shard_name}.tsv.gz')
    df.to_csv(report_filename, sep='\t', compression='gzip', header=False, index=False)
    print(f'Status report saved to {report_filename}')
    
    print('Cleaning up...')
    matched_files = glob.glob(f'{shelve_filename}*')
    for fn in matched_files:
        os.remove(fn)


def gather_image_info(args):
    """Gather image info from the input jsonl"""
    data = []
    with open(args.input_jsonl) as f:
        for line in tqdm.tqdm(f):
            info = json.loads(line.strip())
            for img_item in info['image_info']:
                data.append({
                    'local_identifier': img_item['image_name'],
                    'url': img_item['raw_url'],
                })
    return data


def main():
    args = parse_args()

    # Prepare directory
    for _dir in [args.output_image_dir, args.report_dir]:
        if not os.path.exists(_dir):
            os.makedirs(_dir)

    # Load image info for current shard
    data = gather_image_info(args)
    for d in data:
        d['folder'] = args.output_image_dir
    df = pd.DataFrame(data)

    # Download images
    shelve_filename = download_images_multiprocess(
        args=args, 
        df=df,
        func=download_image,
    )
    
    # Save status & cleaning up
    save_status(
        args=args,
        shelve_filename=shelve_filename,
    )


if __name__ == '__main__':
    main()
