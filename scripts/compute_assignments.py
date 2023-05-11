'''
example usage:
python compute_assignment.py docs_shard_{$SHARD}_v2.jsonl
'''
import argparse
import json
import numpy as np
import linear_assignment
import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_jsonl')
    return parser.parse_args()


def get_image_assignments(im2txt):
    '''
    returns a list assignments of length N_images such that assignments[i] is the sentence index that image i was assigned to.
    '''
    im_idxs_s, txt_idxs_s, sol = linear_assignment.base_solve(-im2txt)
    im2txt_idxs = {im_idxs_s[k]: txt_idxs_s[k] for k in range(len(im_idxs_s))}
    if im2txt.shape[0] > im2txt.shape[1]:
        # there are more images than sentences. we dont want to discard images. so, for unassigned images, we will put them with their corresponding max.
        for imidx in range(len(im2txt)):
            if imidx not in im2txt_idxs:
                im2txt_idxs[imidx] = int(np.argmax(im2txt[imidx]))

    return [im2txt_idxs[idx] for idx in range(len(im2txt_idxs))]


def main():
    args = parse_args()

    docs = []
    with open(args.input_jsonl) as f:
        for line in f:
            docs.append(json.loads(line))

    for d in docs:
        im2txt = np.array(d['similarity_matrix'])
        assignment = get_image_assignments(im2txt)

        for im_idx, im in enumerate(d['image_info']):
            im['matched_text_index'] = int(assignment[im_idx])
            im['matched_sim'] = float(im2txt[im_idx, assignment[im_idx]])

    with open(args.input_jsonl, 'w') as f:
        f.write('\n'.join([json.dumps(d) for d in docs]))


if __name__ == '__main__':
    main()
