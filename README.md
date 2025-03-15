<p align="center">
  <img src="mmc4_logo.png" width=512px>
</p>

<h1 align="center"> :camera: :memo: Multimodal C4 (mmc4) :memo: :camera: </h1>

<h3 align="center"> An open, billion-scale corpus of images interleaved with text. </h3>
<h4 align="center"> <a href="https://arxiv.org/abs/2304.06939">arXiv paper with curation details out now!</a></h4>

<br>

## Updates

- mmc4 is available once again! A huge thanks to [Weizhi Wang](https://victorwz.github.io/) and [Zekun Li](https://github.com/Leezekun/) for saving the corpus!
  - The original copies of mmc4 at ai2 were accidentially deleted in Feb 2025. [If you have any of the original copies of the dataset from before Feb. 2025, do let me know!](#missing-data)
- released mmc4 version 1.1 :fire: which fixes https://github.com/allenai/mmc4/issues/11 and https://github.com/allenai/mmc4/issues/10

## Corpus stats (v1.1)

|                                                     | # images | # docs | # tokens |
|-----------------------------------------------------|----------|--------|----------|
| Multimodal-C4 (mmc4)                                | 571M     | 101.2M | 43B      |
| Multimodal-C4 fewer-faces** (mmc4-ff)               | 375M     | 77.7M  | 33B      |
| Multimodal-C4 core (mmc4-core)                      | 29.9M    | 7.3M   | 2.4B     |
| Multimodal-C4 core fewer-faces** (mmc4-core-ff)     | 22.4M    | 5.5M   | 1.8B     |

** = available for direct download

More details about these datasets and our processing steps [can be found in our paper](https://arxiv.org/abs/2304.06939).

## Accessing mmc4-ff

### Documents

Now hosted on huggingface:

- mmc4 fewer faces (~ ): [jmhessel/mmc4-ff](https://huggingface.co/datasets/jmhessel/mmc4-ff)
- mmc4 core fewer faces (~ ): [jmhessel/mmc4-core](https://huggingface.co/datasets/jmhessel/mmc4-core)


The dataset is split into shards of jsonls.
- The shard number varies between 0 to 23098. [14 shards are missing and are not included in the dataset](#the-missing-shards-%EF%B8%8F).
- Each shard is a jsonl of documents. Each line is a document.

Documents contain text, image URLs, assignments of images to sentences, and image-by-text CLIP ViT-L/14 similarity matrices.

Specifically:

- `text_list`: a list of sentences comprising the text of the document
- `url`: the original url where the document was hosted
- `image_info` is a key mapping to a list of images. each image contains:
  - `image_name`: a filename that you could download the image to
  - `face_detections`: `None` if no faces are detected (which should be the case in "fewer faces")
  - `matched_text_index`: the index within `text_list` representing the sentence that this image is matched to
  - `matched_sim`: the CLIP ViT-L/14 similarity between the image and the sentence at the matched index
- `similarity_matrix`: a matrix of shape `len(image_info) x len(text_list)` where `similarity_matrix[i, j]` is the CLIP ViT-L/14 similarity between image `i` and sentence `j`.
- `could_have_url_duplicate`: a small number of URLs (~3%) in the corpus may have duplicate entries due to commoncrawl collecting multiple snapshots over time. we downsample such that, in expectation, each URL occurs once, but duplicates are technically possible. You can discard all entries with `could_have_url_duplicate` equal to 1 if you want a more strictly deduplicated set.

Here's an example:

```
{'image_info': [{'face_detections': None,
                 'image_name': 'b9040a0dbb22.jpg',
                 'matched_sim': 0.27694183588027954,
                 'matched_text_index': 2,
                 'raw_url': 'http://www.hfitinfo.com/honda_fit_pics/3/2/index.90.jpg'},
                {'face_detections': None,
                 'image_name': 'db1c21bc8474.jpg',
                 'matched_sim': 0.3234919607639313,
                 'matched_text_index': 1,
                 'raw_url': 'http://www.hfitinfo.com/honda_fit_pics/3/2/index.91.jpg'}],
 'similarity_matrix': [[0.24363446235656738,
                        0.31758785247802734,
                        0.27694183588027954],
                       [0.2233106791973114,
                        0.3234919607639313,
                        0.26118797063827515]],
 'text_list': ['When you lock the door using the lock tab on the driver’s '
               'door, all of the other doors and tailgate lock at the same '
               'time.',
               'Press the master door lock switch in as shown to lock or '
               'unlock all doors and the tailgate.',
               'When you lock/unlock the driver’s door and tailgate using the '
               'master lock switch, all the other doors lock/ unlock at the '
               'same time.'],
 'url': 'http://www.hfitinfo.com/hofi-48.html',
 'could_have_url_duplicate': 0 }
```
The assignments of images to sentences are computed using [compute_assignments.py](https://github.com/allenai/mmc4/blob/main/scripts/compute_assignments.py)

## Accessing raw images

Raw images can be downloaded from the provided URLs in the documents using [this script](scripts/download_images.py). The intent is to respect folks who have removed images from the web and not redistribute those images.

However, we understand that some of the URLs may be stale which can harm reproducibility efforts. If you're interested in updates regarding raw image availability, you can contact us using [this google form](https://forms.gle/fPSXY359MT1VvF1g8)

## The missing shards ⛏️💎🔍

.1% of the 23099 shards are missing from the corpus. These were not included in any statistics or experiments, so they are not part of mmc4. The missing shards are:

```
3218,3267,5064,5146,7119,8991,9750,11899,15127,15252,16996,17369,17499,17818
```

## License

- the new contributions of mmc4 beyond text-only c4 (e.g., the similarity matrices/image-text alignments) are released under [ODC-BY](https://opendatacommons.org/licenses/by/1-0/).
- By using mmc4, be aware of that you are also bound by the [Common Crawl terms of use](https://commoncrawl.org/terms-of-use/).

## Citation

If you found our work useful, please consider citing:
```
@article{zhu2023multimodal,
  title={{Multimodal C4}: An Open, Billion-scale Corpus of Images Interleaved With Text},
  author={Wanrong Zhu and Jack Hessel and Anas Awadalla and Samir Yitzhak Gadre and Jesse Dodge and Alex Fang and Youngjae Yu and Ludwig Schmidt and William Yang Wang and Yejin Choi},
  journal={arXiv preprint arXiv:2304.06939},
  year={2023}
}
```

## Missing data

In Feb 2025, the original copy of mmc4 hosted at AI2 was accidentially deleted. Thanks to some heroic efforts from [Weizhi Wang](https://victorwz.github.io/) and [Zekun Li](https://github.com/Leezekun/) who kindly provided their locally saved copies of mmc4 to be re-hosted, the corpus is (partially!) available again. The remaining missing files are:

- mmc4, originally hosted at `https://storage.googleapis.com/ai2-jackh-mmc4-gated-public-41423/data_v1.1/docs_shard_{$SHARD}_v2.jsonl.zip`.
- mmc4-core, originally hosted at `https://storage.googleapis.com/ai2-jackh-mmc4-gated-public-41423/data_core_v1.1/docs_shard_{$SHARD}_v3.jsonl`
- CLIP ViT/L-14 image features, originally hosted at `https://storage.googleapis.com/ai2-jackh-mmc4-public/images/clip_vitl14_shard_{$SHARD}_features.pkl`

If you have access to any of these files and are willing to make them available so we can once again host them for the broader community, [please let me know!](mailto:jmhessel@gmail.com)
