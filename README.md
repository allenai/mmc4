<p align="center">
  <img src="mmc4_logo.png" width=512px>
</p>

<h1 align="center"> :camera: :memo: Multimodal C4 (mmc4) :memo: :camera: </h1>

<h3 align="center"> An open, billion-scale corpus of images interleaved with text. </h3>
<h4 align="center"> <a href="https://arxiv.org/abs/2304.06939">arXiv paper with curation details out now!</a></h4>

<br>

## Corpus stats

|                                                     | # images | # docs | # tokens |
|-----------------------------------------------------|----------|--------|----------|
| Multimodal-C4 (mmc4)                                | 585M     | 103M   | 43B      |
| Multimodal-C4 fewer-faces (mmc4-ff)                 | 385M     | 79M    | 34B      |
| Multimodal-C4 core (mmc4-core)                      | 30.5M    | 7.4M   | 2.5B     |
| Multimodal-C4 core fewer-faces (mmc4-core-ff)       | 22.9M    | 5.6M   | 1.8B     |

More details about these datasets and our processing steps [can be found in our paper](https://arxiv.org/abs/2304.06939)

## Accessing mmc4-ff

### Documents

You can directly download the "fewer faces" multimodal c4 documents at urls like this:

`https://storage.googleapis.com/ai2-jackh-mmc4-public/data/docs_no_face_shard_{$SHARD}_v2.jsonl.zip`

where `SHARD` can vary from 0 to 23098. [14 shards are missing and are not included in the dataset](#the-missing-shards-%EF%B8%8F). 

You can download the smaller "core fewer faces" documents at URLs like this: `https://storage.googleapis.com/ai2-jackh-mmc4-public/data_core/docs_no_face_shard_{$SHARD}_v3.jsonl.zip` 

where `SHARD` can vary from 0 to 23098. The total size of all these files together is approximately 9.4GB.

You can also automatically download & unzip these files from commands, you can run the script by providing the destination folder as an argument, like:

`sh download_scripts/fewer_facesv2.sh /path/to/destination/folder`

`sh download_scripts/fewer_faces_corev3.sh /path/to/destination/folder`

Documents in both sets contain text, image URLs, assignments of images to sentences, and image-by-text CLIP ViT-L/14 similarity matrices. Specifically:

- `text_list`: a list of sentences comprising the text of the document
- `url`: the original url where the document was hosted
- `image_info` is a key mapping to a list of images. each image contains:
  - `image_name`: a filename that you could download the image to
  - `face_detections`: `None` if no faces are detected (which should be the case in "fewer faces")
- `similarity_matrix`: a matrix of shape `len(image_info) x len(text_list)` where `similarity_matrix[i, j]` is the CLIP ViT-L/14 similarity between image `i` and sentence `j`.

Here's an example:

```
{'image_info': [{'face_detections': None,
                 'image_name': 'b9040a0dbb22.jpg',
                 'raw_url': 'http://www.hfitinfo.com/honda_fit_pics/3/2/index.90.jpg'},
                {'face_detections': None,
                 'image_name': 'db1c21bc8474.jpg',
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
 'url': 'http://www.hfitinfo.com/hofi-48.html'}
```
To get the assignments of images to sentences according to our recommended method, please see [compute_assignments.py](https://github.com/allenai/mmc4/blob/main/scripts/compute_assignments.py)

NOTE: We are depricating two fields for now: `matched_text_index` and `matched_sim`, which are incorrect in ~10% of documents. We will bring these fields back after we update the corpus. (see [#11](https://github.com/allenai/mmc4/issues/11))

### Image features

You can directly download CLIP ViT-L/14 features extracted from the images at urls like this:

`https://storage.googleapis.com/ai2-jackh-mmc4-public/images/clip_vitl14_shard_{$SHARD}_features.pkl`

where `SHARD` can vary from 0 to 23098. The total size of all the image feature files together is approximately 1.8Tb. Each `pkl` file is a dictionary that maps from image filename (accessible in the document jsons, see `image_name` above) to the associated CLIP feature.

## Accessing mmc4

If you are interested in accessing mmc4 (and mmc4-core) without the fewer faces restriction, please fill out [this form.](https://forms.gle/VYtcNY8aYaUANK9f8)

## Accessing raw images

We are not releasing raw images for now. But if you are interested in potential updates, you can contact us using [this google form](https://forms.gle/ytcjFNSZeCbEpPTH6).

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
