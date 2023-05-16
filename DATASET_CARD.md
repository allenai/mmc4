Multimodal C4 (mmc4): An Open, Billion-scale Corpus of Images Interleaved With Text

# Dataset Card for mmc4

## Table of Contents
- [Dataset Description](#dataset-description)
  - [Dataset Summary](#dataset-summary)
  - [Supported Tasks](#supported-tasks-and-leaderboards)
  - [Languages](#languages)
- [Dataset Structure](#dataset-structure)
  - [Data Instances](#data-instances)
  - [Data Fields](#data-fields)
  - [Data Splits](#data-splits)
- [Dataset Creation](#dataset-creation)
  - [Curation Rationale](#curation-rationale)
  - [Source Data](#source-data)
  - [Annotations](#annotations)
  - [Personal and Sensitive Information](#personal-and-sensitive-information)
- [Considerations for Using the Data](#considerations-for-using-the-data)
  - [Social Impact of Dataset](#social-impact-of-dataset)
  - [Discussion of Biases](#discussion-of-biases)
  - [Other Known Limitations](#other-known-limitations)
- [Additional Information](#additional-information)
  - [Dataset Curators](#dataset-curators)
  - [Licensing Information](#licensing-information)
  - [Citation Information](#citation-information)

## Dataset Description

- **Homepage:** www.github.com/allenai/mmc4
- **Repository:** www.github.com/allenai/mmc4
- **Paper:** https://arxiv.org/abs/2304.06939
- **Point of Contact:** Jack Hessel (jackh@allenai.org)

### Dataset Summary

We release mmc4, an augmentation of the popular text-only [c4 corpus](https://www.tensorflow.org/datasets/catalog/c4) with images interleaved.
We use a linear assignment algorithm to place images into longer bodies of text using CLIP features. mmc4 spans everyday topics like cooking, travel, techology, etc. A manual inspection of a random sample of documents shows that the images are, on average, relevant to the topic/content of the text, and, frequently specific to their assigned sentences. After filtering NSFW images, ads, etc., the corpus contains 585M images across 103M text documents interleaved with 43B English tokens.

### Supported Tasks and Leaderboards

This is a pre-training corpus.

### Languages

English

## Dataset Structure


You can directly download the "fewer faces" multimodal c4 documents at urls like this:

`https://storage.googleapis.com/ai2-jackh-mmc4-public/data_v1.1/docs_no_face_shard_{$SHARD}_v2.jsonl.zip`

You can directly download CLIP ViT-L/14 features extracted from the images at urls like this:

`https://storage.googleapis.com/ai2-jackh-mmc4-public/images/clip_vitl14_shard_{$SHARD}_features.pkl`

`SHARD` can vary from 0 to 23098. 


### Data Instances

Documents are arranged as follows:

- `text_list`: a list of sentences comprising the text of the document
- `url`: the original url where the document was hosted
- `image_info` is a key mapping to a list of images. each image contains:
  - `image_name`: a filename that you could download the image to
  - `face_detections`: `None` if no faces are detected (which should be the case in "fewer faces")
  - `matched_text_index`: the index within `text_list` representing the sentence that this image is matched to
  - `matched_sim`: the CLIP ViT-L/14 similarity between the image and the sentence at the matched index
- `similarity_matrix`: a matrix of shape `len(image_info) x len(text_list)` where `similarity_matrix[i, j]` is the CLIP ViT-L/14 similarity between the `i`-th image and the `j`-th sentence.

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
 'url': 'http://www.hfitinfo.com/hofi-48.html'}
```

For image features, each `pkl` file is a dictionary that maps from image filename (accessible in the document jsons, see the nested `image_name` in `image_info` above) to the associated CLIP feature.

### Data Fields

See above.

### Data Splits

N/A, this is a pretraining corpus.

## Dataset Creation

The dataset was created from the middle of 2022 to early 2023 at the Allen Institute for AI.

### Curation Rationale

In-context learning \cite{brown2020language} enables sequence models to adapt to new tasks without any parameter updates by interleaving a few supervised examples in a prompt. Some multimodal models also support in-context learning. Prior experiments \cite{alayrac2022flamingo} suggest that performant, multimodal in-context learning is dependent upon pretraining on a corpus containing interleaved sequences of images and text (rather than single image/caption pairs). However, such a large-scale corpus has not been made publicly available.

To address this, we introduce mmc4, a publicly available, billion-scale image-text dataset consisting of interleaved image/text sequences.

### Source Data

#### Initial Data Collection and Normalization

See the paper for more details.

#### Who are the source language producers?

Authors of publicly accessible webpages.

### Annotations

#### Annotation process

See the paper for more details. The corpus, as a whole, is not explicitly annotated.

#### Who are the annotators?

N/A

### Personal and Sensitive Information

See the paper for an assessment of the risks of releasing image URLs. In particular, for the public, directly-downloadable corpus, we attempt to remove instances with faces.

## Considerations for Using the Data

### Social Impact of Dataset

Potential benefits:

- Useful as a pretraining corpus for in-context vision+language models; such models could be adapted later to better align with human preferences/express fewer pernicious social biases
- As in-context vision+language models become more common, if the standard pretraining set is public, it will be easier to audit models with respect to the training corpora.

Potential risks:

- As with most large-scale image datasets: images of individuals who did not explicitly consent to be in the dataset are included. Given the scale of the dataset, we think the risk of including these images is similar to the risk of such images being indexed by search engines.
- mmc4 inherits the risks of the text-only version of c4, e.g., the internet reflects pernicious social biases, and thus models trained on this corpus might also reproduce those biases at test time.

### Discussion of Biases

Web data, especially taken as a whole, often reflects the biases present in society. We encourage model trainers to reflect upon the distinction between an observational model of web text (e.g., as a means of auditing what is contained in that web text) versus a model that one endorses the outputs of as "correct", vs. one connects to other downstream systems that cause deployed systems to make decisions.

### Other Known Limitations

- The dataset is English only.
- Our filtration process discards images that do not relate to the text of web-pages above a specific model-estimated threshold. This might erase images/webpages that use image content in more creative, non-iteral ways.

## Additional Information

### Dataset Curators

This dataset was initially curated by researchers from AI2, UCSB, University of Washington, Columbia University, Yonsei University. The author list of v1 of the arxiv paper is an accurate list of specific contributors.

### Licensing Information

- The new contributions of mmc4 are released under ODC-BY.
- By using mmc4, be aware of that you are also bound by the Common Crawl terms of use.

### Citation Information

```
@article{zhu2023multimodalc4,
    title={{MultimodalC4:} An Open, Billion-scale Corpus of Images Interleaved With Text},
    author={Zhu, Wanrong and Hessel, Jack and Awadalla, Anas and Gadre, Samir Yitzhak and Dodge, Jesse and Fang, Alex and Yu, Youngjae and Schmidt, Ludwig and Wang, William Yang and Cho, Yejin},
    year={2023},
}
```
