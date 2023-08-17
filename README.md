# COCO-SM

This repository contains the scripts and data for the evaluation of Vision-Language models' multilingual properties in 20 different languages.

COCO-SM (**COCO** **S**ynthetic **M**ultilingual Evaluation) includes translations of the COCO Karpathy test split into 20 languages using 3 different neural translation systems: [Google Translate](https://translate.google.com), [Microsoft Bing Translator](https://www.bing.com/translator) and [NLLB](https://ai.meta.com/research/no-language-left-behind/).

## Data Description

The original data is the [COCO Karpathy test split](https://www.kaggle.com/datasets/shtvkumar/karpathy-splits). We excluded 7 samples from it because NLLB could not translate some of their captions. The full list of excluded files is in `meta/excluded.txt`

We made three different versions of the translation:

1. Google Translate – `meta/google.json`
2. Microsoft Bing Translator – `meta/bing.json`
3. NLLB – `meta/nllb.json`

## Supported Languages

| Language        | Speakers |     |  Language         | Speakers |
| :-------------- | -------: | --: | :--------------- |---------:|
| Arabic (ar)     |    274 M |     | Korean (ko)      |     81 M |
| Armenian (hy)   |      4 M |     |  Persian (fa)     |     77 M |
| Chinese (zh)    |  1'118 M |     |  Polish (pl)      |     41 M |
| French (fr)     |    274 M |     |  Portuguese (pt)  |    257 M |
| German (de)     |    134 M |     |  Russian (ru)     |    258 M |
| Hebrew (he)     |      9 M |     |  Spanish (es)     |    548 M |
| Hindi (hi)      |    602 M |     |  Thai (th)        |     61 M |
| Indonesian (id) |    199 M |     |  Turkish (tr)     |     88 M |
| Italian (it)    |    67 M  |     |  Ukranian (uk)    |     41 M |
| Japanese (ja)   |    125 M |     |  Vietnamese (vi)  |     85 M |

## Methodology and Metrics

Since any neural translation system introduces noise in the evaluation due to its imperfection and translation task ambiguity, three different versions of translation from three different NT systems are used for metrics calculation to get a more robust estimation of model performance.

COCO-SM is mainly for text-image retrieval evaluation. We use the following metrics:

* Recall@(1 / 5 / 10): the percentage of relevant results retrieved by a query among top-1/5/10 candidates. It depicts the retrieval performance of the model on a particular language. Recall is in [0, 1] range, the higher – the better.

* NDCG@20 (Normalized Discounted Cumulative Gain at 20): the measure of two rankings similarity. it compares how the model ranks top-20 candidates retrieved by an English query and ones retrieved by the same query on Language X. NDCG is in [0, 1] range, the higher – the better.

If there is a high NDCG on Language X, it will mean consistency of model behaviour on English and Language X. So, in that case, we can assess model performance on Language X via metrics on English which would be a robust estimation because English data isn't synthetic.

## Usage

First of all, the Python file with image and text encoding function definitions and model description is needed. 

Consider the example of [UForm](https://github.com/unum-cloud/uform) models which is located in `modules/uform.py`:

```python
import uform

def image_forward_fn(model, images, device, transform):
    images = model.preprocess_image(images).to(device)
    return model.encode_image(images)

def text_forward_fn(model, texts, device, transform):
    texts = model.preprocess_text(texts)
    texts = {k: v.to(device) for k, v in texts.items()}
    return model.encode_text(texts)

embedding_dim = 256
image_preprocess = None
text_preprocess = None

model = uform.get_model(
    'unum-cloud/uform-vl-multilingual-v2'
)
```

* `image_forward_fn` is the function which will be called to get image embeddings
* `text_forward_fn` is the function which will be called to get text embeddings

Both functions accept the following arguments: `model` – an instance of your model, `images`/`texts` – list of PIL Image instances/list of strings, `device` – id of the device on which evaluation will be done, `transform` – optional image/text transform.

* `embedding_dim` – dimension of image/text embeddings
* `image_preprocess` – optional transform for image preprocessing
* `text_preprocess` – optional transform for text preprocessing
* `model` – model that will be evaluated

An example of [OpenCLIP](https://github.com/mlfoundations/open_clip) model file is also available in the `modules` directory.

After the model file is ready, place it in the `modules` directory.
Finally, evaluation can be run via:

```bash
cd coco-sm

python eval.py \
> --model_name {model file name without .py} \
> --image_dir_path {path to the directory with COCO val images} \
> --meta_files_paths {paths to meta files with translations, located in the meta directory} \
> --batch_size {size of batch} \
> --device {device id} \
> --report_name {name of the file with the report, will be located in the reports directory}
```

For instance, for evaluating the UForm model on all translations execute this command:

```bash
python3 eval.py --model_name 'uform' --image_dir_path 'val2014' --meta_files_paths 'meta/google.json' 'meta/bing.json' 'meta/nllb.json' --batch_size 512 --device 'cuda:0' --report_name 'uform-multilingual-v2'
```

The evaluation script produces two files:

* `reports/{report_name}.csv` with metrics for all translations 
* `reports/{report_name}_reduced.csv` with averaged metrics across translations/languages

The evaluation results for UForm and OpenCLIP models can be found in the `reports` directory.


## Results

### Recall

| Target Language       | OpenCLIP @ 1 | UForm @ 1     | OpenCLIP @ 5 | UForm @ 5     | OpenCLIP @ 10 | UForm @ 10     | Speakers |
| :-------------------- | -----------: | ------------: | -----------: | -------------:| ------------: | --------------:| -------: |
| Arabic             |         22.7 |      **31.7** |         44.9 |      **57.8** |          55.8 |       **69.2** |    274 M |
| Armenian           |          5.6 |      **22.0** |         14.3 |      **44.7** |          20.2 |       **56.0** |      4 M |
| Chinese            |         27.3 |      **32.2** |         51.3 |      **59.0** |          62.1 |       **70.5** |  1'118 M |
| English            |     **37.8** |          37.7 |         63.5 |      **65.0** |          73.5 |       **75.9** |  1'452 M |
| French             |         31.3 |      **35.4** |         56.5 |      **62.6** |          67.4 |       **73.3** |    274 M |
| German             |         31.7 |      **35.1** |         56.9 |      **62.2** |          67.4 |       **73.3** |    134 M |
| Hebrew             |         23.7 |      **26.7** |         46.3 |      **51.8** |          57.0 |       **63.5** |      9 M |
| Hindi              |         20.7 |      **31.3** |         42.5 |      **57.9** |          53.7 |       **69.6** |    602 M |
| Indonesian         |         26.9 |      **30.7** |         51.4 |      **57.0** |          62.7 |       **68.6** |    199 M |
| Italian            |         31.3 |      **34.9** |         56.7 |      **62.1** |          67.1 |       **73.1** |     67 M |
| Japanese           |         27.4 |      **32.6** |         51.5 |      **59.2** |          62.6 |       **70.6** |    125 M |
| Korean             |         24.4 |      **31.5** |         48.1 |      **57.8** |          59.2 |       **69.2** |     81 M |
| Persian            |         24.0 |      **28.8** |         47.0 |      **54.6** |          57.8 |       **66.2** |     77 M |
| Polish             |         29.2 |      **33.6** |         53.9 |      **60.1** |          64.7 |       **71.3** |     41 M |
| Portuguese         |         31.6 |      **32.7** |         57.1 |      **59.6** |          67.9 |       **71.0** |    257 M |
| Russian            |         29.9 |      **33.9** |         54.8 |      **60.9** |          65.8 |       **72.0** |    258 M |
| Spanish            |         32.6 |      **35.6** |         58.0 |      **62.8** |          68.8 |       **73.7** |    548 M |
| Thai               |         21.5 |      **28.7** |         43.0 |      **54.6** |          53.7 |       **66.0** |     61 M |
| Turkish            |         25.5 |      **33.0** |         49.1 |      **59.6** |          60.3 |       **70.8** |     88 M |
| Ukranian           |         26.0 |      **30.6** |         49.9 |      **56.7** |          60.9 |       **68.1** |     41 M |
| Vietnamese         |         25.4 |      **28.3** |         49.2 |      **53.9** |          60.3 |       **65.5** |     85 M |
|                       |              |               |              |               |               |                |          |
| Mean                  |     26.5±6.4 |  **31.8±3.5** |     49.8±9.8 |  **58.1±4.5** |     60.4±10.6 |   **69.4±4.3** |        - |
| Google Translate      |     27.4±6.3 |  **31.5±3.5** |     51.1±9.5 |  **57.8±4.4** |     61.7±10.3 |   **69.1±4.3** |        - |
| Microsoft Translator  |     27.2±6.4 |  **31.4±3.6** |     50.8±9.8 |  **57.7±4.7** |     61.4±10.6 |   **68.9±4.6** |        - |
| Meta NLLB             |     24.9±6.7 |  **32.4±3.5** |    47.5±10.3 |  **58.9±4.5** |     58.2±11.2 |   **70.2±4.3** |        - |

### NDCG@20

|               |     Arabic |     Armenian |     Chinese |     French |     German |     Hebrew |     Hindi |     Indonesian |     Italian |     Japanese |     Korean |     Persian |     Polish |     Portuguese |     Russian |     Spanish |     Thai |     Turkish |     Ukranian |     Vietnamese |   Mean (all) | Mean (Google Translate) | Mean(Microsoft Translator) | Mean(NLLB)
| :------------ | ----: | ----: | ----: | ----: | ----: | ----: | ----: | ----: | ----: | ----: | ----: | ----: | ----: | ----: | ----: | ----: | ----: | ----: | ----: | ----: | ----: | ----: | ----: | ----: |
| OpenCLIP NDCG | 0.639 | 0.204 | 0.731 | 0.823 | 0.806 | 0.657 | 0.616 | 0.733 | 0.811 | 0.737 | 0.686 | 0.667 | 0.764 | 0.832 | 0.777 | 0.849 | 0.606 | 0.701 | 0.704 | 0.697 | 0.716 ± 0.149 | 0.732 ± 0.145 | 0.730 ± 0.149 | 0.686 ± 0.158
| UForm NDCG    | 0.868 | 0.691 | 0.880 | 0.932 | 0.927 | 0.791 | 0.879 | 0.870 | 0.930 | 0.885 | 0.869 | 0.831 | 0.897 | 0.897 | 0.906 | 0.939 | 0.822 | 0.898 | 0.851 | 0.818 | 0.875 ± 0.064 | 0.869 ± 0.063 | 0.869 ± 0.066 | 0.888 ± 0.064