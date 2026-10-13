# Arabic Wiki-727K: Building a Sectioned Arabic Text Dataset from Wikipedia

A reproducible pipeline that turns the raw Arabic Wikipedia dump into a clean, section-structured corpus in the style of **Wiki-727K** (the English text-segmentation dataset of [Koshorek et al., 2018](https://arxiv.org/abs/1803.09337)). Each article is split into sections and sentences, so the data can be used for text segmentation and other Arabic NLP research.

## Dataset

- **Source:** Arabic Wikipedia dump (2021), via [Kaggle](https://www.kaggle.com/datasets/z3rocool/arabic-wikipedia-dump-2021)
- **Language:** Arabic
- **Format:** one `.txt` file per article, one sentence per line, section headers marked

## File format

Each article is a text file. Section headers start with `========` followed by the section level and title; every other line is one sentence.

```
========,1,مقدمة
<sentence>
<sentence>
========,2,<section title>
<sentence>
<sentence>
...
```

The first section of every article is the introduction (`مقدمة`, level 1). Levels 2 to 5 follow the heading depth in the original article.

Files are spread evenly across a folder tree like the original Wiki-727K: 99 top-level folders (`AA`, `AB`, ...), each with 100 sub-folders (`000` to `099`).

## Pipeline

All steps are in [`notebooks/arabic_wiki727k_pipeline.ipynb`](notebooks/arabic_wiki727k_pipeline.ipynb).

1. **Download:** fetch the Arabic Wikipedia dump with `kagglehub` and extract it.
2. **Restructure:** convert wiki-style headings (`==`, `===`, ...) into the `========,level,title` format.
3. **Remove non-content sections:** drop "See also", "References" and "Sources" (`انظر أيضًا`, `مراجع`, `مصادر`).
4. **Split into sentences:** one sentence per line; lines shorter than 10 characters are removed.
5. **Filter short articles:** delete files with 3 lines or fewer.
6. **Normalize the Arabic text** (see below).
7. **Organize and export:** distribute the files into the folder tree and zip them.

### Text normalization

- Unify alef forms (`إ أ آ` → `ا`), `ئ` → `ي`, `ؤ` → `و`, and `ة` → `ه`
- Remove diacritics (tashkeel)
- Remove Latin letters, digits and symbols (only Arabic letters and basic punctuation are kept)
- Remove standalone `و`
- Tokenize with NLTK and remove Arabic stop words
- Split sentences with PyArabic and drop sentences shorter than 20 words (section headers are kept)

## Limitations

- **The text is normalized, not natural.** Stop words, diacritics, digits and short sentences are removed, and several letters are merged. This suits segmentation and topic-style tasks, but it is not suitable for training language models that need fluent, natural Arabic.
- **Wikipedia bias:** topic coverage and writing style follow Arabic Wikipedia, which is uneven across subjects.
- **Sentence splitting** relies on punctuation, so some splits may be imperfect.

## License and attribution

The text comes from Arabic Wikipedia, which is available under **CC BY-SA 4.0**, so this derived dataset is shared under the same license with attribution to Wikipedia contributors. Please also check the terms of the Kaggle source dump.

## How to reproduce

The notebook runs on Google Colab.

```bash
pip install -r requirements.txt
```

Open the notebook in Colab, run the cells in order, and update the output paths (Google Drive) to your own.

## Repository structure

```
.
├── README.md
├── notebooks/
│   └── arabic_wiki727k_pipeline.ipynb
├── requirements.txt
└── .gitignore
```

## Tools

Python, NLTK, PyArabic, kagglehub, tqdm, Google Colab

## Authors

- **Farah Youssef** · [LinkedIn](https://linkedin.com/in/farah-yousef-66031a328)
- **Adel Dwaedari**
