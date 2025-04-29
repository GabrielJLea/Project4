# docchat
![Tests](https://github.com/GabrielJLea/Project4/actions/workflows/tests.yml/badge.svg)

DocChat is a chatbot that allows users to ask questions and engage in continuous conversations about documents.

## Supported Document Types:
- HTML files
- PDFs
- Webpages
- Images

## Requirements

```
$ pip3 install -r requirements.txt
```

## Examples

Gif showing how to run:

![example](Video/LLM.gif)

Text examples:

```
$ python docchat.py animals.txt

docchat> what is this document about
DOCCHAT: This document appears to be a collection of species profiles, highlighting unique and interesting facts about five different animals: the Peregrine Falcon, Axolotl, Quokka, Mantis Shrimp, and Orangutan.

docchat> tell me about the dolphin in this document
DOCCHAT: There is no mention of a dolphin in this document.
```
