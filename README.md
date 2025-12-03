# Table OCR Column Extractor  -died-project

This repository contains a small utility script for extracting **structured tables**
from images using Tesseract OCR and a simple geometric heuristic.

It was originally developed for a finance project that required converting
semi-structured table images (e.g., supplier tables in reports) into clean CSV files.

---

## Features

- Uses **Tesseract OCR** via `pytesseract` to recognize text and bounding boxes.
- Automatically detects the **header row** using configurable keyword anchors.
- Infers **column boundaries** from the horizontal positions of header words.
- Assigns each token to a column based on its x-coordinate.
- Groups words into rows and exports a **clean CSV** with predefined column names.

---

## Requirements

- Python 3.8+ (other versions may also work)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) installed on your system
- Python packages (see `requirements.txt`):
  - `pytesseract`
  - `opencv-python`
  - `pandas`

Make sure `tesseract` is available in your system PATH, or configure
`pytesseract.pytesseract.tesseract_cmd` if needed.

---

## Installation

Create a virtual environment (optional but recommended), then:

```bash
pip install -r requirements.txt

