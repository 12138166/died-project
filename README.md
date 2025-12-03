# Table OCR Column Extractor  -died-project （for Bloomberg 

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

```



## Install Tesseract OCR (examples):

macOS (Homebrew):
```bash
brew install tesseract
```

Ubuntu / Debian:
```bash
sudo apt-get install tesseract-ocr
```

Windows:
Download and install from the official project page or a trusted binary distribution,
and ensure the installation path is added to your system PATH.



## Usage

Run the script from the command line:
```bash
python ocr_table_extractor.py \
  --input_folder /path/to/images \
  --output_folder /path/to/output_csvs
```

--input_folder: directory containing input images (.png, .jpg, .jpeg, .tiff)

--output_folder: directory where per-image CSV files will be saved

For each image, the script will:

Preprocess the image (grayscale + invert) to improve OCR.

Run Tesseract OCR with image_to_data to obtain per-word bounding boxes.

Detect the header row using predefined header keywords.

Infer column boundaries from header word positions.

Assign each token to a column based on its x-center.

Group tokens into rows and output a CSV with the following columns:

```bash
Supplier Name
Industry
Mkt Cap (M)
Total Relationship Size M)
% Cost
Cost Category
% Supplier's Revenue
Source
Size Source
```

You can modify the list of column_names in ocr_table_extractor.py to match
your own table layout.

## Customization

Header detection:
The function find_column_boundaries uses a dictionary of header keywords to
locate the header row and compute column anchors. You can edit the
header_keywords mapping to adapt the script to other table formats.

Column set:
The final column order is defined in column_names inside main().
Add/remove columns or rename them as needed.

Filtering & cleaning:
Confidence thresholds and text cleaning (e.g., removing row numbers from
"Supplier Name") can be adjusted inside extract_data_from_image_by_coordinates.



## Disclaimer

This repository is intended as a template for building your own ETL
pipeline between Snowflake and Dropbox.

Always make sure:
- Your use of data complies with your data provider’s terms and conditions.
- You keep all credentials in a secure place and never commit them to git.
