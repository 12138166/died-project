# ocr-table-to-csv
import os
import cv2
import pytesseract
import pandas as pd
import re
from collections import defaultdict

# ===================================================================
# == MAGIC FIX: Force OpenCV to fully initialize before use. ==
_ = cv2.getVersionString()


import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", required=True)
    parser.add_argument("--output_folder", required=True)
    return parser.parse_args()
    
def preprocess_image(image_path):
    """
    Loads image, converts to grayscale, and inverts colors to improve OCR.
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Failed to read image {image_path}")
            return None
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        inverted_image = cv2.bitwise_not(gray)
        return inverted_image
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None


def find_column_boundaries(ocr_df, image_width):
    """
    Finds header row and defines boundaries based on the average position of header keywords.
    This version reverts to the more stable anchor-finding logic.
    """
    header_keywords = {
        'Supplier Name': ['Supplier', 'Name'], 'Industry': ['Industry'],
        'Mkt Cap (M)': ['Mkt', 'Cap'], 'Total Relationship Size M)': ['Total', 'Relationship'],
        '% Cost': ['Cost'], 'Cost Category': ['Cost', 'Category'],  # Note: Using just 'Cost' for '% Cost'
        "% Supplier's Revenue": ["Supplier's", 'Revenue'], 'Source': ['Source'],
        'Size Source': ['Size', 'Source']
    }

    ocr_df = ocr_df[(ocr_df['conf'] > 30) & (ocr_df['text'].notna()) & (ocr_df['text'].str.strip() != '')].copy()

    header_candidates = defaultdict(int)
    keyword_locations = defaultdict(list)
    for col_name, keywords in header_keywords.items():
        pattern = '|'.join(keywords)
        matches = ocr_df[ocr_df['text'].str.contains(pattern, case=False, na=False)]
        for _, row in matches.iterrows():
            line_id = (row['block_num'], row['par_num'], row['line_num'])
            header_candidates[line_id] += 1
            keyword_locations[col_name].append(row)

    if not header_candidates: return None, -1
    best_line_id = max(header_candidates, key=header_candidates.get)

    words_in_header = ocr_df[(ocr_df['block_num'] == best_line_id[0]) & (ocr_df['par_num'] == best_line_id[1]) & (
                ocr_df['line_num'] == best_line_id[2])]
    header_top_y = words_in_header['top'].mean() if not words_in_header.empty else 0

    anchors = []
    for col_name, keywords in header_keywords.items():
        col_header_words = [word_row for word_row in keyword_locations.get(col_name, []) if
                            (word_row['block_num'], word_row['par_num'], word_row['line_num']) == best_line_id]
        if col_header_words:
            # Revert to the more stable "average of keyword centers" logic
            positions = [row['left'] + row['width'] / 2 for row in col_header_words]
            anchor_x = sum(positions) / len(positions)
            anchors.append({'name': col_name, 'x': anchor_x})

    anchors.sort(key=lambda a: a['x'])

    boundaries = []
    for i in range(len(anchors)):
        col_name = anchors[i]['name']
        left_bound = 0 if i == 0 else (anchors[i - 1]['x'] + anchors[i]['x']) / 2
        right_bound = image_width if i == len(anchors) - 1 else (anchors[i]['x'] + anchors[i + 1]['x']) / 2
        boundaries.append({'name': col_name, 'left': left_bound, 'right': right_bound})

    return boundaries, header_top_y


def extract_data_from_image_by_coordinates(image_path, column_names):
    """
    Extracts table data by assigning words to predefined column boundaries.
    """
    preprocessed_img = preprocess_image(image_path)
    if preprocessed_img is None: return None

    image_height, image_width = preprocessed_img.shape[:2]

    custom_config = r'--oem 3 --psm 6'
    ocr_df = pytesseract.image_to_data(preprocessed_img, config=custom_config, output_type=pytesseract.Output.DATAFRAME)

    boundaries, header_y = find_column_boundaries(ocr_df, image_width)
    if not boundaries:
        print(f"Warning: Could not locate headers in {image_path}. Skipping file.")
        return None

    data_df = ocr_df[(ocr_df['top'] > header_y + 10) & (ocr_df['conf'] > 40) & (ocr_df['text'].notna()) & (
                ocr_df['text'].str.strip() != '')].copy()
    if data_df.empty:
        print(f"Warning: No valid data rows found in {image_path}.")
        return None

    def get_column_for_word(x_pos, boundaries):
        for col in boundaries:
            if col['left'] <= x_pos < col['right']:
                return col['name']
        return None

    data_df['x_center'] = data_df['left'] + data_df['width'] / 2
    data_df['col'] = data_df['x_center'].apply(lambda x: get_column_for_word(x, boundaries))
    data_df.dropna(subset=['col'], inplace=True)

    lines = defaultdict(lambda: defaultdict(list))
    for _, row in data_df.sort_values('left').iterrows():
        line_id = (row['block_num'], row['par_num'], row['line_num'])
        lines[line_id][row['col']].append(str(row['text']))

    processed_rows = [{col: ' '.join(words) for col, words in lines[line_id].items()} for line_id in
                      sorted(lines.keys())]
    if not processed_rows: return None

    final_df = pd.DataFrame(processed_rows)

    # Precise cleaning for the row number, as per your instruction
    if 'Supplier Name' in final_df.columns:
        final_df['Supplier Name'] = final_df['Supplier Name'].str.replace(r'^\d+\)?\s*', '', regex=True)

    found_and_ordered_columns = [col for col in column_names if col in final_df.columns]
    return final_df[found_and_ordered_columns]

# path!!!!
def main():
    # input_folder = ''
    # output_folder = ''
    args = parse_args()
    input_folder = args.input_folder
    output_folder = args.output_folder

    if not os.path.exists(output_folder): os.makedirs(output_folder)

    # The exact 9 column names you specified
    column_names = [
        'Supplier Name', 'Industry', 'Mkt Cap (M)', 'Total Relationship Size M)',
        '% Cost', 'Cost Category', "% Supplier's Revenue", 'Source', 'Size Source'
    ]
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff'))]

    print(f"Found {len(image_files)} images. Starting processing...")
    for i, filename in enumerate(image_files):
        print(f"Processing image {i + 1}/{len(image_files)}: {filename} ...")
        image_path = os.path.join(input_folder, filename)

        df = extract_data_from_image_by_coordinates(image_path, column_names)

        if df is not None and not df.empty:
            base_name = os.path.splitext(filename)[0]
            output_csv_path = os.path.join(output_folder, f"{base_name}.csv")
            df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
            print(f"  -> Successfully saved to {output_csv_path}")
        else:
            print(f"  -> Failed to extract data from {filename}.")
    print("All images processed!")


if __name__ == '__main__':
    main()
