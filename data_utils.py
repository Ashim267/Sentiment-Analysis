import os
import csv
import pandas as pd

def process_txt_to_csv(input_txt, output_csv):
    if not os.path.exists(output_csv):
        if os.path.exists(input_txt):
            with open(output_csv, "w", newline="", encoding="utf-8") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(["label", "title", "review"])

                with open(input_txt, "r", encoding="utf-8") as text_file:
                    for line in text_file:
                        review_data = line.split(" ", 1)
                        label = review_data[0]
                        review_data = review_data[1].split(":", 1) if len(review_data) > 1 else ["", ""]
                        title = review_data[0].strip()
                        review = review_data[1].strip()
                        writer.writerow([label, title, review])
        else:
            print(f"File {input_txt} not found!")
    else:
        print(f"File {output_csv} already exists.")

def load_and_shuffle(train_csv, test_csv):
    train = pd.read_csv(train_csv).sample(frac=1).reset_index(drop=True)
    test = pd.read_csv(test_csv).sample(frac=1).reset_index(drop=True)
    return train, test
