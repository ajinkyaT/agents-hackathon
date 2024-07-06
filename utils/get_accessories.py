from unstructured.partition.csv import partition_csv
import os
import json

def get_accessories_table():
    jsonl_file = "stihl_doc_store/product_accessories.json"
    if os.path.exists(jsonl_file):
        with open(jsonl_file, "r") as f:
            table_text = f.read()
    else:
        csv_file = "data/stihl/product_accessories.csv"
        table_text = str(partition_csv(filename=csv_file)[0]).strip()
        # Save the raw table text as a JSONL file
        with open(jsonl_file, "w") as f:
            f.write(table_text)
    return table_text
