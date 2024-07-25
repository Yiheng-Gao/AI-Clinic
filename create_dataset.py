import csv
from RecordDAL import get_full_records

def get_csv():
    file_path="./datasets/dataset.csv"

    records = get_full_records()

    with open(file_path,'w',newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["question","answer","rate","fine_tune"])
        writer.writerows(records)


