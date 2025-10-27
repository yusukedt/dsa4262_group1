# Download test data from the server
import json
import csv

json_path = "SGNex_A549_directRNA_replicate6_run1/data.json"
csv_path = "data.csv"

with open(json_path, 'r', encoding='utf-8') as json_file, \
     open(csv_path, 'w', newline='', encoding='utf-8') as csv_file:

    writer = None

    for line in json_file:
        if line.strip():
            record = json.loads(line)

            # Create the header row only once
            if writer is None:
                writer = csv.DictWriter(csv_file, fieldnames=record.keys())
                writer.writeheader()

            writer.writerow(record)

print("✅ Conversion complete:", csv_path)
