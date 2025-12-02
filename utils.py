# myapp/utils.py
import csv
from .models import Event

def import_csv(filepath, log_name):
    count = 0
    with open(filepath, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            Event.objects.create(
                log_name=log_name,
                case_id=row["case_id"],
                activity=row["activity"],
                timestamp=row["timestamp"],
            )
            count += 1

    return {
        "log_name": log_name,
        "rows_imported": count,
        "status": "success"
    }
