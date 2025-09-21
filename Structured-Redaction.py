import json
import random
from faker import Faker
from datetime import datetime

fake = Faker()

with open("faker_config.json") as f:
    faker_config = json.load(f)



def generate_fake_value(field, original_value, rule):
    faker_type = rule["faker"]

    if faker_type == "name":
        return fake.name()
    elif faker_type == "name_with_title":
        return fake.prefix() + " " + fake.name()
    elif faker_type == "date":
        dt = fake.date_between(start_date="-20y", end_date="today")
        return dt.strftime(rule.get("format", "%Y-%m-%d"))
    elif faker_type == "address":
        return fake.address().replace("\n", ", ")
    elif faker_type == "phone_number":
        return fake.phone_number()
    elif faker_type == "email":
        return fake.email()
    elif faker_type == "ssn":
        return fake.ssn()
    elif faker_type == "bban":
        return fake.bban()
    elif faker_type == "random_number":
        return str(random.randint(rule["min"], rule["max"]))
    else:
        return original_value  # fallback

def process_document(doc):
    output = {}
    for section, fields in doc.items():
        output[section] = {}
        for field, val in fields.items():
            if field in faker_config:  # redact only configured fields
                output[section][field] = generate_fake_value(field, val, faker_config[field])
            else:
                output[section][field] = val
    return output

def process_batch(docs):
    return [process_document(doc) for doc in docs]


# ---- Example Run ----
if __name__ == "__main__":
    with open("input_docs.json") as f:
        input_docs = json.load(f)

    redacted_docs = process_batch(input_docs)

    print(json.dumps(redacted_docs, indent=2))
