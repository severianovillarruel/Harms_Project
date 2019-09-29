#!/usr/bin/env python3

import json

JASON_FILE = open("/home/severiano/harms_proj/data/aaindex-pca.json", "r")
data = json.load(JASON_FILE)

print(data["aaindex_pca_1"]["values"])

JSON_FILE.close()
