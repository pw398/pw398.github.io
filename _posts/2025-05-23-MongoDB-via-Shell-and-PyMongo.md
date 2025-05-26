---
layout: post
title:  "MongoDB via Shell and PyMongo"
date:   2025-05-23 00:00:00 +0000
categories: MongoDB Bash Python
---

Intro text...

Image?


# Outline

1. NoSQL Databases
    - Advantages of Unstructured Data (flexible schema, discuss ACID/CAPS...)
2. MongoDB
3. Installation
    - Linux Ubuntu, Windows, Conda
4. Connect to Server
    - Import Common Libraries
    - Connect with MongoClient
5. Import Clickstream Data
    - Delete DB if Exists (Optional)
    - Set Variables
    - List Databases
6. Select DB and View Collections
7. View Sample Documents
8. Count Records
    - Record Counts by Collectioin
    - Count of Unique Values by Field
9. CRUD Operations (Create, Read, Update, Delete) 
    - Remove and Create
    - Read
    - Update
    - Create New Field
10. Indexing
    - View Indexes
    - Create Indexes


**review the outline for accuracy**


# Introduction

## Advantages of Unstructured Data
- ...
- ...

## MongoDB
- ...
- ...



# Installation

- Linux
- Windows
- Mongo Tools (mongosrestore, mongodump, etc.)
- PyMongo

```python
# !pip install pymongo
```



# Import Libraries and Data

## Import Libraries

```python
from pymongo import MongoClient
from pymongo import ASCENDING, DESCENDING
import os
import subprocess 
import json
from bson import json_util, ObjectId
from datetime import datetime
import time
import pprint as pp
```


## Connect to Server

<p></p>

```python
HOST = "localhost"
PORT = "27017"
```

<p></p>

```python
conn_str = f"mongodb://{HOST}:{PORT}/"

print("Connection string:")
print(conn_str)
```

<p></p>

```python
# Connection string:
# mongodb://localhost:27017/
```


### Using Bash

```python
print("Databases:")
!mongosh "$conn_str" --quiet --eval "db.adminCommand('listDatabases').databases.map(d => d.name)"
```

<p></p>

```python
# Databases:
# [ 'admin', 'config', 'local' ]
```


### Using Python

```python
client = MongoClient(conn_str)
print("Databases:")
print(client.list_database_names())
```

<p></p>

```python
# Databases:
# ['admin', 'config', 'local']
```


## Import Data

```python
DBNAME = "clickstream"
db = client[DBNAME]
```

### Delete DB If Exists (Optional)

This is only for demonstration purposes, as the <code>--drop</code> statement used in our below import commands will drop the target database before importing, if it exists.

#### Delete Using Bash

```bash
!mongosh --quiet --eval "db.getSiblingDB('{DBNAME}').dropDatabase()"
!mongosh --quiet --eval "db.adminCommand('listDatabases').databases.forEach(db => print(db.name))"
```

```bash
# { ok: 1, dropped: 'clickstream' }
# admin
# config
# local
```


#### Delete Using Python 

```python
client.drop_database("clickstream")
print(client.list_database_names())
```

<p></p>

```python
# ['admin', 'config', 'local']
```


###  Import Data From File

#### Set Variables

```python
IMPORT_FILE_FOLDER = r"C:\Users\patwh\Downloads"
BSON_FILE_NAME = "clicks"
JSON_FILE_NAME = "clicks.metadata"

bson_file = f"{IMPORT_FILE_FOLDER}/{BSON_FILE_NAME}.bson" 
json_file = f"{IMPORT_FILE_FOLDER}/{JSON_FILE_NAME}.json" 

COLLECTION_BSON = BSON_FILE_NAME
COLLECTION_JSON = JSON_FILE_NAME
```


#### Using Bash Only

```python
start_time = time.time()

!mongorestore --host localhost:{PORT} --db {DBNAME} --collection {COLLECTION_BSON} --drop "{bson_file}"
!mongoimport --host localhost:{PORT} --db {DBNAME} --collection {COLLECTION_JSON} --drop --type json "{json_file}"

elapsed_time = time.time() - start_time
print(f"Time to Import: {elapsed_time:.0f} seconds")
```

<p></p>

```bash
# 2025-05-25T13:10:22.270-0600    finished restoring clickstream.clicks (6100000 documents, 0 failures)
# 2025-05-25T13:10:22.270-0600    no indexes to restore for collection clickstream.clicks
# 2025-05-25T13:10:22.270-0600    6100000 document(s) restored successfully. 0 document(s) failed to restore.
# 2025-05-25T13:10:22.954-0600    connected to: mongodb://localhost:27017/
# 2025-05-25T13:10:22.955-0600    dropping: clickstream.clicks.metadata
# 2025-05-25T13:10:22.987-0600    1 document(s) imported successfully. 0 document(s) failed to import.

# Time to Import: 105 seconds
```


#### Using Python's <code>subprocess</code>

```python
start_time = time.time()

# Import bson file
subprocess.run(f'mongorestore --host {HOST}:{PORT} --db {DBNAME} --collection {COLLECTION_BSON} --drop "{bson_file}"')

# Import json file
with open(json_file) as f:
    # Use json_util to parse extended JSON
    data = json_util.loads(f.read())
    # Ensure data is a list for insert_many, or a single document for insert_one
    if isinstance(data, list):
        db[COLLECTION_JSON].insert_many(data)
    else:
        db[COLLECTION_JSON].insert_one(data)

# Print document counts
print(f"Records in {COLLECTION_BSON}: {db[COLLECTION_BSON].count_documents({})}")
print(f"Records in {COLLECTION_JSON}: {db[COLLECTION_JSON].count_documents({})}")

elapsed_time = time.time() - start_time
print(f"Time to Import: {elapsed_time:.0f} seconds")
```

<p></p>

```python
# Records in clicks: 6100000
# Records in clicks.metadata: 1
# Time to Import: 99 seconds
```


## List Databases

### Using Bash

```python
!mongosh --host {HOST} --port {PORT} --quiet --eval "db.getMongo().getDBs().databases.map(db => db.name)"
```

<p></p>

```python
# [ 'admin', 'clickstream', 'config', 'local' ]
```


### Using Python

```python
print(client.list_database_names())
```

<p></p>

```python
# ['admin', 'clickstream', 'config', 'local']
```



# Select DB and View Collections


### Using Bash with Environment Variables

While the below demonstrates that we can select the <code>clickstream</code> and <code>list</code> out its collections, these commands executed by the <code>.bat</code> field will not persist; i.e., we will need to select the database again when a separate command is issued later.

Using an external file with environment variables (which appear like <code>%DBNAME%</code>):

**explain the metadata file**

```python
# list_collections_env_vars.bat

filename = "list_collections_env_vars.bat"

batch_content = """
@echo off
echo Listing collections in %DBNAME%
mongosh --host %HOST% --port %PORT% --quiet --eval "print(JSON.stringify(db.getSiblingDB('%DBNAME%').getCollectionNames()));"
"""

with open(filename, 'w') as file:
    file.write(batch_content)

print(f"Created {filename}")
```

<p></p>

```python
# Created list_collections_env_vars.bat
```

<p></p>

```python
os.environ['DBNAME'] = DBNAME
os.environ['HOST'] = HOST
os.environ['PORT'] = PORT

# Run the batch file
result = subprocess.run(
    r'C:\Users\patwh\Downloads\list_collections_env_vars.bat',
    shell=True,
    capture_output=True,
    text=True,
    check=True
)
output_lines = result.stdout.strip().splitlines()
json_line = output_lines[-1]
collections = json.loads(json_line)
print(collections)
```

<p></p>

```python
# ['clicks', 'clicks.metadata']
```


### Using Bash with Python Variables

Using an external file with Python (f-string) variables (which appear like <code>%1</code>, <code>%2</code>, etc. in the .bat file:

```python
# list_collections_py_vars.bat

filename = "list_collections_py_vars.bat"

batch_content = """
@echo off
echo Listing collections in %1
mongosh --host %2 --port %3 --quiet --eval "print(JSON.stringify(db.getSiblingDB('%1').getCollectionNames()));"
"""

with open(filename, 'w') as file:
    file.write(batch_content)
    
print(f"Created {filename}")
```

<p></p>

```python
Created list_collections_py_vars.bat
````

<p></p>

```python
result = subprocess.run(
    f'C:\\Users\\patwh\\Downloads\\list_collections_py_vars.bat "{DBNAME}" "{HOST}" "{PORT}"',
    shell=True,
    capture_output=True,
    text=True,
    check=True
)
output_lines = result.stdout.strip().splitlines()
json_line = output_lines[-1]
collections = json.loads(json_line)
print(collections)
```

<p></p>

```python
# ['clicks', 'clicks.metadata']
```

Without using an external file:

```python
result = subprocess.run(
    f'mongosh --host {HOST} --port {PORT} --quiet --eval "print(JSON.stringify(db.getSiblingDB(\'{DBNAME}\').getCollectionNames()))"',
    shell=True,
    capture_output=True,
    text=True
)
collections = json.loads(result.stdout.strip().splitlines()[-1])
print(collections)
```

<p></p>

```python
# ['clicks', 'clicks.metadata']
```

Going forward, we'll use the environment variables approach, since it is more descriptive.


### Using Python

```python
db = client["clickstream"]
db.list_collection_names()
```

<p></p>

```python
# ['clicks', 'clicks.metadata']
```



# Sample Documents


### Using Bash

```python
for collection in db.list_collection_names():
    result = subprocess.run(
        f'mongosh --host {HOST} --port {PORT} --quiet --eval "print(JSON.stringify(db.getSiblingDB(\'{DBNAME}\').{collection}.findOne()))"',
        shell=True,
        capture_output=True,
        text=True,
        check=True
    )
    output = json.loads(result.stdout.strip())
    print(f"collection: {collection}")
    pp.pprint(output)
    print("")
```

```python
# collection: clicks
# {'Activity': 'click',
#  'ProductID': 'Pr100037',
#  'VisitDateTime': '2018-05-25T04:51:14.179Z',
#  '_id': '60df1029ad74d9467c91a932',
#  'device': {'Browser': 'Firefox', 'OS': 'Windows'},
#  'user': {'City': 'Colombo', 'Country': 'Sri Lanka'},
#  'webClientID': 'WI100000244987'}

# collection: clicks.metadata
# {'_id': '6833a52eb8acd79ee91e3776'}
```


### Using Python

```python
collections = db.list_collection_names()
for collection in collections:
    collection = db[collection]
    print(f"collection: {collection.name}")
    pp.pprint(collection.find_one())
    print("")
```

<p></p>

```python
# collection: clicks
# {'Activity': 'click',
#  'ProductID': 'Pr100037',
#  'VisitDateTime': datetime.datetime(2018, 5, 25, 4, 51, 14, 179000),
#  '_id': ObjectId('60df1029ad74d9467c91a932'),
#  'device': {'Browser': 'Firefox', 'OS': 'Windows'},
#  'user': {'City': 'Colombo', 'Country': 'Sri Lanka'},
#  'webClientID': 'WI100000244987'}

# collection: clicks.metadata
# {'_id': ObjectId('6833a52eb8acd79ee91e3776')}
```



# Count Records


## Record Counts by Collection

### Using Bash

```python
# count_records_in_collections.bat

filename = "count_records_in_collections.bat"

batch_content = """
@echo off
echo Listing collections and document counts in %DBNAME%
mongosh --host %HOST% --port %PORT% --quiet --eval "db.getSiblingDB('%DBNAME%').getCollectionNames().forEach(name => print(name + ': ' + db.getSiblingDB('%DBNAME%')[name].countDocuments({}) + ' documents'))"
"""

with open(filename, 'w') as file:
    file.write(batch_content)
    
print(f"Created {filename}")
```

<p></p>

```python
# Created count_records_in_collections.bat
```

<p></p>

```python
os.environ['DBNAME'] = DBNAME
os.environ['HOST'] = HOST
os.environ['PORT'] = PORT

result = subprocess.run(
    filename,
    shell=True,
    capture_output=True,
    text=True,
    check=True
)

# Parse text output
output_lines = result.stdout.strip().splitlines()
collections_counts = {}
for line in output_lines[1:]:  # Skip echo line
    if line:
        collection, count = line.split(': ')
        collections_counts[collection] = int(count.split()[0])
print(collections_counts)
```

<p></p>

```python
# {'clicks': 6100000, 'clicks.metadata': 1}
```


### Using Python

```python
collections = db.list_collection_names()
for collection_name in collections:
    collection = db[collection_name]
    count = collection.count_documents({})
    print(f"{collection_name}: {count} documents")
```

<p></p>

```python
# clicks: 6100000 documents
# clicks.metadata: 1 documents
```


## Count of Unique Values by Field

```python
collection = db['clicks']
record_limit = 100000

# Get all fields except _id from a sample document
sample_doc = collection.find_one()
fields = []
def extract_fields(doc, prefix=''):
    for key, value in doc.items():
        if key != '_id':  # Skip _id field
            full_key = f"{prefix}{key}" if prefix else key
            fields.append(full_key)
            if isinstance(value, dict):
                extract_fields(value, f"{full_key}.")
extract_fields(sample_doc)

# Count unique values
unique_counts = {}
for field in fields:
    pipeline = [
        {"$sample": {"size": record_limit}},
        {"$group": {"_id": f"${field}"}},
        {"$count": "count"}
    ]
    result = list(collection.aggregate(pipeline))
    unique_counts[field] = result[0]['count'] if result else 0

# Sort and print in descending order
for field, count in sorted(unique_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"{field}: {count} unique values")
```

<p></p>

```python
# VisitDateTime: 99996 unique values
# webClientID: 77297 unique values
# user: 13364 unique values
# user.City: 7206 unique values
# ProductID: 2856 unique values
# user.Country: 175 unique values
# device: 66 unique values
# device.Browser: 40 unique values
# device.OS: 13 unique values
# Activity: 2 unique values
```



# CRUD Operations

## Remove and Create

### Remove the Last Record

Remove the last record from <code>clicks</code> and re-insert it.

```python
# collection = db['clicks']
```

<p></p>

```python
last_record = collection.find_one(sort=[("_id", -1)])
collection.delete_one({"_id": last_record["_id"]})
```

<p></p>

```python
# DeleteResult({'n': 1, 'ok': 1.0}, acknowledged=True)
```

### Insert it Back In

```python
collection.insert_one(last_record)
```

<p></p>

```python
# InsertOneResult(ObjectId('60df129dad74d9467ceebd51'), acknowledged=True)
```


### Remove the Last 5 Records

```python
last_five_records = list(collection.find().sort("_id", -1).limit(5))
ids_to_delete = [record["_id"] for record in last_five_records]
collection.delete_many({"_id": {"$in": ids_to_delete}})
```

<p></p>

```python
# DeleteResult({'n': 5, 'ok': 1.0}, acknowledged=True)
```

<p></p>

### Insert Them Back In

```python
collection.insert_many(last_five_records)
```

<p></p>

```python
# InsertManyResult([ObjectId('60df129dad74d9467ceebd51'), ObjectId('60df129dad74d9467ceebd50'), ObjectId('60df129dad74d9467ceebd4f'), ObjectId('60df129dad74d9467ceebd4e'), ObjectId('60df129dad74d9467ceebd4d')], acknowledged=True)
```


## Read

### Filter to `_id` Equal to `60df129dad74d9467ceebd51`


```python
record = collection.find_one({"_id": ObjectId("60df129dad74d9467ceebd51")})
print("Record with _id 60df129dad74d9467ceebd51:")
pp.pprint(record)
```

<p></p>

```python
# Record with _id 60df129dad74d9467ceebd51:
# {'Activity': 'click',
#  'ProductID': 'Pr101251',
#  'VisitDateTime': datetime.datetime(2018, 5, 26, 11, 51, 44, 263000),
#  '_id': ObjectId('60df129dad74d9467ceebd51'),
#  'device': {'Browser': 'Chrome', 'OS': 'Windows'},
#  'user': {'City': 'Vijayawada', 'Country': 'India'},
#  'webClientID': 'WI100000118333'}
 ```
 

### Find First Record Where <code>device.Browser</code> is not Firefox
























