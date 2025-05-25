---
layout: post
title:  "MongoDB via Shell and PyMongo"
date:   2025-05-23 00:00:00 +0000
categories: MongoDB Bash Python
---

Intro text...

Image?


# Outline
- ...
- ...


# Introduction

## The Advantages of Unstructured Data


## MongoDB



# Installation

- Linux
- Windows
- Conda (all systems)
- Mongo Tools (mongosrestore, mongodump, etc.)



# Import Common Libraries

```python
from pymongo import MongoClient
import os
import subprocess 
import json
import pprint as pp
```


# Connect and List Databases

```python
HOST = "localhost"
PORT = "27017"
```

<p></p>

Get Mongo client connection.

```python
conn_str = f"mongodb://{HOST}:{PORT}/"

print("Connection string:")
print(conn_str)

print("\n Databases")
client = MongoClient(conn_str)
print(client.list_database_names())
```

<p></p>

```python
# Connection string:
# mongodb://localhost:27017/

#  Databases
# ['admin', 'config', 'local']
```

<p></p>

```python
DBNAME = "clickstream"
db = client[DBNAME]
```


# Drop <code>clickstream</code> if DB Currently Exists (Optional)

### Using Bash

```bash
!mongosh --quiet --eval "db.getSiblingDB('{DBNAME}').dropDatabase()"
!mongosh --quiet --eval "db.adminCommand('listDatabases').databases.forEach(db => print(db.name))"
```

<p></p>

```bash
# { ok: 1, dropped: 'clickstream' }
# admin
# config
# local
```


### Using Python 

```python
client.drop_database("clickstream")
print(client.list_database_names())
```

<p></p>

```python
# ['admin', 'config', 'local']
```



# Import Data (If Not Present)

### Set Variables

```python
IMPORT_FILE_FOLDER = "C:\\Users\\patwh\\Downloads"
BSON_FILE_NAME = "clicks"
JSON_FILE_NAME = "clicks.metadata"

bson_file = f"{IMPORT_FILE_FOLDER}/{BSON_FILE_NAME}.bson" 
json_file = f"{IMPORT_FILE_FOLDER}/{JSON_FILE_NAME}.json" 

COLLECTION_BSON = BSON_FILE_NAME
COLLECTION_JSON = JSON_FILE_NAME
```


### Using Bash Only

```bash
!mongorestore --host localhost:{PORT} --db {DBNAME} --collection {COLLECTION_BSON} --drop "{bson_file}"
!mongoimport --host localhost:{PORT} --db {DBNAME} --collection {CELLECTION_JSON} --drop --type json "{json_file}"
```

```bash
# 2025-05-25T13:10:22.270-0600    finished restoring clickstream.clicks (6100000 documents, 0 failures)
# 2025-05-25T13:10:22.270-0600    no indexes to restore for collection clickstream.clicks
# 2025-05-25T13:10:22.270-0600    6100000 document(s) restored successfully. 0 document(s) failed to restore.
# 2025-05-25T13:10:22.954-0600    connected to: mongodb://localhost:27017/
# 2025-05-25T13:10:22.955-0600    dropping: clickstream.clicks.metadata
# 2025-05-25T13:10:22.987-0600    1 document(s) imported successfully. 0 document(s) failed to import.
```


### Using Python's <code>subprocess</code>

```python
# Import bson file
subprocess.run(f'mongorestore --host {HOST}:{PORT} --db {DBNAME} --collection {COLLECTION_BSON} --drop "{bson_file}"')

# Import json file
with open(json_file) as f:
    data = json.load(f)
    db[COLLECTION_JSON].insert_many(data) if isinstance(data, list) else db[COLLECTION_JSON].insert_one(data)

# Print document counts
print(f"Records in {COLLECTION_BSON}: {db[COLLECTION_BSON].count_documents({})}")
print(f"Records in {COLLECTION_JSON}: {db[COLLECTION_JSON].count_documents({})}")
```

<p></p>

```python
# Records in clicks: 6100000
# Records in clicks.metadata: 1
```

As we'll see shortly, the string passed to <code>subprocess</code> can also be passed in the form of a <code>.bat</code> file.



# List Databases


### Using Bash

```bash
!mongosh --host {HOST} --port {PORT} --quiet --eval "db.getMongo().getDBs().databases.map(db => db.name)"
```

<p></p>

```bash 
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



# List Collections of <code>clickstream</code> Database


### Using Bash with Environment Variables

While the below demonstrates that we can select the <code>clickstream</code> and <code>list</code> out its collections, these commands executed by the <code>.bat</code> field will not persist; i.e., we will need to select the database again when a separate command is issued later.

Using an external file with environment variables (which appear like <code>%DBNAME%</code>):


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
# ['clicks.metadata', 'clicks']
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
# Created list_collections_py_vars.bat
```

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
# ['clicks.metadata', 'clicks']
```

Not using an external file, with Python variables:

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

Going forward, we'll use the environment variables approach, since it is more descriptive.


### Using Python

```python
db = client["clickstream"]
db.list_collection_names()
```

<p></p>

```python
# ['clicks.metadata', 'clicks']
```


# View Indexes

```python
print(list(db['clicks'].index_information()))
```

<p></p>

```python
# ['_id_']
```

<p></p>

```python
db['clicks.metadata'].find_one()
```

<p></p>

```python
# {'_id': ObjectId('68336b1e80286365bea819e3'),
#  'indexes': [{'v': 2, 'key': {'_id': 1}, 'name': '_id_'}],
#  'uuid': 'ee6da5fe5bdf42b2bc3cecee40723af6',
#  'collectionName': 'clicks'}
```

<p>Only found the <code>_id</code> index, so no need to run a script... but here is what we would use if there was an additional index:</p>

```python
# with open('clicks.metadata.json') as file:
#     metadata = json.load(file)
# if not isinstance(metadata, list):
#     metadata = [metadata]
# for meta_doc in metadata:
#     if isinstance(meta_doc, str):
#         meta_doc = json.loads(meta_doc)
#     for index in meta_doc.get('indexes', []):
#         if index.get('name') != '_id_':
#             key = [(k, v.get('$numberInt', v) if isinstance(v, dict) else v) for k, v in index.get('key', {}).items()]
#             index_options = {k: v for k, v in index.items() if k not in ['v', 'key', 'name']}
#             index_options['name'] = index.get('name', 'unnamed_index')
#             clicks_collection.create_index(key, **index_options)
```



# Sample Document


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

<p></p>

```python
# collection: clicks.metadata
# {'_id': '68336b1e80286365bea819e3',
#  'collectionName': 'clicks',
#  'indexes': [{'key': {'_id': 1}, 'name': '_id_', 'v': 2}],
#  'uuid': 'ee6da5fe5bdf42b2bc3cecee40723af6'}

# collection: clicks
# {'Activity': 'click',
#  'ProductID': 'Pr100037',
#  'VisitDateTime': '2018-05-25T04:51:14.179Z',
#  '_id': '60df1029ad74d9467c91a932',
#  'device': {'Browser': 'Firefox', 'OS': 'Windows'},
#  'user': {'City': 'Colombo', 'Country': 'Sri Lanka'},
#  'webClientID': 'WI100000244987'}
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
# collection: clicks.metadata
# {'_id': ObjectId('68336b1e80286365bea819e3'),
#  'collectionName': 'clicks',
#  'indexes': [{'key': {'_id': 1}, 'name': '_id_', 'v': 2}],
#  'uuid': 'ee6da5fe5bdf42b2bc3cecee40723af6'}

# collection: clicks
# {'Activity': 'click',
#  'ProductID': 'Pr100037',
#  'VisitDateTime': datetime.datetime(2018, 5, 25, 4, 51, 14, 179000),
#  '_id': ObjectId('60df1029ad74d9467c91a932'),
#  'device': {'Browser': 'Firefox', 'OS': 'Windows'},
#  'user': {'City': 'Colombo', 'Country': 'Sri Lanka'},
#  'webClientID': 'WI100000244987'}
```



# Record Counts

## Get Count of Records in Each Collection


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
Created count_records_in_collections.bat
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
# {'clicks.metadata': 1, 'clicks': 6100000}
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
# clicks.metadata: 1 documents
# clicks: 6100000 documents
```

