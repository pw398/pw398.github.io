---
layout: post
title:  "MongoDB via Mongo Shell"
date:   2025-05-28 00:00:00 +0000
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



# Opening the Mongo Shell

### From the Command Line

This will default to the localhost server.

```cmd
mongosh
```

### Opening it Directly

This will prompt for a server:

<code>Please enter a MongoDB connection string (Default: mongodb://localhost/):</code>

```js
mongodb://localhost/
```


# Show Databases

```js
show dbs
```

<p></p>

```js
// admin        132.00 KiB
// config       116.00 KiB
// local         96.00 KiB
```

<code>show databases</code> would work equally well.



# Import Data 

### Drop clickstream if Exists (Optional)

```js
use clickstream
```

<p></p>

```js
// switched to db clickstream
```

<p></p>

```js
db.dropDatabase()
```

<p></p>

```js
// { ok: 1, dropped: 'clickstream' }
```


<code>mongorestore</code> and <code>mongoimport</code> must be run from the command line, not the mongo shell.

```python
# import_data.bat:

HOST = "localhost"
PORT = 27017

DBNAME = "clickstream"

IMPORT_FILE_FOLDER = r"C:\Users\patwh\Downloads"
BSON_FILE_NAME = "clicks"
JSON_FILE_NAME = "clicks.metadata"

bson_file = f"{IMPORT_FILE_FOLDER}/{BSON_FILE_NAME}.bson" 
json_file = f"{IMPORT_FILE_FOLDER}/{JSON_FILE_NAME}.json" 

COLLECTION_BSON = BSON_FILE_NAME
COLLECTION_JSON = JSON_FILE_NAME

!mongorestore --host {HOST}:{PORT} --db {DBNAME} --collection {COLLECTION_BSON} --drop "{bson_file}"
!mongoimport --host {HOST}:{PORT} --db {DBNAME} --collection {COLLECTION_JSON} --drop --type json "{json_file}"
```

<p></p>

```python
C:\Users\patwh\Downloads\import_data.bat
```

<p></p>

```python 
# 2025-05-28T18:38:13.827-0600  finished restoring clickstream.clicks (6100000 documents, 0 failures)
# 2025-05-28T18:38:13.827-0600  no indexes to restore for collection clickstream.clicks
# 2025-05-28T18:38:13.827-0600  6100000 document(s) restored successfully. 0 document(s) failed to restore.
# 2025-05-28T18:38:14.488-0600  connected to: mongodb://localhost:27017/
# 2025-05-28T18:38:14.490-0600  dropping: clickstream.clicks.metadata
# 2025-05-28T18:38:14.507-0600  1 document(s) imported successfully. 0 document(s) failed to import.
```


# Select Imported DB

```js
show dbs
```

<p></p>

```js
// admin        132.00 KiB
// clickstream  428.30 MiB
// config       108.00 KiB
// local         96.00 KiB
```

<p></p>

```js
use clickstream
```

<p></p>

```js
// switched to db clickstream
```


# Show Collections

```js
show collections
```

<p></p>

```js
// clicks
// clicks.metadata
```



# Sample Documents

```js
db.clicks.findOne()
```

<p></p>

```js
// {
//   _id: ObjectId('60df1029ad74d9467c91a932'),
//   webClientID: 'WI100000244987',
//   VisitDateTime: ISODate('2018-05-25T04:51:14.179Z'),
//   ProductID: 'Pr100037',
//   Activity: 'click',
//   device: { Browser: 'Firefox', OS: 'Windows' },
//   user: { City: 'Colombo', Country: 'Sri Lanka' }
// }
```

<p></p>

```js
db.clicks.metadata.findOne()
```

<p></p>

```js
// {
//   _id: ObjectId('6837ada071d28360c34516c3'),
//   indexes: [ { v: 2, key: { _id: 1 }, name: '_id_' } ],
//   uuid: 'ee6da5fe5bdf42b2bc3cecee40723af6',
//   collectionName: 'clicks'
// }
````



# Get Record Counts

```js
db.clicks.countDocuments()
````

<p></p>

```js
// 6100000
```

<p></p>

```js
db.clicks.metadata.countDocuments()
```

<p></p>

```js
// 1
```


### Count of Unique Values (Field <code>device.Browser</code>)
















