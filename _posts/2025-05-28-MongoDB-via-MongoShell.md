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

Below is what it would look like if executed from a Jupyter notebook. The variable-setting is optional, but serves to let the restore/import commands to act dynamically.

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
!"C:/Users/patwh/Downloads/import_data.bat"
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


# Get List of Distinct Fields

Limiting to 1M records (it's faster with pymongo)

```js
(function() {
  var fields = {};
  db.clicks.find().limit(1000000).forEach(function(doc) {
    Object.keys(doc).forEach(function(key) {
      fields[key] = true;
    });
  });
  printjson(Object.keys(fields));
})();
```

<p></p>

```js
// [
//   '_id',
//   'webClientID',
//   'VisitDateTime',
//   'ProductID',
//   'Activity',
//   'device',
//   'user'
// ]
````

Including nested fields:

```js
// Function for Creating .js Files From Text in Python
import os

def save_js_commands(js_input, js_folder, js_filename):
    filepath = f"{js_folder}/{js_filename}.js"
    
    try:
        # Ensure input is a string and strip leading/trailing whitespace
        if isinstance(js_input, str):
            lines = [line.rstrip() for line in js_input.splitlines() if line.strip()]
        elif isinstance(js_input, list):
            lines = [line.rstrip() for line in js_input if line.strip()]
        else:
            raise TypeError("Input must be a string or list of strings.")

        # Write to file with explicit UTF-8 encoding and Windows line endings
        with open(filepath, 'w', encoding='utf-8', newline='\r\n') as file:
            for line in lines:
                file.write(line + '\n')

        print(f"✅ JavaScript code saved successfully to: {filepath}")
        return filepath
    except Exception as e:
        print(f"❌ Error saving file: {e}")
        return None
```

<p></p>

```python
# unique_fields_nested.js

js_code = """
(function() {
  var fields = {};

  function extractFields(obj, prefix) {
    for (var key in obj) {
      if (!obj.hasOwnProperty(key)) continue;

      var fullKey = prefix ? prefix + "." + key : key;
      fields[fullKey] = true;

      // Recurse into nested objects (not arrays or null)
      if (typeof obj[key] === "object" && obj[key] !== null && !Array.isArray(obj[key])) {
        extractFields(obj[key], fullKey);
      }
    }
  }

  db.clicks.find().limit(1000000).forEach(function(doc) {
    extractFields(doc, "");
  });

  printjson(Object.keys(fields));
})();

"""

js_folder = r"C:/Users/patwh/Downloads/js_commands"
js_filename = "unique_fields_nested"

js_file = save_js_commands(js_code, js_folder, js_filename)
exe_file = f"{js_folder}\\{js_filename}.js"
```

<p></p>

```python
# ✅ JavaScript code saved successfully to: C:/Users/patwh/Downloads/js_commands/unique_fields_nested.js
```

<p></p>

```js
load("C:/Users/patwh/Downloads/js_commands/unique_fields_nested.js")
````

<p></p>

```js
// [
//   '_id',
//   'webClientID',
//   'VisitDateTime',
//   'ProductID',
//   'Activity',
//   'device',
//   'device.Browser',
//   'device.OS',
//   'user',
//   'user.City',
//   'user.Country',
//   'user.UserID'
// ]
```



# Get Number of Distinct Values by Field

Hard-coding the file names above (for speed):

```python
js_code = """
(function() {
  const collection = db.clicks;

  const fields = [
    '_id',
    'webClientID',
    'VisitDateTime',
    'ProductID',
    'Activity',
    'device',
    'device.Browser',
    'device.OS',
    'user',
    'user.City',
    'user.Country',
    'user.UserID'
  ];

  fields.forEach(field => {
    const pipeline = [
      { $group: { _id: `$${field}` } },
      { $group: { _id: null, count: { $sum: 1 } } }
    ];

    const result = collection.aggregate(pipeline).toArray();
    const count = result.length > 0 ? result[0].count : 0;
    print(`${field}: ${count} unique values`);
  });
})();

"""

js_folder = r"C:/Users/patwh/Downloads/js_commands"
js_filename = "unique_value_counts_hardcoded_fields"

js_file = save_js_commands(js_code, js_folder, js_filename)
exe_file = f"{js_folder}\\{js_filename}.js"
```

<p></p>

```python
# ✅ JavaScript code saved successfully to: C:/Users/patwh/Downloads/js_commands/unique_value_counts_hardcoded_fields.js
```

<p></p>

```js
load("C:/Users/patwh/Downloads/js_commands/unique_value_counts_hardcoded_fields.js")
```

<p></p>

```js
// _id: 6100000 unique values
// webClientID: 1091455 unique values
// VisitDateTime: 6089023 unique values
// ProductID: 10938 unique values
// Activity: 2 unique values
// device: 151 unique values
// device.Browser: 82 unique values
// device.OS: 18 unique values
// user: 72162 unique values
// user.City: 26260 unique values
// user.Country: 222 unique values
// user.UserID: 34051 unique values
```


### Dynamic Version (First Finds Fields, then Distinct Value Counts)

```python
js_code = """
// Use the correct database
db = db.getSiblingDB("clickstream");

print("=== Starting Unique Field Count ===");

const SAMPLE_SIZE = 1000; // Sample more than 1 doc to detect more fields
const MAX_FIELDS = 100;   // Safety cap
let fieldSet = {};

function extractFields(obj, prefix = "") {
    for (let key in obj) {
        if (!obj.hasOwnProperty(key)) continue;
        const fullKey = prefix ? prefix + "." + key : key;
        fieldSet[fullKey] = true;
        if (typeof obj[key] === "object" && obj[key] !== null && !Array.isArray(obj[key])) {
            extractFields(obj[key], fullKey);
        }
    }
}

// Sample documents to build a broader field list
db.clicks.find().limit(SAMPLE_SIZE).forEach(doc => {
    extractFields(doc);
});

const fields = Object.keys(fieldSet).slice(0, MAX_FIELDS);
let results = [];

fields.forEach(field => {
    try {
        const pipeline = [
            { $group: { _id: `$${field}` } },
            { $group: { _id: null, count: { $sum: 1 } } }
        ];

        const res = db.clicks.aggregate(pipeline).toArray();
        const count = res.length ? res[0].count : 0;
        results.push({ field, count });
    } catch (e) {
        print(`Error counting values for ${field}: ${e.message}`);
    }
});

// Sort by count descending
results.sort((a, b) => b.count - a.count);

// Print nicely
results.forEach((r, i) => {
    print(`${i + 1}. ${r.field}: ${r.count} unique values`);
});

print("=== Done ===");
"""

js_folder = r"C:/Users/patwh/Downloads/js_commands"
js_filename = "count_unique_values_dynamic"

js_file = save_js_commands(js_code, js_folder, js_filename)
exe_file = f"{js_folder}\\{js_filename}.js"
```

<p></p>

```python
# ✅ JavaScript code saved successfully to: C:/Users/patwh/Downloads/js_commands/count_unique_values_dynamic.js
```

<p></p>

```js
load("C:/Users/patwh/Downloads/js_commands/count_unique_values_dynamic.js")
````

<p></p>

```js
// === Starting Unique Field Count Across Entire Collection ===
// 1. _id: 6100000 unique values
// 2. VisitDateTime: 6089023 unique values
// 3. webClientID: 1091455 unique values
// 4. user: 72162 unique values
// 5. user.UserID: 34051 unique values
// 6. user.City: 26260 unique values
// 7. ProductID: 10938 unique values
// 8. user.Country: 222 unique values
// 9. device: 151 unique values
// 10. device.Browser: 82 unique values
// 11. device.OS: 18 unique values
// 12. Activity: 2 unique values
// 13. _id.serverVersions: 1 unique values
// 14. _id.platforms: 1 unique values
// 15. _id.topologies: 1 unique values
// 16. _id.help: 1 unique values
// === Done ===
// true
````



# CRUD Operations










































