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
// === Starting Unique Field Count ===
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
// === Done ===
// true
````



# CRUD Operations

## Remove and Create

### Remove and Re-Insert the Last Record

Remove the last record from <code>clicks</code> and re-insert it.

```js
// capture data in a javascript variable
var lastDoc = db.clicks.find().sort({ _id: -1 }).limit(1).next();
````

<p></p>

```js
// {
//   _id: ObjectId('60df129dad74d9467ceebd51'),
//   webClientID: 'WI100000118333',
//   VisitDateTime: ISODate('2018-05-26T11:51:44.263Z'),
//   ProductID: 'Pr101251',
//   Activity: 'click',
//   device: { Browser: 'Chrome', OS: 'Windows' },
//   user: { City: 'Vijayawada', Country: 'India' }
// }
```

<p></p>

```js
// remove the record from the collection
db.clicks.deleteOne({ _id: lastDoc._id });
```

<p></p>

```js
// { acknowledged: true, deletedCount: 1 }
````

<p></p>

```js
// insert the record back into the collection
db.clicks.insertOne(lastDoc)
```
<p></p>

```js
// {
//   acknowledged: true,
//   insertedId: ObjectId('60df129dad74d9467ceebd51')
// }
````


### Remove and Re-Insert the Last 5 Records

```js
// capture data in a javascript variable
var lastDocs = db.clicks.find().sort({ _id: -1 }).limit(5).toArray();
var idsToDelete = lastDocs.map(doc => doc._id);
```

<p></p>

```js
// remove the records from the collection
db.clicks.deleteOne({ _id: lastDoc._id });
```

<p></p>

```js
// { acknowledged: true, deletedCount: 5 }
```

<p></p>

```js
// insert them back in
db.clicks.insertMany(lastDocs);
````

<p></p>

```js
// {
//   acknowledged: true,
//   insertedIds: {
//     '0': ObjectId('60df129dad74d9467ceebd51'),
//     '1': ObjectId('60df129dad74d9467ceebd50'),
//     '2': ObjectId('60df129dad74d9467ceebd4f'),
//     '3': ObjectId('60df129dad74d9467ceebd4e'),
//     '4': ObjectId('60df129dad74d9467ceebd4d')
//   }
// }
```



## Read

<h3>Filter to <code>_id</code> Equal to <code>60df129dad74d9467ceebd51</code></h3>

```js
db.clicks.findOne({ _id: ObjectId("60df129dad74d9467ceebd51") });
```

<p></p>

```js
// {
//   _id: ObjectId('60df129dad74d9467ceebd51'),
//   webClientID: 'WI100000118333',
//   VisitDateTime: ISODate('2018-05-26T11:51:44.263Z'),
//   ProductID: 'Pr101251',
//   Activity: 'click',
//   device: { Browser: 'Chrome', OS: 'Windows' },
//   user: { City: 'Vijayawada', Country: 'India' }
// }
````


### Find First Record Where <code>device.Browser</code> is not Firefox

```js
db.clicks.findOne({ "device.Browser": "Firefox" });
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
````


### Find First 2 Records Where <code>device.Browser</code> is not Firefox

```js
db.clicks.find({ "device.Browser": { $ne: "Firefox" } }).limit(2);
```

<p></p>

```js
// [
//   {
//     _id: ObjectId('60df1029ad74d9467c91a933'),
//     webClientID: 'WI10000061461',
//     VisitDateTime: ISODate('2018-05-25T05:06:03.700Z'),
//     ProductID: 'Pr100872',
//     Activity: 'pageload',
//     device: { Browser: 'Chrome Mobile', OS: 'Android' },
//     user: {}
//   },
//   {
//     _id: ObjectId('60df1029ad74d9467c91a934'),
//     webClientID: 'WI10000075748',
//     VisitDateTime: ISODate('2018-05-17T11:51:09.265Z'),
//     ProductID: 'Pr100457',
//     Activity: 'click',
//     device: { Browser: 'Chrome', OS: 'Linux' },
//     user: { City: 'Ottawa', Country: 'Canada' }
//   }
// ]
```



### Find First 2 Records Where <code>device.Browser</code> and <code>VisitDateTime</code> > 5/20/2018


```js
db.clicks.find({
  "device.Browser": { $exists: true, $ne: null },
  VisitDateTime: { $gt: new Date("2018-05-20T00:00:00Z") }
}).limit(2);
````

<p></p>

```js
// [
//   {
//     _id: ObjectId('60df1029ad74d9467c91a932'),
//     webClientID: 'WI100000244987',
//     VisitDateTime: ISODate('2018-05-25T04:51:14.179Z'),
//     ProductID: 'Pr100037',
//     Activity: 'click',
//     device: { Browser: 'Firefox', OS: 'Windows' },
//     user: { City: 'Colombo', Country: 'Sri Lanka' }
//   },
//   {
//     _id: ObjectId('60df1029ad74d9467c91a933'),
//     webClientID: 'WI10000061461',
//     VisitDateTime: ISODate('2018-05-25T05:06:03.700Z'),
//     ProductID: 'Pr100872',
//     Activity: 'pageload',
//     device: { Browser: 'Chrome Mobile', OS: 'Android' },
//     user: {}
//   }
// ]
```



### Get the Minimum and Maximum <code>VisitDateTime</code>

```js
db.clicks.aggregate([
  { $group: {
      _id: null,
      minVisitDateTime: { $min: "$VisitDateTime" },
      maxVisitDateTime: { $max: "$VisitDateTime" }
    } }
]);
````

<p></p>

```js
// [
//   {
//     _id: null,
//     minVisitDateTime: ISODate('2018-05-07T00:00:01.190Z'),
//     maxVisitDateTime: ISODate('2018-05-27T23:59:59.576Z')
//   }
// ]
````



### Get Count of Records Where <code>VisitDateTime</code> is Greater Than 5/20/2018

```js
db.clicks.countDocuments({
  VisitDateTime: { $gt: new Date("2018-05-20T00:00:00Z") }
});
````

<p></p>

```js
// 2453050
````


### Get Count of Records Where <code>user.Country</code> is <code>India</code> or <code>United States</code>


#### Using <code>$or</code>

```js
db.clicks.countDocuments({
  $or: [
    { "user.Country": "India" },
    { "user.Country": "United States" }
  ]
});
````

<p></p>

```js
// 3497232
````


#### Using <code>$in</code>

```js
db.clicks.countDocuments({
  "user.Country": { $in: ["India", "United States"] }
});
````

<p></p>

```js
// 3497232
```


### Get Count of Records Where <code>user.Country</code> is Neither <code>India</code> Nor <code>United States</code>

#### Using <code>$and</code>

```js
db.clicks.countDocuments({
  $and: [
    { "user.Country": { $ne: "India" } },
    { "user.Country": { $ne: "United States" } }
  ]
});
```

<p></p>

```js
// 2602768
```


#### Using `$not` and `$in`

```js
db.clicks.countDocuments({
  "user.Country": { $not: { $in: ["India", "United States"] } }
});
```

<p></p>

```js
// 2602768
```


#### Using <code>$nin</code>

```js
db.clicks.countDocuments({
  "user.Country": { $nin: ["India", "United States"] }
});
```

<p></p>

```js
// 2602768
```


### Get Count of Records with <code>user.UserID</code>

```js
db.clicks.countDocuments({
  "user.UserID": { $exists: true, $ne: null }
});
````

<p></p>

```js
// 602293
```


## Update

### Update <code>device.Browser</code> for Record <code>60df129dad74d9467ceebd51</code> to <code>Firefox</code>

```js
db.collectionName.updateOne(
  { _id: ObjectId("60df129dad74d9467ceebd51") },
  { $set: { "device.Browser": "Firefox" } }
);
```

<p></p>

```js
// {
//   acknowledged: true,
//   insertedId: null,
//   matchedCount: 0,
//   modifiedCount: 0,
//   upsertedCount: 0
// }
```

Set it back to original state for accuracy:

```js
db.collectionName.updateOne(
  { _id: ObjectId("60df129dad74d9467ceebd51") },
  { $set: { "device.Browser": "Chrome" } }
);
```

```js
// {
//   acknowledged: true,
//   insertedId: null,
//   matchedCount: 0,
//   modifiedCount: 0,
//   upsertedCount: 0
// }
```


### Update All <code>device.Browser</code> Records to be <code>Firefox</code>

(If we wanted to; I'll leave it commented out)

```js
// db.collectionName.updateMany(
//   {},
//   { $set: { "device.Browser": "Firefox" } }
// );
```


### Create New Field

### Add Field Called <code>NewField</code> to First 1000 Records, Set Value to <code>Default</code>

```js
db.clicks.find().limit(1000).forEach(doc => {
  db.clicks.updateOne(
    { _id: doc._id },
    { $set: { NewField: "Default" } }
  );
});
```

View a record to confirm update:

```js
db.clicks.findOne({ NewField: "Default" });
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
//   user: { City: 'Colombo', Country: 'Sri Lanka' },
//   NewField: 'Default'
// }
```


### Remove the Added Field

```js
db.clicks.updateMany(
  { NewField: { $exists: true } },
  { $unset: { NewField: "" } }
);
```

<p></p>

```js
// {
//   acknowledged: true,
//   insertedId: null,
//   matchedCount: 1000,
//   modifiedCount: 1000,
//   upsertedCount: 0
// }
```



# Indexes

## View Indexes

```js
db.clicks.getIndexes();
```

<p></p>

```js
[ { v: 2, key: { _id: 1 }, name: '_id_' } ]
```

<p></p>

```js
db.clicks.metadata.findOne();
```

<p></p>

```js
// {
//   _id: ObjectId('6837ada071d28360c34516c3'),
//   indexes: [ { v: 2, key: { _id: 1 }, name: '_id_' } ],
//   uuid: 'ee6da5fe5bdf42b2bc3cecee40723af6',
//   collectionName: 'clicks'
}
```


## Create Indexes

```js
db.clicks.createIndex({ "device.OS": 1 });
```

<p></p>

```js
// device.OS_1
```

<p></p>

```js
db.clicks.createIndex({ "device.Browser": 1 });
```

<p></p>

```js
db.clicks.createIndex({ "device.Browser": 1 });
```




















