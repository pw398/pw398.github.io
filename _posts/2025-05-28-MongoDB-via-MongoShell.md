---
layout: post
title:  "MongoDB via Mongo Shell"
date:   2025-05-28 00:00:00 +0000
categories: MongoDB Bash Python
---


This is the first of 3 articles on MongoDB and the power of unstructured databases. The focus is on using the Mongo shell, though parallel resources linked to within utilize command-line (Bash) and PyMongo commands.



# Outline

1. ...
2. ...
3. Installation
    - Windows
    - Ubuntu Linux
    - Mac?
    - MongoDB Tools
    - PyMongo
4. The Mongo Shell
    - Capabilities and Limitations
    - Getting Started
5. Database Operations
    - Drop 
    - Import From File
        - .bson and .json
        - .csv
6. Data Analysis
    - Dataset
    - View Collections
    - Sample Records
    - Record Counts
    - Distinct Fields
    - Distinct Values by Field
7. CRUD Operations
    - Remove and Create
    - Read
    - Update
8. Indexes
    - View Indexes
    - Create Indexes



Unstructured databases have greatly increased in popularity over the past 15 years, addressing the need to manage increasingly large, diverse, and evolving datasets. The lack of adherence to a rigid schema allows data to be stored in variable formats, rather than pre-specified columns, and this flexibility is well-suited toward modern data sources such as multimedia with metadata, text, and embedded or hierarchical data. Normalization and joins are avoided, lending toward the ability to horizontally scale compute resources, and this is heavily relied upon by organizations who deal with massive amounts of web transactions.

MongoDB is the most popular of the unstructured databases, with a large developer community, integration with a multitude of APIs, and a cloud service called MongoDB Atlas. The native language for command-line instructions is Javascript, however the Mongo shell provides its own simplified language. The 'documents', a term analogous to a record in a structured database, are in a JSON-like format, and are organized into 'collections', the analog to a table.

In this article, we will focus on making commands through the Mongo shell, which is the simplest method. However, parallel notebooks utilizing the command line (Bash) and Python (PyMongo) are linked to below.
- **link1** (the mongo shell workbook)
- **link2**
- **link3**

Subsequent articles will focus on PyMongo. The content of this article will provide an overview of querying and database operations, the second article will focus on aggregation pipelines, and the third will focus on deploying machine learning upon streaming text data.



# Installation -> (link to instructions)

The MongoDB website has robust tutorials for installation. Be sure to get the Mongo shell and add it to PATH so you can follow along with the below. Although the code is provided in a notebook format, the commands in this article and the first 'notebook' will only work through the Mongo shell. This can be opened directly (by clicking on the .exe file), or from the command prompt using <code>mongosh</code>.

- <a href="https://www.mongodb.com/docs/manual/installation/">MongoDB Installation Tutorials</a>

- <a href="https://www.mongodb.com/try/download/shell">MongoDB Shell</a>

Also, regardless of which language or platform you plan on using, be sure to get the MongoDB command line tools, as this will be essential toward actions like reading and writing to file.

- <a href="https://www.mongodb.com/try/download/database-tools">MongoDB Command Line Tools</a>

For PyMongo, if you are using Anaconda, I recommend using <code>conda install pymongo</code> from the Anaconda command prompt (<code>conda activate base</code> to activate it from the general command prompt). Otherwise, review the instructions <code>here</code>.



# The Mongo Shell

The capabilities of the Mongo shell include performing CRUD (create, read, update, delete) operations, querying and index management, user and database administration, and Javascript support. The commands are simpler and less verbose than calling upon MongoDB through the APIs, however the Mongo shell is not optimized for large-scale data processing, so APIs like PyMongo will perform better in this regard.

Multi-line commands can be entered by pressing Enter to move to the next line, and then ctrl+Enter when ready to execute. This can be cumbersome for complex commands, though you can use <code>load()</code> to execute from a Javascript file.



# Getting Started

## Opening the MongoDB Shell (mongosh)

### From the Command Line

Once installed, we can open the Mongo shell simply by typing <code>mongosh</code> (or the appropriate environment variable name) into the command prompt. 

```bash
mongosh
```
This will default to the localhost server. We can specify an alternative upon opening by using the command:

```bash
mongosh --host <hostname> --port <port>
```


### Opening Directly

If opening the Mongo shell by clicking on the .exe. file, it will prompt for a server, suggesting <code>localhost</code> by default.

```bash
# Please enter a MongoDB connection string (Default: mongodb://localhost/):</code>
```

I will go with <code>localhost</code>. We can type in the string it suggested, or simply press Enter.

```js
mongodb://localhost/
```


# Show Databases

We can use <code>show dbs</code> or <code>show databases</code> to get the list of currently existing databases. The three listed below are system-related databases which came with the installation.

```js
show dbs
```

<p></p>

```js
// admin        132.00 KiB
// config       116.00 KiB
// local         96.00 KiB
```



# Import Data 




We will be importing clickstream data from a .bson file with the data records, along with a .json file with a single record of metadata.

You may be interested in how to drop this clickstream database, or another that is currently in existence. For that, we simply select the database using <code>use clickstream</code>, and then apply the command <code>db.dropDatabase()</code>.


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

We will also use the <code>--drop</code> option in our import commands, which will drop any database we are trying to import, prior to the import operation. Although the code in this workbook is almost entirely focused on making commands through the Mongo shell, the Mongo tools such as <code>mongorestore</code> and <code>mongoimport</code> cannot be used through the Mongo shell, and must be called upon from the command line.

The syntax for these commands is evident from the last two lines below. You could use hard-coding and ignore all of the variable-setting done by the preceding lines. The below creates a .bat file which can be run from the command line to import our .bson data file (using <code>mongorestore</code>, and .json metadata (using <code>mongoimport</code>).


```bat
# import_data.bat:
SET HOST=localhost
SET PORT=27017
SET DBNAME=clickstream
SET IMPORT_FILE_FOLDER=C:\Users\patwh\Downloads
SET BSON_FILE_NAME=clicks
SET JSON_FILE_NAME=clicks.metadata
SET BSON_FILE=%IMPORT_FILE_FOLDER%\%BSON_FILE_NAME%.bson
SET JSON_FILE=%IMPORT_FILE_FOLDER%\%JSON_FILE_NAME%.json
SET COLLECTION_BSON=%BSON_FILE_NAME%
SET COLLECTION_JSON=%JSON_FILE_NAME%

mongorestore --host %HOST%:%PORT% --db %DBNAME% --collection %COLLECTION_BSON% --drop "%BSON_FILE%"
mongoimport --host %HOST%:%PORT% --db %DBNAME% --collection %COLLECTION_JSON% --drop --type json "%JSON_FILE%"
```

With the .bat file created, simply call upon it from the command line, replacing my directory below with your own.

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

We see that the data file contains 6.1M records, and the metadata file contains only one record. The first step toward viewing the details is to select a database. First, we use <code>show dbs</code> to confirm that our imported data exists, under the name <code>clickstream</code>.


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

To select it, we simply use:

```js
use clickstream
```

<p></p>

```js
// switched to db clickstream
```


# Show Collections

Rather than SQL, where we refer to a database as containing tables of fields and records, we say that a MongoDB database contains 'collections' of documents, each of which has a set of fields. Due to the unstructured nature of the data, it is less common to have collections that require joins to others, as you would typically see with SQL.

Below, we list the collections belonging to <code>clickstream</code>.


```js
show collections
```

<p></p>

```js
// clicks
// clicks.metadata
```



# Sample Documents

To view the first document found in a collection, we use <code>db.collection.findOne()</code>. Note that the fields we see in this sample are not necessarily representative of the fields contained by other records in the collection. 

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

<p>The <code>_id</code>code> field is a unique identifier attached to each record. Duplicate IDs are not permitted, and nor is deleting the field.

We'll take a look at the lone record in the <code>clicks.metadata</code> collection as well.

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

We see that this contains some index information. Indexing will be covered in more detail at the end of this article. 



# Get Record Counts

We already noticed the number of records in this database upon import, but if we didn't, you would use something like this to assess the count of documents.

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

Below, we'll use a command that is a little more involved than our prior commands. I would like to loop through the documents and determine the list of distinct fields contained by documents in the collection. Javascript is the native language of the Mongo shell, so if we want to create variables and loops, we must either rely on Javascript entered directly into the shell, .js files, or an API like PyMongo. PyMongo would actually execute the below in quicker fashion than the Mongo shell, which is why I'll limit the search to the first 1M records (and focus on PyMongo in the upcoming articles).


**why is pymongo faster?**

To run a multi-line command like the following, using the Mongo shell, we can enter each line one at a time and press Enter, and then finally press ctrl+Enter when ready to execute the accumulation of lines. Alternatively, as you will see momentarily, we could put the Javascript into a .js file, and use <code>load(\<file\>)</code> with the Mongo shell to execute.


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


Great - but working with unstructured data means we could have second-level fields nested within the above. The following script will pull us a unique list include nested fields. It's too lengthy to enter into the shell one line at a time, so we'll save the Javascript to a .js file. 

Though we do not need to create the .js file using the method below, it is convenient to define a Python function which will take in some plain text, clean it if necessary, and output a .js file with the specified directory and filename.


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

This next piece of Python code specifies the Javascript as text, and calls upon the above function to create the .js file.

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

The below is what we type into the Mongo shell to execute the .js script. Replace my directory with the directory pertaining to yourself.

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

It would be informative to know how many distinct values correspond to each of the fields in the collection. 

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

We see some fields we didn't see with the sample document, such as <code>user.UserID</code>.These correspond to users of the Kirana store dataset who have signed up to create an account. 



### Dynamic Version (First Finds Fields, then Distinct Value Counts)


Finding the unique list of fields and then hard-coding them into the search for distinct values may have saved us some time - or at least, it broke a very long task (given the 6.1M documents) into two shorter tasks. But the code to perform both actions in dynamic fashion, without hard-coding, is provided below.

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

Fundamental database operations include creating new records, removing records, updating records, and deleting records - hence the acronym CRUD. The below will demonstrate some examples of each.


## Remove and Create

The task below will be to remove a record, or multiple records, capture the data in a Javascript variable, and then perform an insertion operation to put them back into the collection.


### Remove and Re-Insert the Last Record

To capture the data of the <code>clicks</code>-collection record that we will momentarily delete, we use the <code>find()</code> operation combined with a <code>sort</code>:

```js
// capture data in a javascript variable
var lastDoc = db.clicks.find().sort({ _id: -1 }).limit(1).next();
````

And the data is then presented in JSON format.

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

Next, we use <code>deleteOne()</code> to remove it from the collection.

```js
// remove the record from the collection
db.clicks.deleteOne({ _id: lastDoc._id });
```

<p></p>

```js
// { acknowledged: true, deletedCount: 1 }
````

Finally, we use <code>insertOne()</code> with reference to our stored variable to re-insert the record. If the data were not in JSON format, we would need to transform it to such.

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

Below, same drill as above, but for multiple records at the same time. We capture the data of the last 5 records:

```js
// capture data in a javascript variable
var lastDocs = db.clicks.find().sort({ _id: -1 }).limit(5).toArray();
var idsToDelete = lastDocs.map(doc => doc._id);
```

Then, apply a delete operation to remove them from the collection:

```js
// remove the records from the collection
db.clicks.deleteMany({ _id: { $in: idsToDelete } });
```

<p></p>

```js
// { acknowledged: true, deletedCount: 5 }
```

And finally, use <code>insertMany()</code> to insert them back in, all at the same time.

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

We've done a healthy amount of 'read' operations already, but the below will provide some more examples of how to query for the data you are looking for. First, we'll filter to a particular field, in this case the record ID. No documents have duplicate IDs, so the below <code>findOne()</code> will either return one document or none.


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

<p>If multiple records meet the specified criteria of a <code>findOne()</code> query, the first record encountered will be returned. Below, we simply replace the above criteria of having a particular <code>_id</code> code with having a browser of Firefox.</p>

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

If wanting to return more than one document, we use <code>find()</code> rather than <code>findOne()</code>. We can use <code>limit</code> as done below in order to truncate the data returned to a certain number of records - in this case, the first two records where the browser is not equal to Firefox, using the <code>$ne</code> (not equal) operator.

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

Of course, we also have comparison operators such as <code>$gt</code>, used below to get the first two records which have a date later than May 20.

```js
db.clicks.find({
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

To get the minimum and maximum values of a field that spans a numerical or date-based range, we can use something like the following. Aggregation will be covered further in the following articles.


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

We can use <code>countDocuments()</code> as we did to get the count of records in a collection, but apply a filter such as the one we used just above. Below, we see that about 2.45M records have a date greater than May 20.

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
// device.OS_1
```




















