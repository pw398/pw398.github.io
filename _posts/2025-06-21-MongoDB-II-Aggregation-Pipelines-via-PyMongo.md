---
layout: post
title:  "MongoDB Aggregation Pipelines via PyMongo"
date:   2025-06-21 00:00:00 +0000
categories: MongoDB Python
---

In this article, we'll continue to work with the Kirana Store clickstream data. What started as an intent to include some SQL-analogies became a determination to provide matching aggregation results between MongoDB on unstructured data and SQL upon flattened data.


<img src="https://github.com/pw398/pw398.github.io/blob/main/_posts/images/mg2.png" style="height: 350px; width:auto;">


# Outline

1. ...
2. ...



# Introduction

To briefly recap, the last article included the basics of operating MongoDB through the Mongo shell, Bash, or Python (PyMongo), such as for basic queries and CRUD operations. Because this article is a little more involved, we'll focus on PyMongo, though as I mentioned above, an SQL notebook (using Google Colab) is provided as a companion piece. For myself at least, this makes the PyMongo code a lot more relatable, and I figure that is likely the case for others.

To be honest, it took me quite a bit of troubleshooting to get the results to match (though of course hindsight is 20/20). Flattening the data was relatively easy, as there aren't many layers of nested fields in the clickstream dataset. The acts of matching top-line results, user-level results, and eventually country-level results were a little more of a process. AI assistance (from Grok) was helpful, but there was definitely a trade-off between concise recommendations and high-quality ones. Trickle-charging with step-by-step input in a workspace, and modularization of code using attachments, proved to be a productive approach, though the limits of its capabilities (for now) were still noticeably encountered.



# The Kirana Store <code>clickstream</code> Data

As a reminder, while we refer to structured databases as containing tables full of records containing fields, we refer to unstructured databases as containing collections of documents containing fields. A document from the data we are dealing with looks something like:

```js
{'_id': ObjectId('60df102aad74d9467c94272a'),
 'webClientID': 'WI10000021937',
 'VisitDateTime': datetime.datetime(2018, 5, 23, 14, 27, 15, 118000),
 'ProductID': 'Pr100472',
 'Activity': 'click',
 'device': {'Browser': 'Safari', 'OS': 'Mac OS X'},
 'user': {'UserID': 'U100095', 'Country': 'Turkey'}}
```

which is similar to Javascript Object Notation (JSON), though <code>ObjectId</code> is a MongoDB-specific data type, the <code>datetime</code> data is a Python data type, and JSON would use double-quotes instead of apostrophes.

It's also the case with unstructured data that the fields above may not exist for all records, and other records may contain data (like <code>user.City</code>) which the above does not.

The dataset is available here, in a .zip file <a href="https://drive.google.com/file/d/1ZRrNKa9sBtyRi1jZ5ocFMrcuuY_erVYH/view?usp=drive_link">on Google Drive</a>.

The MongoDB/PyMongo Jupyter notebook is available **here**, and the MySQL companion piece, which uses a Google Colab notebook (for replicability) is located **here**.



# Aggregation Pipelines

We refer to the stage-based framework of aggregation in MongoDB as aggregation pipelines, which analyze and transform data into filtered, aggregated, or calculated results. These stages include operations like:

- $match: ...
- $sum: ...
- $count
- $group
- $project
- $sort
- $limit
- $unwind
- $addFields
- $lookup

**link**



# Import Libraries and Data

## Import Libraries

We'll be using the following libraries. Install using <code>pip</code>, <code>!pip</code> or the conda shell if necessary.


```python
# To connect to a MongoDB instance
from pymongo import MongoClient
# For some types of data manipulation
import pandas as pd
# For nicer document printing
import pprint as pp
# For working with datetime objects
from datetime import datetime
# For timing operations
import time
# For exporting checkpoints
import subprocess
# For interactive plotting
import plotly.graph_objects as go
from plotly.subplots import make_subplots
```


Next, establish a connection to a MongoDB instance, and print the list of databases currently in existence.

## Establish a MongoDB Connection

```python
client = MongoClient("mongodb://localhost:27017/")
print(client.list_database_names())
```

<p></p>

```python
# ['admin', 'config', 'local']
```

In my case, the list currently contains only the system-related data.



## Drop <code>clickstream</code> if Exists (Optional)

If the <code>clickstream</code> database exists from prior tinkering, and you wish to delete it, use the following.

```python
db_name = "clickstream"
client.drop_database(db_name)
print(client.list_database_names())
```


## Import Data

Next, we import the data from the <code>clicks.bson</code> and <code>clicks.metadata.json</code> files. The following cell is about setting variables to be used one cell below, you can skip to the next cell if you wish to use hard-coded parameters.

```python
HOST = "localhost"
PORT = "27017"
DBNAME = "clickstream"
IMPORT_FILE_FOLDER = "data"
BSON_FILE_NAME = "clicks"
JSON_FILE_NAME = "clicks.metadata"
bson_file = f"{IMPORT_FILE_FOLDER}\\{BSON_FILE_NAME}.bson"
json_file = f"{IMPORT_FILE_FOLDER}\\{JSON_FILE_NAME}.json"
collection_bson = BSON_FILE_NAME
collection_json = JSON_FILE_NAME
```

The following shell commands, facilitated in Jupyter notebook via the preceding exclamation mark, will import the data from file, so long as you have the Mongo tools package (referenced in the prior article) installed. The <code>--drop</code> element will clear any existing data of the same database and collection name before import.


```bash
!mongorestore --host {HOST}:{PORT} --db {DBNAME} --collection {collection_bson} --drop "{bson_file}"
!mongoimport --host {HOST}:{PORT} --db {DBNAME} --collection {collection_json} --drop --type json "{json_file}"
```

<p></p>

```bash
# 2025-06-08T13:59:20.871-0600  finished restoring clickstream.clicks (6100000 documents, 0 failures)
# 2025-06-08T13:59:20.872-0600  no indexes to restore for collection clickstream.clicks
# 2025-06-08T13:59:20.873-0600  6100000 document(s) restored successfully. 0 document(s) failed to restore.
# 2025-06-08T13:59:21.819-0600  connected to: mongodb://localhost:27017/
# 2025-06-08T13:59:21.821-0600  dropping: clickstream.clicks.metadata
# 2025-06-08T13:59:21.853-0600  1 document(s) imported successfully. 0 document(s) failed to import.
```


# Select DB and View Collections


With the database created and data imported, we next list out the collection names.


```python
db_name = "clickstream"
db = client[db_name]
collections = db.list_collection_names()
print(collections)
```

<p></p>

```python
# ['clicks.metadata', 'clicks']
```



# Data Exploration

## Sample Record


Next, view a sample record of the data.


```python
collection = db['clicks']
collection.find_one()
```

<p></p>

```python
# {'_id': ObjectId('60df1029ad74d9467c91a932'),
#  'webClientID': 'WI100000244987',
#  'VisitDateTime': datetime.datetime(2018, 5, 25, 4, 51, 14, 179000),
#  'ProductID': 'Pr100037',
#  'Activity': 'click',
#  'device': {'Browser': 'Firefox', 'OS': 'Windows'},
#  'user': {'City': 'Colombo', 'Country': 'Sri Lanka'}}
```


## Count of Records

Though the number of documents is mentioned upon import, we can also use the following command. To apply a filter, we would include something like <code>collection.count_documents({"user.City": "Colombo"})</code>, with either apostrophes or double-quotes being acceptable. In the Mongo shell, we would not have to encapsulate the keys and values this way.


```python
collection.count_documents({})
```

<p></p>

```python
# 6100000
```


## Get Date Range


<p>To get the range of dates in the data, we can use something like the following. Note that this is our first pipeline (of the current article), in which the <code>$group</code> operator is used to aggregate documents. Setting <code>_id</code> to <code>None</code> effectively removes any grouping by specific field values and treats the entire collection as one group. Within this group, we compute the earliest and latest values of the <code>VisitDateTime</code>


```python
pipeline = [
    {
        "$group": {
            "_id": None,
            "minDate": {"$min": "$VisitDateTime"},
            "maxDate": {"$max": "$VisitDateTime"}
        }
    }
]

result = collection.aggregate(pipeline)
for doc in result:
    pp.pprint(doc)
```

<p></p>

```python
# {'_id': None,
#  'maxDate': datetime.datetime(2018, 5, 27, 23, 59, 59, 576000),
#  'minDate': datetime.datetime(2018, 5, 7, 0, 0, 1, 190000)}
```


## Count of Unique <code>webClientID</code> Values


<code>webClientID</code> groups individual visits by user, so long as they connect using the same device and system as prior visits. As we'll see later, the data also include a <code>user.UserID</code> field for those who have created (and logged into) an account, however a minority of documents contain that information, whereas all records contain a <code>webClientID</code>. This introduces the <code>$count</code> operator.


```python
pipeline = [
    { "$group": { "_id": "$webClientID" } },
    { "$count": "uniqueCount" }
]

result = list(collection.aggregate(pipeline))
num_unique = result[0]['uniqueCount'] if result else 0
print(f"Number of unique webClientID values: {num_unique}")
```

<p></p>

```python
# Number of unique webClientID values: 1091455
```

With smaller data (less than 16MB in Jupyter notebook), we could do this in simpler fashion, using:

```python
# collection = db['clicks']
# len(collection.distinct('webClientID'))
```

It is analogous to the following in SQL:

```sql
SELECT COUNT(DISTINCT webClientID)
FROM clicks;
```


## Count of <code>webClientID</code> Values with a <code>user.UserID</code>


Next, we'll see how many instances of <code>webClientID</code> correspond to having a <code>user.UserID</code>. This introduces the <code>$match</code> operator, which lets us filter to only records where the field <code>user.UserID</code> exists, and is not equal (<code>$ne</code>) to being <code>None</code> (i.e., not null).


```python
result = collection.distinct(
    "webClientID",
    {
        "user.UserID": {
            "$exists": True,
            "$ne": None
        }
    }
)

len(result)
```

<p></p>

```python
# 36791
```

Clearly a small proportion. The SQL analog would be as follows.


```sql
SELECT COUNT(DISTINCT webClientID)
FROM clicks
WHERE user_UserID IS NOT NULL;
```



## Count of Unique <code>user.UserID</code> Values


The number of unique user IDs is slightly smaller still.


```python
pipeline = [
    {"$match": {"user.UserID": {"$exists": True, "$ne": None}}},
    {"$group": {"_id": "$user.UserID"}},
    {"$count": "uniqueUserIDs"}
]

result = collection.aggregate(pipeline)
count = next(result, {"uniqueUserIDs": 0})["uniqueUserIDs"]
print(count)
```

<p></p>

```python
# 34050
```

The SQL analog would simple be:

```sql
SELECT COUNT(DISTINCT user_UserID)
FROM clicks;
```



# Classify Device Type as Bot, Desktop, or Mobile

## Distinct Values for <code>device.OS</code>


Now for something a little more interesting. Let's suppose we are interested in understanding our customers' device-related preferences. For this, we can utilize the nested <code>device.OS</code> and <code>device.Browser</code> fields. The operating systems are a good indicator of whether a user is on a mobile or desktop device, and the browsers may offer a clear indication of whether a visitor is a robot. I relied on AI to make the classifications, so forgive any technical inaccuracies, but we will use the logic that a visitor is a robot if the browser indicates so, and that this classification will take precedence over distinctions in operating system. If not a robot, we will classify the visitor as either desktop or mobile based on operating system, and label each document accordingly by adding a field called <code>device_type</code> to our collection.

To get the list of unique operating systems, we can use the <code>distinct</code> keyword as follows. The line below simply provides the output as a horizontal list to save space.

```python
os_list = collection.distinct("device.OS")
print(", ".join(map(str, os_list)))
```

<p></p>

```python
# Android, BlackBerry OS, Chrome OS, Chromecast, Fedora, FreeBSD, Kindle, Linux, Mac OS X, NetBSD, OpenBSD, Other, Solaris, Tizen, Ubuntu, Windows, Windows Phone, iOS
```

The SQL analog would be as follows for a vertical list:

```sql
SELECT COUNT(DISTINCT device_OS) AS Num_Device_OS
FROM clicks
WHERE device_OS IS NOT NULL;
```

Or as follows for a horizontal one:

```sql
SELECT STRING_AGG(DISTINCT device_OS, ', ') AS os_list
FROM clicks
WHERE device_OS IS NOT NULL; 
```


## Distinct Values for <code>device.Browser</code>


Similarly, for the list of unique browsers:


```python
os_list = collection.distinct("device.Browser")
print(", ".join(map(str, os_list)))
```

<p></p>

```python
# AdsBot-Google, AhrefsBot, Amazon Silk, Android, AppEngine-Google, Apple Mail, BingPreview, BlackBerry WebKit, Chrome, Chrome Mobile, Chrome Mobile WebView, Chrome Mobile iOS, Chromium, Coc Coc, Coveobot, Crosswalk, Dragon, DuckDuckBot, Edge, Edge Mobile, Electron, Epiphany, Facebook, FacebookBot, Firefox, Firefox Mobile, Firefox iOS, HbbTV, HeadlessChrome, HubSpot Crawler, IE, IE Mobile, Iceweasel, Iron, JobBot, Jooblebot, K-Meleon, Kindle, Konqueror, Magus Bot, Mail.ru Chromium Browser, Maxthon, Mobile Safari, Mobile Safari UI/WKWebView, MobileIron, NetFront, Netscape, Opera, Opera Coast, Opera Mini, Opera Mobile, Other, PagePeeker, Pale Moon, PetalBot, PhantomJS, Pinterest, Puffin, Python Requests, QQ Browser, QQ Browser Mobile, Radius Compliance Bot, Safari, Samsung Internet, SeaMonkey, Seekport Crawler, SiteScoreBot, Sleipnir, Sogou Explorer, Thunderbird, UC Browser, Vivaldi, WebKit Nightly, WordPress, Yandex Browser, YandexAccessibilityBot, YandexBot, YandexSearch, Yeti, YisouSpider, moatbot, net/bot
```


## Classify as Bot, Desktop, or Mobile 


The categorizations are as follows, in Python list format:


```python
mobile_os = [
    'Android', 'iOS', 'Windows Phone', 'BlackBerry OS', 'Tizen', 'Kindle', 'Chromecast'
]
desktop_os = [
    'Windows', 'Mac OS X', 'Linux', 'Ubuntu', 'Fedora', 'FreeBSD', 
    'NetBSD', 'OpenBSD', 'Solaris', 'Chrome OS', 'Other'
]
bot_browsers = [
    'AdsBot-Google', 'AhrefsBot', 'BingPreview', 'DuckDuckBot', 'FacebookBot',
    'HubSpot Crawler', 'JobBot', 'Jooblebot', 'Magus Bot', 'PetalBot',
    'Radius Compliance Bot', 'Seekport Crawler', 'SiteScoreBot', 'YandexBot',
    'YandexAccessibilityBot', 'YandexSearch', 'Yeti', 'YisouSpider', 'moatbot',
    'net/bot', 'AppEngine-Google', 'PagePeeker', 'Pinterest', 'Facebook',
    'Python Requests', 'Coveobot', 'HeadlessChrome', 'PhantomJS', 'WordPress'
]
```


The classification operation will leverage Python, with a bit of PyMongo, using the <code>$set</code> command for updating:


```python
mobile_os = [x.lower() for x in mobile_os]
desktop_os = [x.lower() for x in desktop_os]
bot_browsers = [x.lower() for x in bot_browsers]

# report progress while updating
record_count = 0
progress_interval = 100000

# start a timer
start_time = time.time()

# loop through records in the collection
for record in collection.find({}, {'device.OS': 1, 'device.Browser': 1}):
    browser = record.get('device', {}).get('Browser', '').lower().strip()
    os = record.get('device', {}).get('OS', '').lower().strip()

    # determine device_type for record
    if browser in bot_browsers:
        device_type = 'bot'
    elif os in mobile_os:
        device_type = 'mobile'
    elif os in desktop_os:
        device_type = 'desktop'
    else:
        device_type = None
    
    # update the record's device_type
    result = collection.update_one(
        {"_id": record['_id']},
        {"$set": {
            "device_type": device_type
        }}
    )
    
    # increment count
    if result.modified_count > 0:
        record_count += 1

    # report progress
    if record_count % progress_interval == 0 and record_count > 0:
        print(f"Processed {record_count} records")

elapsed_time = time.time() - start_time
print(f"Completed: Updated device_type for {record_count} records in {elapsed_time:.0f} seconds")
```

The SQL analog, which I'll write as if we are using SQL magic commands in a Python notebook, is as follows.


```sql
%sql # UPDATE clicks \
SET device_type = \
    CASE \
        WHEN LOWER(TRIM(device_Browser)) IN ( \
            'adsbot-google', 'ahrefsbot', 'bingpreview', 'duckduckbot', 'facebookbot', \
            'hubspot crawler', 'jobbot', 'jooblebot', 'magus bot', 'petalbot', \
            'radius compliance bot', 'seekport crawler', 'sitescorebot', 'yandexbot', \
            'yandexaccessibilitybot', 'yandexsearch', 'yeti', 'yisouspider', 'moatbot', \
            'net/bot', 'appengine-google', 'pagepeeker', 'pinterest', 'facebook', \
            'python requests', 'coveobot', 'headlesschrome', 'phantomjs', 'wordpress' \
        ) THEN 'bot' \
        WHEN LOWER(TRIM(device_OS)) IN ( \
            'android', 'ios', 'windows phone', 'blackberry os', 'tizen', 'kindle', 'chromecast' \
        ) THEN 'mobile' \
        WHEN LOWER(TRIM(device_OS)) IN ( \
            'windows', 'mac os x', 'linux', 'ubuntu', 'fedora', 'freebsd', \
            'netbsd', 'openbsd', 'solaris', 'chrome os', 'other' \
        ) THEN 'desktop' \
        ELSE NULL \
    END;
# Record end time and calculate duration
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.0f} seconds")
```


The above PyMongo code used <code>update_one</code> to make reporting progress easier, but a bulk-write method is also available, and we could even weave in some progress tracking, such as the following (somewhat confusingly, it does leverage a PyMongo function called <code>UpdateOne</code>).


```python
# from pymongo import UpdateOne

# mobile_os = [x.lower() for x in mobile_os]
# desktop_os = [x.lower() for x in desktop_os]
# bot_browsers = [x.lower() for x in bot_browsers]

# batch_size = 500000
# operations = []
# records_processed = 0
# records_written = 0

# # start a timer
# start_time = time.time()

# for record in collection.find({}, {'device.OS': 1, 'device.Browser': 1}):
#     browser = record.get('device', {}).get('Browser', '').lower().strip()
#     os = record.get('device', {}).get('OS', '').lower().strip()

#     if browser in bot_browsers:
#         device_type = 'bot'
#     elif os in mobile_os:
#         device_type = 'mobile'
#     elif os in desktop_os:
#         device_type = 'desktop'
#     else:
#         device_type = None
    
#     operations.append(
#         UpdateOne(
#             {"_id": record['_id']},
#             {"$set": {"device_type": device_type}}
#         )
#     )
    
#     records_processed += 1
#     if records_processed % batch_size == 0:
#         print(f"Read {records_processed} records")

#     if len(operations) >= batch_size:
#         collection.bulk_write(operations)
#         records_written += len(operations)
#         print(f"Written {records_written} records")
#         operations = []

# if operations:
#     collection.bulk_write(operations)
#     records_written += len(operations)
#     print(f"Written {records_written} records")

# elapsed_time = time.time() - start_time
# print(f"Completed: Updated device_type in {elapsed_time:.0f} seconds")
```

<p></p>

```python
# Read 500000 records
# Written 500000 records
# Read 1000000 records
# Written 1000000 records
# ...
# Read 5500000 records
# Written 5500000 records
# Read 6000000 records
# Written 6000000 records
# Written 6100000 records
# Completed: Updated device_type in 920 seconds
```


## Post-Update Records Inspection


If all went well, we should fail to see any records that were skipped over (unless they didn't contain the corresponding <code>device</code> fields), and the below confirms that this is the case.

```python
count = collection.count_documents({ "device_type": None })
print(f"Number of records with device_type == None: {count}")
```

<p></p>

```python
# Number of records with device_type == None: 0
```


# Export a Checkpoint


With all that work and all that waiting done, you may want to export the current state of the database to file, such that you can skip re-doing the <code>device_type</code> classification, if for some reason you decide to delete the database or collection. Unless you are working in an environment like Google Colab, your data should persist even if you close your session.


```python
# this will create the json metadata file as well

export_folder = r'data\checkpoint'

subprocess.run([
    'mongodump',
    '--host', 'localhost',
    '--port', '27017',
    '--db', 'clickstream',
    '--collection', 'clicks',
    '--out', export_folder
], check=True)

print(f"Exported to {export_folder}")
```

<p></p>

```python
# Exported to data\checkpoint
```

The SQL analog (via MySQL through Bash):

```bash
mysqldump --host=localhost --port=3306 --user=root --password --databases clickstream --tables clicks > data/checkpoint/clickstream_clicks.sql
```


# Import From Checkpoint

You can import the data from the checkpoint file as follows. The process is identical to the original import.

```python
HOST = "localhost"
PORT = "27017"
DBNAME = "clickstream"
IMPORT_FILE_FOLDER = r"data\checkpoint\clickstream"
BSON_FILE_NAME = "clicks"
JSON_FILE_NAME = "clicks.metadata"
bson_file = f"{IMPORT_FILE_FOLDER}\\{BSON_FILE_NAME}.bson"
json_file = f"{IMPORT_FILE_FOLDER}\\{JSON_FILE_NAME}.json"
collection_bson = BSON_FILE_NAME
collection_json = JSON_FILE_NAME
```

<p></p>

```bash
!mongorestore --host {HOST}:{PORT} --db {DBNAME} --collection {collection_bson} --drop "{bson_file}"
!mongoimport --host {HOST}:{PORT} --db {DBNAME} --collection {collection_json} --drop --type json "{json_file}"
```

The MySQL via Bash analog would be soemthing like the following.


```bash
# create DB first if doesn't exist
!mysql --host=localhost --port=3306 --user=root --password -e "CREATE DATABASE IF NOT EXISTS clickstream;"

# delete records from table if they exist
!mysql --host=localhost --port=3306 --user=root --password -e "TRUNCATE TABLE clickstream.clicks;"

# import from file
!mysql --host=localhost --port=3306 --user=root --password -e clickstream < checkpoint/clickstream_clicks.sql
```



# Export Flattened Data to CSV

It's a conceivable use-case that you may want to bring data from an unstructured MongoDB database, such as used in the early stages of analysis in a data lake, into a structured SQL data warehouse with enforcable schema, normalizing entity relationships, etc. Below, we will 'flatten' the data such that nested fields are brought out of their hierarchy and assigned null values in the records for which the nested fields do not exist. The MySQL Colab notebook linked to above, after some library installations and imports, will import this flattened data, and provide query analogies to illuminate our understanding of the increasingly complex MongoDB queries and operations below. This flattening and export operation is performed in Python below.


```python
collection = db['clicks']

start_time = time.time()

# Fetch data
data = list(collection.find())

elapsed_time = time.time() - start_time
print(f"Fetched records in {elapsed_time:.0f} seconds")
start_time = time.time()

# Flatten nested fields
flattened_data = []
for doc in data:
    flat_doc = {}
    def flatten(d, parent=''):
        for k, v in d.items():
            new_key = f"{parent}{k}" if parent else k
            if isinstance(v, dict):
                flatten(v, f"{new_key}.")
            else:
                flat_doc[new_key] = v
    flatten(doc)
    flattened_data.append(flat_doc)

# Convert to DataFrame
df = pd.DataFrame(flattened_data)
# Replace periods with underscores in column names
df.columns = df.columns.str.replace('.', '_')

# Export to CSV
df.to_csv(r'clicks_flattened.csv', index=False)

elapsed_time = time.time() - start_time
print(f"Completed: Exported to structured format CSV in {elapsed_time:.0f} seconds")
```

<p></p>

```python
# Fetched records in 79 seconds
# Completed: Exported to structured format CSV in 177 seconds
```

Notice that the nested fields now stand alone, with a format like <code>user_City</code> instead of the hierarchical <code>user.City</code>.


```python
df = pd.DataFrame(pd.read_csv('clicks_structured.csv', nrows=5))
list(df.columns)
```

<p></p>

```python
# ['_id',
#  'webClientID',
#  'VisitDateTime',
#  'ProductID',
#  'Activity',
#  'device_Browser',
#  'device_OS',
#  'user_City',
#  'user_Country',
#  'device_type',
#  'user_UserID']
```

The SQL equivalent:

```sql
USE clickstream;
SELECT COLUMN_NAME
FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_NAME = 'clicks';
```

As a sanity check, we'll count the records in the pandas dataframe exported to CSV, expecting 6.1M like the .bson data originally imported.

```python
df = pd.DataFrame(pd.read_csv('clicks_structured.csv'))
len(df)
```

<p></p>

```python
# 6100000
```



# Add a Sparse Index to user.UserID

```python
# add a userID index
db.clicks.create_index([("user.UserID", 1)], sparse=True)
```

<p></p>

```sql
CREATE INDEX idx_UserID ON clicks (user_UserID(255) ASC);
SHOW INDEXES FROM clicks WHERE Key_name = 'idx_UserID';
```

























# Title....

Time for some analysis. I would like to see the distribution of activity by country, for the top ones anyway. It will also be interesting to compare pageload counts to click counts, and users to non-users. 

**DESCRIBE CODE, ADD COMMENTS, ETC.**



```python
# Function to fetch data for a collection
def fetch_data(collection, top_countries=None, sort_by_total=True):
    # Pipeline for pageloads
    pipeline = [
        {
            "$match": {
                "Country": {"$exists": True, "$ne": None}
            }
        },
        {
            "$project": {
                "Country": 1,
                "pageloads": {"$objectToArray": "$pageloads"}
            }
        },
        {
            "$unwind": "$pageloads"
        },
        {
            "$group": {
                "_id": "$Country",
                "pageloads_count": {"$sum": "$pageloads.v"}
            }
        },
        {
            "$project": {
                "_id": 0,
                "Country": "$_id",
                "pageloads_count": 1
            }
        }
    ]
    if top_countries:
        pipeline.insert(0, {"$match": {"Country": {"$in": top_countries}}})
    pageloads_data = list(collection.aggregate(pipeline))

    # Pipeline for clicks
    pipeline = [
        {
            "$match": {
                "Country": {"$exists": True, "$ne": None}
            }
        },
        {
            "$project": {
                "Country": 1,
                "clicks": {"$objectToArray": "$clicks"}
            }
        },
        {
            "$unwind": "$clicks"
        },
        {
            "$group": {
                "_id": "$Country",
                "clicks_count": {"$sum": "$clicks.v"}
            }
        },
        {
            "$project": {
                "_id": 0,
                "Country": "$_id",
                "clicks_count": 1
            }
        }
    ]
    if top_countries:
        pipeline.insert(0, {"$match": {"Country": {"$in": top_countries}}})
    clicks_data = list(collection.aggregate(pipeline))

    # Merge data
    countries = top_countries if top_countries else sorted(set([d['Country'] for d in pageloads_data + clicks_data]))
    pageloads_counts = [next((d['pageloads_count'] for d in pageloads_data if d['Country'] == c), 0) for c in countries]
    clicks_counts = [next((d['clicks_count'] for d in clicks_data if d['Country'] == c), 0) for c in countries]

    return countries, pageloads_counts, clicks_counts

# Get top 5 countries by total count (pageloads + clicks) for users
users_collection = db['users']
pipeline = [
    {
        "$match": {
            "Country": {"$exists": True, "$ne": None}
        }
    },
    {
        "$project": {
            "Country": 1,
            "pageloads": {"$objectToArray": "$pageloads"},
            "clicks": {"$objectToArray": "$clicks"}
        }
    },
    {
        "$unwind": "$pageloads"
    },
    {
        "$unwind": "$clicks"
    },
    {
        "$group": {
            "_id": "$Country",
            "total_count": {
                "$sum": {"$add": ["$pageloads.v", "$clicks.v"]}
            }
        }
    },
    {
        "$sort": {"total_count": -1}
    },
    {
        "$limit": 5
    },
    {
        "$project": {
            "_id": 0,
            "Country": "$_id"
        }
    }
]
top_countries = [doc['Country'] for doc in users_collection.aggregate(pipeline)]

# Fetch data for users and non_users
users_data = fetch_data(users_collection, top_countries=top_countries)
non_users_collection = db['non_users_weekly']
non_users_data = fetch_data(non_users_collection, top_countries=top_countries)

# Create subplot: 2 rows, 1 column
fig = make_subplots(
    rows=2, cols=1,
    subplot_titles=('Users: Country Distribution', 'Non-Users: Country Distribution'),
    vertical_spacing=0.15
)

# Add traces for Users (Row 1) with legend
fig.add_trace(
    go.Bar(name='Pageloads', x=users_data[0], y=users_data[1], marker_color='#40E0D0', showlegend=True),
    row=1, col=1
)
fig.add_trace(
    go.Bar(name='Clicks', x=users_data[0], y=users_data[2], marker_color='#C71585', showlegend=True),
    row=1, col=1
)

# Add traces for Non-Users (Row 2) without legend
fig.add_trace(
    go.Bar(name='Pageloads', x=non_users_data[0], y=non_users_data[1], marker_color='#40E0D0', showlegend=False),
    row=2, col=1
)
fig.add_trace(
    go.Bar(name='Clicks', x=non_users_data[0], y=non_users_data[2], marker_color='#C71585', showlegend=False),
    row=2, col=1
)

# Update layout
fig.update_layout(
    barmode='group',
    title_text='Distribution of Top 5 Countries for Pageloads and Clicks',
    template='plotly_dark',
    showlegend=True,
    height=800  # Adjust height for two subplots
)

# Update axes
fig.update_xaxes(title_text='Country', row=1, col=1)
fig.update_xaxes(title_text='Country', row=2, col=1)
fig.update_yaxes(title_text='Count', row=1, col=1)
fig.update_yaxes(title_text='Count', row=2, col=1)

# Save and show the plot
fig.write_html('country_distribution_subplots.html')
fig.show()  # Display interactively
```

<p></p>

{% include country_distribution_subplots.html %}


Perhaps we're interested in the correlation between pageloads and clicks. Plotly makes it easy to encode additional information into our charts, such as tooltips when hovering, and the ability to color-code for country. I'll filter to the top 5 countries that we saw above. Note that you can click on legend items to filter the visual to those countries.

**CONFIRM THAT**

```python
collection = db['users_weekly']

# Define countries to filter
countries = ["India", "United States", "United Kingdom", "Australia", "Indonesia"]

# Aggregate data: sum pageloads and clicks per user, filter by country
pipeline = [
    {
        "$match": {
            "Country": {"$in": countries}
        }
    },
    {
        "$project": {
            "userID": 1,
            "Country": 1,
            "pageloads": {"$objectToArray": "$pageloads"},
            "clicks": {"$objectToArray": "$clicks"}
        }
    },
    {
        "$unwind": "$pageloads"
    },
    {
        "$unwind": "$clicks"
    },
    {
        "$group": {
            "_id": {
                "userID": "$userID",
                "Country": "$Country"
            },
            "total_pageloads": {"$sum": "$pageloads.v"},
            "total_clicks": {"$sum": "$clicks.v"}
        }
    },
    {
        "$project": {
            "_id": 0,
            "userID": "$_id.userID",
            "Country": "$_id.Country",
            "total_pageloads": 1,
            "total_clicks": 1
        }
    }
]
data = list(collection.aggregate(pipeline))

# Prepare data for scatter plot
countries_data = {country: {"pageloads": [], "clicks": [], "userIDs": []} for country in countries}
for doc in data:
    country = doc["Country"]
    countries_data[country]["pageloads"].append(doc["total_pageloads"])
    countries_data[country]["clicks"].append(doc["total_clicks"])
    countries_data[country]["userIDs"].append(doc["userID"])

# Define colors for countries
color_map = {
    "India": "#FF6347",  # Tomato
    "United States": "#4682B4",  # SteelBlue
    "United Kingdom": "#32CD32",  # LimeGreen
    "Australia": "#FFD700",  # Gold
    "Indonesia": "#9932CC"  # DarkOrchid
}

# Create scatter plot
fig = go.Figure()

for country in countries:
    if countries_data[country]["pageloads"]:  # Only add traces with data
        fig.add_trace(
            go.Scatter(
                x=countries_data[country]["pageloads"],
                y=countries_data[country]["clicks"],
                mode='markers',
                name=country,
                marker=dict(
                    size=10,
                    color=color_map[country],
                    opacity=0.5,
                    line=dict(width=0.5, color='black')
                ),
                text=countries_data[country]["userIDs"],
                hovertemplate="User: %{text}<br>Pageloads: %{x}<br>Clicks: %{y}<extra></extra>"
            )
        )

# Update layout
fig.update_layout(
    title='Pageloads vs Clicks by User (Filtered by Country)',
    xaxis_title='Total Pageloads',
    yaxis_title='Total Clicks',
    template='plotly_white',  # White background
    xaxis=dict(range=[0, 1500]),  # X-axis limit
    yaxis=dict(range=[0, 10000]),  # Y-axis limit
    showlegend=True,
    height=600
)

# Save and show the plot
fig.write_html('pageloads_vs_clicks_scatter.html')
fig.show()  # Display interactively
```

<p></p>

{% include pageloads_vs_clicks_scatter.html %}


Clearly India is the dominant country. Let's do a drilldown into India by municipality.


```python
collection = db['users_weekly']

# Aggregate data: sum pageloads and clicks per user, filter by India
pipeline = [
    {
        "$match": {
            "Country": "India",
            "City": {"$exists": True, "$ne": None}
        }
    },
    {
        "$project": {
            "userID": 1,
            "City": 1,
            "pageloads": {"$objectToArray": "$pageloads"},
            "clicks": {"$objectToArray": "$clicks"}
        }
    },
    {
        "$unwind": "$pageloads"
    },
    {
        "$unwind": "$clicks"
    },
    {
        "$group": {
            "_id": {
                "userID": "$userID",
                "City": "$City"
            },
            "total_pageloads": {"$sum": "$pageloads.v"},
            "total_clicks": {"$sum": "$clicks.v"}
        }
    },
    {
        "$project": {
            "_id": 0,
            "userID": "$_id.userID",
            "City": "$_id.City",
            "total_pageloads": 1,
            "total_clicks": 1
        }
    }
]
data = list(collection.aggregate(pipeline))

# Prepare data for scatter plot
cities = sorted(set(doc["City"] for doc in data))
cities_data = {city: {"pageloads": [], "clicks": [], "userIDs": []} for city in cities}
for doc in data:
    city = doc["City"]
    cities_data[city]["pageloads"].append(doc["total_pageloads"])
    cities_data[city]["clicks"].append(doc["total_clicks"])
    cities_data[city]["userIDs"].append(doc["userID"])

# Define colors for cities (using Plotly qualitative colors)
colors = [
    "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", 
    "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"
]
color_map = {city: colors[i % len(colors)] for i, city in enumerate(cities)}

# Create scatter plot
fig = go.Figure()

for city in cities:
    if cities_data[city]["pageloads"]:  # Only add traces with data
        fig.add_trace(
            go.Scatter(
                x=cities_data[city]["pageloads"],
                y=cities_data[city]["clicks"],
                mode='markers',
                name=city,
                marker=dict(
                    size=10,
                    color=color_map[city],
                    opacity=0.5,
                    line=dict(width=0.5, color='black')
                ),
                text=cities_data[city]["userIDs"],
                hovertemplate="User: %{text}<br>Pageloads: %{x}<br>Clicks: %{y}<extra></extra>"
            )
        )

# Update layout
fig.update_layout(
    title='Pageloads vs Clicks by User in India (Color-Coded by City)',
    xaxis_title='Total Pageloads',
    yaxis_title='Total Clicks',
    template='plotly_white',  # White background
    xaxis=dict(range=[0, 1000]),  # X-axis limit
    yaxis=dict(range=[0, 5000]),  # Y-axis limit
    showlegend=True,
    height=600
)

# Save and show the plot
fig.write_html('india_pageloads_vs_clicks_scatter.html')
fig.show()  # Display interactively
```

<p></p>

{% include india_pageloads_vs_clicks_scatter.html %}


Plotly is also capable of giving a very compelling map-based visual. Installing and using <code>pycountry</code> will help to provide it with the country code required, and we'll import the <code>math</code> module as well to create a logarithmic color-scale, due to the extreme skew toward India, which would occlude the ability to spot differences among other countries.

```python
import pycountry
import math
```

<p></p>

```python
collection = db['users_weekly']

# Aggregate data: sum pageloads and clicks per country
pipeline = [
    {
        "$match": {
            "Country": {"$exists": True, "$ne": None}
        }
    },
    {
        "$project": {
            "Country": 1,
            "pageloads": {"$objectToArray": "$pageloads"},
            "clicks": {"$objectToArray": "$clicks"}
        }
    },
    {
        "$unwind": "$pageloads"
    },
    {
        "$unwind": "$clicks"
    },
    {
        "$group": {
            "_id": "$Country",
            "total_activity": {
                "$sum": {"$add": ["$pageloads.v", "$clicks.v"]}
            }
        }
    },
    {
        "$project": {
            "_id": 0,
            "Country": "$_id",
            "total_activity": 1
        }
    }
]
data = list(collection.aggregate(pipeline))

# Map country names to ISO 3-letter codes using pycountry
country_code_map = {c.name: c.alpha_3 for c in pycountry.countries}
# Manual overrides for common mismatches
manual_map = {
    "United States": "USA",
    "United Kingdom": "GBR",
    "South Korea": "KOR",
    "Russia": "RUS"
}
country_code_map.update(manual_map)

countries = []
activity = []
log_activity = []
iso_codes = []

for doc in data:
    country = doc["Country"]
    code = country_code_map.get(country)
    if code:  # Only include countries with valid ISO codes
        countries.append(country)
        activity.append(doc["total_activity"])
        log_activity.append(math.log2(doc["total_activity"] + 1))  # Log scale, +1 to handle zeros
        iso_codes.append(code)

# Create choropleth map
fig = go.Figure(data=go.Choropleth(
    locations=iso_codes,  # ISO 3-letter codes
    z=log_activity,  # Log-transformed activity values
    text=countries,  # Country names for hover
    zmin=min(log_activity, default=0),
    zmax=max(log_activity, default=1),
    colorscale='Viridis',  # Color scale
    autocolorscale=False,
    marker_line_color='white',  # Country borders
    marker_line_width=0.5,
    colorbar_title='Log(Activity + 1)',
    hovertemplate="%{text}<br>Activity: %{customdata}<extra></extra>",
    customdata=activity  # Show original activity on hover
))

# Update layout
fig.update_layout(
    title='Total Activity (Pageloads + Clicks) by Country (Logarithmic Scale)',
    template='plotly_white',  # White background
    geo=dict(
        showframe=False,
        showcoastlines=True,
        projection_type='equirectangular'
    ),
    height=600
)

# Save and show the plot
fig.write_html('country_activity_map_log.html')
fig.show()  # Display interactively
```

<p></p>


{% include country_activity_map_log.html %}






# What's Next?



# References

Mongo DB User Docs
- <a href="https://www.mongodb.com/docs/">https://www.mongodb.com/docs/</a>

Plotly User Docs
- ...




