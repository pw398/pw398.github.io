---
layout: post
title:  "MongoDB Aggregation via PyMongo"
date:   2025-06-09 00:00:00 +0000
categories: MongoDB Python
---


In this second MongoDB article, we continue with the Kirana Store clickstream data, and use aggregation pipelines to derive business insights, and make some beautiful and interactive charts via Plotly. 


<img src="https://github.com/pw398/pw398.github.io/blob/main/_posts/images/mg2.jpg?raw=true" style="height: 350px; width:auto;">


# Outline

1. ...
2. ...



# Introduction

Brief recap...



# The <code>clickstream</code> Data

As last time, we'll use...

It reflects...

The dataset is available in a .zip file <a href="https://drive.google.com/file/d/1ZRrNKa9sBtyRi1jZ5ocFMrcuuY_erVYH/view?usp=drive_link">here</a>, on Google Drive.



# Aggregation Pipelines




# Import Libraries and Data


```python
from pymongo import MongoClient
from pymongo import ASCENDING, DESCENDING
from pymongo import UpdateOne
import pprint as pp
from datetime import datetime
import time
import subprocess
```


## Connect to Server

```python
client = MongoClient("mongodb://localhost:27017/")
print(client.list_database_names())
```

<p></p>

```python
# ['admin', 'config', 'local']
```


## Drop <code>clickstream</code> if Exists (Optional)

```python
db_name = "clickstream"
client.drop_database(db_name)
print(client.list_database_names())
```


## Import Data

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

<p></p>

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

```python
collection.count_documents({})
```

<p></p>

```python
# 6100000
```


## Date Range

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

<code>webClientID</code> represents ....

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


## Count of <code>webClientID</code>s with a <code>user.UserID</code>


```python
pipeline = [
    {"$match": {"user.UserID": {"$exists": True, "$ne": None}}},
    {"$group": {"_id": "$webClientID"}},
    {"$count": "uniqueWebClientIDs"}
]

result = collection.aggregate(pipeline)
count = next(result, {"uniqueWebClientIDs": 0})["uniqueWebClientIDs"]
print(count)
```


## Count of Unique <code>user.UserID</code> Values

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


# Device Segmentation Analysis

## Distinct Values for <code>device.OS</code>

```python
os_list = collection.distinct("device.OS")
print(", ".join(map(str, os_list)))
```

<p></p>

```python
# Android, BlackBerry OS, Chrome OS, Chromecast, Fedora, FreeBSD, Kindle, Linux, Mac OS X, NetBSD, OpenBSD, Other, Solaris, Tizen, Ubuntu, Windows, Windows Phone, iOS
```


## Distinct Values for <code>device.Browser</code>

<p></p>

```python
os_list = collection.distinct("device.Browser")
print(", ".join(map(str, os_list)))
```

<p></p>

```python
# AdsBot-Google, AhrefsBot, Amazon Silk, Android, AppEngine-Google, Apple Mail, BingPreview, BlackBerry WebKit, Chrome, Chrome Mobile, Chrome Mobile WebView, Chrome Mobile iOS, Chromium, Coc Coc, Coveobot, Crosswalk, Dragon, DuckDuckBot, Edge, Edge Mobile, Electron, Epiphany, Facebook, FacebookBot, Firefox, Firefox Mobile, Firefox iOS, HbbTV, HeadlessChrome, HubSpot Crawler, IE, IE Mobile, Iceweasel, Iron, JobBot, Jooblebot, K-Meleon, Kindle, Konqueror, Magus Bot, Mail.ru Chromium Browser, Maxthon, Mobile Safari, Mobile Safari UI/WKWebView, MobileIron, NetFront, Netscape, Opera, Opera Coast, Opera Mini, Opera Mobile, Other, PagePeeker, Pale Moon, PetalBot, PhantomJS, Pinterest, Puffin, Python Requests, QQ Browser, QQ Browser Mobile, Radius Compliance Bot, Safari, Samsung Internet, SeaMonkey, Seekport Crawler, SiteScoreBot, Sleipnir, Sogou Explorer, Thunderbird, UC Browser, Vivaldi, WebKit Nightly, WordPress, Yandex Browser, YandexAccessibilityBot, YandexBot, YandexSearch, Yeti, YisouSpider, moatbot, net/bot
```


## Classify as Bot, Desktop, or Mobile 

I deferred to Grok to make the following classifications. If a user's browser indicates that it is likely a bot, then that will supersede the desktop vs. mobile classification which the rest of the records will be assigned based on operating system.


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


**FIGURE OUT WHICH UPDATE METHOD TO USE**



To determine whether any records could not be updated:

```python
count = collection.count_documents({ "device_type": None })
print(f"Number of records with device_type == None: {count}")
```

<p></p>

```python
# Number of records with device_type == None: 0
```



## Export to Checkpoint


```python
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
Exported to data\checkpoint
```



## Import From Checkpoint (If Necessary)

If you closed the previous session and wish to pick up where we left off:

```python
# variable-setting; unnecessary if wanting to hard-code the import commands in the next cell
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

List databases:

```python
client = MongoClient("mongodb://localhost:27017/")
print(client.list_database_names())
```

<p></p>

```python
['admin', 'clickstream', 'config', 'local']
```

List collections in <code>clickstream</code>.

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

Sample a record.

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
#  'user': {'City': 'Colombo', 'Country': 'Sri Lanka'},
#  'device_type': 'desktop'}
```

We've seen that the number of documents in the collection is very large; there are 6.1M records spanning a date range of only 3 weeks, with most not corresponding to a user who has created an account. For analytical purposes, some aggregation and segmentation might prove useful in terms of distilling insights and reducing query time. 

That is precisely what I'll do next - create a new collection called <code>users_weekly</code>, which aggregates the <code>clicks</code> collection to the user and week level, filtering to those with a user ID and incorporating our <code>device_type</code> classifications. This will include the following fields:


**REVISE TO BE ALL LOWER CASE?**


- <code>userID</code>: corresponding to unique instances of <code>user.UserID</code> in the <code>clicks</code> collection
- <code>weeknum</code>: the week number corresponding to the <code>VisitDateTime</code> in the <code>clicks</code> collection, with Monday being the start of week.
- <code>numDays</code>: the number of unique days per week that a user has shown activity
- <code>pageloads.bot</code>: the number of records where <code>Activity='pageload'</code> and <code>device_type='bot'</code>
- <code>pageloads.desktop</code>: the number of records where <code>Activity='pageload'</code> and <code>device_type='desktop'</code>
- <code>pageloads.mobile</code>: the number of records where <code>Activity='pageload'</code> and <code>device_type='mobile'</code>
- <code>clicks.bot</code>: the number of records where <code>Activity='click'</code> and <code>device_type='bot'</code>
- <code>clicks.desktop</code>: the number of records where <code>Activity='click'</code> and <code>device_type='desktop'</code>
- <code>clicks.mobile</code>: the number of records where <code>Activity='click'</code> and <code>device_type='mobile'</code>


It also makes sense to include any user descriptors, which are conveniently nested within the <code>user</code> field of the <code>clicks</code> collection. It's not clear what the full list of nested fields are from a single sample, so we'll run the following query to get all of the unique fields in <code>user</code>, providing a nice introduction to some of the aggregation operators we'll be doing with.

```python
# aggregation pipeline to extract fields nested in 'user'

pipeline = [
    # filter to documents where user is not null
    {
        "$match": {
            "user": {"$exists": True, "$ne": None}
        }
    },
    # convert 'user' object to array of key-value pairs
    {
        "$project": {
            "userFields": {"$objectToArray": "$user"}
        }
    },
    # 'unwind' the array to create a document for each key-value pair
    {
        "$unwind": "$userFields"
    },
    # group by field name (key) to get unique field names
    {
        "$group": {
            "_id": "$userFields.k"
        }
    },
    # Project to rename '_id' to 'field' and exclude '_id'
    {
        "$project": {
            "_id": 0,
            "field": "$_id"
        }
    }
]

# execute the pipeline
result = collection.aggregate(pipeline)

# extract field names to list
unique_fields = [doc['field'] for doc in result]

# print
print(unique_fields)
```

<p></p>

```python
# ['City', 'Country', 'UserID']
```

Turns out <code>City</code> and <code>Country</code> are the only other nested fields <code>UserID</code>. We will include these descriptors in the aggregated <code>users_weekly</code> collection.



































# What's Next?



# References

Mongo DB User Docs
- <a href="https://www.mongodb.com/docs/">https://www.mongodb.com/docs/</a>




