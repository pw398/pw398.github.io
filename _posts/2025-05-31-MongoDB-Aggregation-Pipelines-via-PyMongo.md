---
layout: post
title:  "MongoDB Aggregation Pipelines via PyMongo"
date:   2025-06-09 00:00:00 +0000
categories: MongoDB Python
---


In this second MongoDB article, we continue with the Kirana Store clickstream data, and use aggregation pipelines to derive business insights, and make some beautiful and interactive charts via Plotly. 


<img src="https://github.com/pw398/pw398.github.io/blob/main/_posts/images/mg2.png?raw=true" style="height: 350px; width:auto;">


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


<details open>
<summary>Show Python code</summary>

<pre><code class="language-python">
from pymongo import MongoClient
from pymongo import ASCENDING, DESCENDING
from pymongo import UpdateOne
import pprint as pp
from datetime import datetime
import time
import subprocess
import plotly.graph_objects as go
from plotly.subplots import make_subplots
</code></pre>

</details>





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
# Exported to data\checkpoint
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



# Aggregating by User

We've seen that the number of documents in the collection is very large; there are 6.1M records spanning a date range of only 3 weeks, with most records not corresponding to a user who has created an account. For analytical purposes, some aggregation and segmentation might be useful to reduce redundancy and query time.

And that is precisely what we'll do next: create a new collection called <code>users_weekly</code>, which aggregates the <code>clicks</code> collection to the user and week level, filtering to those with a user ID, and incorporating our <code>device_type</code> classifications. This will include the following fields:


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


It also makes sense to include any user descriptors, which are conveniently nested within the <code>user</code> field of the <code>clicks</code> collection, and should only have one value per user. It's not clear what the full list of nested <code>user</coede>fields are from a single sample, so we'll run the following query to get the full list. This will also provide an introduction to some of the aggregation operators that we'll be doing with.

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
    # project to rename '_id' to 'field' and exclude '_id'
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

Since only a fraction of records contain a user ID, it will make sense to create an index on the <code>user.UserID</code> field. This will be a sparse index, as per the <code>sparse=True</code> element of the command below, which simply means that it will only be applied to the records which contain that field, as applying it to the rest would be redundant. The <code>,1</code> below simply indicates that the sort order will be ascending (-1 would sort it in descending order).

Rather than define the fields like <code>pageload.bot</code> and <code>pageload.desktop</code> explicitly through hard-coding, we'll do what is more natural in an unstructured database, which is loop through all device types when defining the field names starting with <code>pageload</code> or <code>clicks</code>. This way, we would include any types that, in a hypothetical situation, we might not be aware of. It's notable that any device types that a user does not have in the <code>clicks</code> collection (and it's typically the case that only one or two will apply) will not have that nested field in the <code>pageload</code> or <code>clicks</code> fields.

We'll break it up into two cells of code, one doing ........, and the next doing .........


```python
# define collection variable
collection = db['clicks']

# get unique device types
device_types = collection.distinct("device_type")

# group stage for aggregation
group_stage = {
    
    # group by user ID and week number
    "_id": {
        "userID": "$user.UserID",
        "weeknum": "$weeknum"
    },
    
    # get unique days for each group
    "uniqueDays": {"$addToSet": "$day"},
    
    # get the first city value for each group
    "City": {"$first": "$user.City"},
    
    # get the first country value for each group
    "Country": {"$first": "$user.Country"}
}

# iterate through each device type
for device in device_types:
    
    # add pageload count field for the specific device
    group_stage[f"pageloads_{device}"] = {
        
        # sum the matches...
        "$sum": {

            # where...
            "$cond": [

                # both activity = pageload, and device_type = state of current iteration
                {"$and": [{"$eq": ["$Activity", "pageload"]}, {"$eq": ["$device_type", device]}]},
                1,
                0
            ]
        }
    }
    # add click count field for the specific device
    group_stage[f"clicks_{device}"] = {
        
        # sum the matches...
        "$sum": {

            # where...
            "$cond": [

                # both activity = click, and device_type = state of current iteration
                {"$and": [{"$eq": ["$Activity", "click"]}, {"$eq": ["$device_type", device]}]},
                1,
                0
            ]
        }
    }

# create a dictionary mapping device types to pageload fields
project_pageloads = {device: f"$pageloads_{device}" for device in device_types}

# create a dictionary mapping device types to click fields
project_clicks = {device: f"$clicks_{device}" for device in device_types}
```

<p></p>

```python
# define the aggregation pipeline
pipeline = [
    
    # match documents where user.UserID is not null
    {
        "$match": {
            "user.UserID": {"$exists": True, "$ne": None}
        }
    },
    # add week number, and day for unique-days count
    {
        "$addFields": {
            
            # calculate week number from VisitDateTime, subtract 1 so week starts on Monday
            "weeknum": {
                "$week": {
                    "$dateSubtract": {
                        "startDate": "$VisitDateTime",
                        "unit": "day",
                        "amount": 1
                    }
                }
            },
            # format date
            "day": {
                "$dateToString": {
                    "format": "%Y-%m-%d",
                    "date": "$VisitDateTime"
                }
            }
        }
    },
    # group documents according to group_stage
    {"$group": group_stage},
    
    # project fields for the output, restructuring the data
    {
        "$project": {
            # exclude the _id field
            "_id": 0,
            
            # extract the following:
            "userID": "$_id.userID",
            "weeknum": "$_id.weeknum",
            "numDays": {"$size": "$uniqueDays"},
            "City": 1,
            "Country": 1,
            "pageloads": project_pageloads,
            "clicks": project_clicks
        }
    },
    # filter out zero-value pageloads and clicks, converting to objects
    {
        "$project": {
            "userID": 1,
            "weeknum": 1,
            "numDays": 1,
            "City": 1,
            "Country": 1,
            
            # convert pageloads to an object, filtering out zero values
            "pageloads": {
                "$arrayToObject": {
                    "$filter": {
                        
                        # convert pageloads object to array
                        "input": {
                            "$objectToArray": "$pageloads"
                        },
                        
                        # alias for each array element
                        "as": "item",
                        
                        # keep only non-zero values
                        "cond": {"$gt": ["$$item.v", 0]}
                    }
                }
            },
            # Convert clicks to an object, filtering out zero values
            "clicks": {
                "$arrayToObject": {
                    "$filter": {
                        
                        # convert clicks object to array
                        "input": {
                            "$objectToArray": "$clicks"
                        },
                        
                        # alias for each array element
                        "as": "item",
                        
                        # keep only non-zero values
                        "cond": {"$gt": ["$$item.v", 0]}
                    }
                }
            }
        }
    },
    # write results to the 'users_weekly' collection
    {
        "$out": "users_weekly"
    }
]

# execute the aggregation pipeline
collection.aggregate(pipeline)
```

<p></p>

```python
# <pymongo.command_cursor.CommandCursor at 0x26ae3128f80>
```


Confirm the new collection is created.


```python
db.list_collection_names()
````

<p></p>

```python
# ['clicks', 'users_weekly', 'clicks.metadata']
```

View a sample record of the new <code>users_weekly</code> collection.


```python
collection = db['users_weekly']
collection.find_one()
```

<p></p>

```python
# {'_id': ObjectId('68460c946359a7228adb904b'),
#  'City': 'Bengaluru',
#  'Country': 'India',
#  'userID': 'U101337',
#  'weeknum': 20,
#  'numDays': 1,
#  'pageloads': {'desktop': 1},
#  'clicks': {}}
```

Counting the number of documents, you can see that we've reduced the data down to an easily manageable 50K records.

```python
collection.count_documents({})
```

<p></p>

```python
# 50466
```

That is at the user and week level; if we look at the count of unique users, we find about 34K.

<p></p>

```python
len(collection.distinct('userID'))
```

<p></p>

```python
# 34050
```


**GET THE COUNT OF INSTANCES BY USER, SORTED DESC**





# Aggregating by <code>webClientID</code>

Perhaps we don't want to simply discard all of that data corresponding to users without an account. The <code>webClientID</code> does provide more information than just a session ID, as it will remain the same for repeat visitors without an account, if the IP and browser have not changed.

We can execute a pipeline very similar to the above; in fact, identical to the above except that the <code>userID</code> key will be replaced with <code>webClientID</code>. This will enable faster queries to analyze how the behavior of users without accounts compares to "non-users"; a term that seems a little odd because they are using the website, but I'll use it in place of a term like "non-subscribers", as the distinguishing field is a user ID.


**REVISE COMMENTS BELOW**


```python
collection = db['clicks']

# Get unique device types
device_types = collection.distinct("device_type")

# Build dynamic group stage
group_stage = {
    "_id": {
        "webClientID": "$webClientID",
        "weeknum": "$weeknum"
    },
    "uniqueDays": {"$addToSet": "$day"},
    "City": {"$first": "$user.City"},
    "Country": {"$first": "$user.Country"}
}

# Add dynamic fields for each device type
for device in device_types:
    group_stage[f"pageloads_{device}"] = {
        "$sum": {
            "$cond": [
                {"$and": [{"$eq": ["$Activity", "pageload"]}, {"$eq": ["$device_type", device]}]},
                1,
                0
            ]
        }
    }
    group_stage[f"clicks_{device}"] = {
        "$sum": {
            "$cond": [
                {"$and": [{"$eq": ["$Activity", "click"]}, {"$eq": ["$device_type", device]}]},
                1,
                0
            ]
        }
    }
```

<p></p>

```python
# Build dynamic project stage
project_pageloads = {device: f"$pageloads_{device}" for device in device_types}
project_clicks = {device: f"$clicks_{device}" for device in device_types}

pipeline = [
    {
        "$match": {
            "user.UserID": {"$exists": False},
            "webClientID": {"$exists": True, "$ne": None}
        }
    },
    {
        "$addFields": {
            "weeknum": {
                "$week": {
                    "$dateSubtract": {
                        "startDate": "$VisitDateTime",
                        "unit": "day",
                        "amount": 1
                    }
                }
            },
            "day": {
                "$dateToString": {
                    "format": "%Y-%m-%d",
                    "date": "$VisitDateTime"
                }
            }
        }
    },
    {"$group": group_stage},
    {
        "$project": {
            "_id": 0,
            "webClientID": "$_id.webClientID",
            "weeknum": "$_id.weeknum",
            "numDays": {"$size": "$uniqueDays"},
            "City": 1,
            "Country": 1,
            "pageloads": project_pageloads,
            "clicks": project_clicks
        }
    },
    {
        "$project": {
            "webClientID": 1,
            "weeknum": 1,
            "numDays": 1,
            "City": 1,
            "Country": 1,
            "pageloads": {
                "$arrayToObject": {
                    "$filter": {
                        "input": {
                            "$objectToArray": "$pageloads"
                        },
                        "as": "item",
                        "cond": {"$gt": ["$$item.v", 0]}
                    }
                }
            },
            "clicks": {
                "$arrayToObject": {
                    "$filter": {
                        "input": {
                            "$objectToArray": "$clicks"
                        },
                        "as": "item",
                        "cond": {"$gt": ["$$item.v", 0]}
                    }
                }
            }
        }
    },
    {
        "$out": "non_users"
    }
]

collection.aggregate(pipeline)
```

<p></p>

```python
# <pymongo.command_cursor.CommandCursor at 0x26ae3128f80>
```

List the collection names.


```python
db.list_collection_names()
```

```python
# ['non_users_weekly', 'clicks.metadata', 'users_weekly', 'clicks']
```


Sample a record.


```python
collection = db['non_users_weekly']
collection.find_one()
```

<p></p>

```python
# {'_id': ObjectId('68460ee46359a7228adc5756'),
#  'City': 'Milan',
#  'Country': 'Italy',
#  'webClientID': 'WI1000001',
#  'weeknum': 18,
#  'numDays': 1,
#  'pageloads': {},
#  'clicks': {'desktop': 2}}
```

Get the count of documents.

```python
collection.count_documents({})
```

<p></p>


```python
# 1165288
```

Get the count of unique <code>webClientID</code> values.

```python
pipeline = [
    {"$group": {"_id": "$webClientID"}},
    {"$count": "uniqueWebClientIDs"}
]

result = collection.aggregate(pipeline)
count = next(result, {"uniqueWebClientIDs": 0})["uniqueWebClientIDs"]
print(count)
```

<p></p>

```python
# 1054664
```

Get the count of instances for each unique <code>webClientID</code>, sorted descending.

**MISSING CONTENT**


Add a <code>webClientID</code> index.





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




