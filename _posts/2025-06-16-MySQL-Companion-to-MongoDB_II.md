---
layout: post
title:  "SQL II - MongoDB II Companion Piece"
date:   2025-05-14 00:00:00 +0000
categories: SQL MongoDB Bash Python
---


To make the MongoDB II article easier to follow, I included SQL snippets for analogous queries, upon structured data that we derived by flattening out the unstructured data. Since not everything about following along with that in article in SQL will has been detailed there, this will serve as a companion piece, and follow along step-for-step following the creation of the flattened data, in the MongoDB article. Likewise, some content involving shell and Python commands for MySQL setup are only present in this article and notebook.

The intent is to make the largely Javascript-style queries in PyMongo or the Mongo shell look highly relatable, under the assumption you already understand SQL, or find it relatively easier to pick up. Like with the first SQL article, I have chosen to use Google Colab as the notebook's platform, because this is highly replicable, however a downside is that the database will not persist, and you will have to re-install and import libraries (which doesn't take too long) with each new session.



# Outline
- ...
- ...


0


We created the data from...



# Installing and Importing Libraries


```bash
# updates package lists for upgrades, suppresses output
!apt-get update > /dev/null 2>&1

# installs mysql server
!apt-get install -y mysql-server > /dev/null 2>&1

# changes the home directory for stability
!usermod -d /var/lib/mysql mysql

# enables the database server
!service mysql start

# checks status
!service mysql status

# SQL interpreter and options
!pip install ipython-sql
!pip install mysql-connector-python
%load_ext sql
%config SqlMagic.style = '_DEPRECATED_DEFAULT'
%config SqlMagic.autopandas = True
```

<p></p>

```bash
#  * Starting MySQL database server mysqld
#    ...done.
#  * /usr/bin/mysqladmin  Ver 8.0.42-0ubuntu0.22.04.1 for Linux on x86_64 ((Ubuntu))
# Copyright (c) 2000, 2025, Oracle and/or its affiliates.

# Oracle is a registered trademark of Oracle Corporation and/or its
# affiliates. Other names may be trademarks of their respective
# owners.

# Server version      8.0.42-0ubuntu0.22.04.1
# Protocol version    10
# Connection      Localhost via UNIX socket
# UNIX socket     /var/run/mysqld/mysqld.sock
# Uptime:         2 sec

# Threads: 2  Questions: 8  Slow queries: 0  Opens: 119  Flush tables: 3  Open tables: 38  Queries per second avg: 4.000
```

The following file will let us skip username and password entry with each shell command.

```bash
# Create .my.cnf file for password-based authentication
!rm -f ~/.my.cnf /root/.my.cnf                                     # clear if existing
!echo -e "[client]\nuser=root\npassword=pw" > ~/.my.cnf            # print text to CLI
!chmod 600 ~/.my.cnf                                               # grants read/write permissions to file owner
```

<p></p>

```bash
# another connection test
!mysql -N -e "SELECT 1;" || echo "Failed to connect"
```

<p></p>

```bash
# +---+
# | 1 |
# +---+
```

<p></p>

```bash
!sudo mysql -e "ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY 'pw'; FLUSH PRIVILEGES;"
```



# Create Database

Create the <code>clickstream</code> database if it doesn't already exist.

```bash
!mysql --host=localhost -e "CREATE DATABASE IF NOT EXISTS clickstream;"
```

Drop the <code>clicks</code> table if it exists.

```bash
!mysql --host=localhost -e "DROP TABLE IF EXISTS clicks;" clickstream
```



# Show Databases


```python
%sql mysql+mysqlconnector://root:pw@localhost/clickstream
```

<p></p>

```sql
%%sql

show databases;
```

**img sq2-1**




# Read CSV Into Database Table


```python
# connection details
user = 'root'
password = 'pw'
host = 'localhost'
database = 'clickstream'
table_name = 'clicks'

# file path
csv_file = 'clicks_structured.csv'

# connection
engine = create_engine(f'mysql+mysqlconnector://{user}:{password}@{host}/{database}')

# read CSV in chunks, infer schema from a sample
chunk_size = 100000
sample_size = 1000

sample_df = pd.read_csv(csv_file, nrows=sample_size)
dtypes = sample_df.dtypes.to_dict()

# map pandas dtypes to MySQL dtypes
mysql_dtypes = {
    'int64': 'BIGINT',
    'float64': 'DOUBLE',
    'object': 'TEXT',
    'bool': 'BOOLEAN',
    'datetime64': 'DATETIME'
}

# create table schema
columns = [f"`{col}` {mysql_dtypes.get(str(dtype).split('[')[0], 'TEXT')}" for col, dtype in dtypes.items()]
create_table_query = f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(columns)})"

# execute table creation
with engine.connect() as conn:
    conn.execute(text(create_table_query))

progress_interval = 100000
record_count = 0
start_time = time.time()

# process CSV in chunks
for chunk in pd.read_csv(csv_file, chunksize=chunk_size, dtype=dtypes):
    chunk.to_sql(table_name, engine, if_exists='append', index=False)
    record_count = record_count + chunk_size
    print(f"Records processed: {record_count}")

elapsed_time = time.time() - start_time
print(f"Completed: Imported data to database in {elapsed_time:.0f} seconds")
```

<p></p>

```python
# Records processed: 100000
# Records processed: 200000
# ...
# Records processed: 6000000
# Records processed: 6100000
# Completed: Imported data to database in 579 seconds
```


# Inspect Data

## View Table Information


```sql
%%sql

SELECT TABLE_NAME, COLUMN_NAME, COLUMN_TYPE
FROM information_schema.COLUMNS
WHERE TABLE_SCHEMA = 'clickstream'
ORDER BY TABLE_NAME, ORDINAL_POSITION;
```


**img sq2-2**



## Get Count of Unique <code>webClientID</code> Values


```sql
%%sql 

SELECT COUNT(DISTINCT webClientID) 
FROM clicks;
```

**sq2-3**


## Get Count of <code>webClientID</code> Values with a <code>userID</code>


# sql analog to the below
# -----------------------
# SELECT COUNT(DISTINCT webClientID)
# FROM clicks
# WHERE user_UserID IS NOT NULL;

# one-line PyMongo version
# ------------------------
# result = collection.distinct("webClientID", {"user.UserID": {"$exists": True, "$ne": None}})

# two-line PyMongo version
# ------------------------
# result = collection.distinct("webClientID", {"user.UserID": {"$exists": True, "$ne": None}})
# len(result)

## Number of Unique <code>user.UserID</code> Values


```sql
%%sql 

SELECT COUNT(DISTINCT user_UserID) 
FROM clicks;
```



# Classify Device Type as Bot, Desktop, or Mobile


```sql
%%sql 

SELECT COUNT(DISTINCT device_OS) AS Num_Device_OS
FROM clicks
WHERE device_OS IS NOT NULL;
```

<p></p>

```sql
%%sql 

SELECT DISTINCT device_OS
FROM clicks
WHERE device_OS IS NOT NULL
LIMIT 5;
```

<p></p>

```sql
%%sql 

SELECT COUNT(DISTINCT device_Browser) AS Num_Device_Browser
FROM clicks
WHERE device_Browser IS NOT NULL;
```

<p></p>

```sql
%%sql 

SELECT DISTINCT device_Browser
FROM clicks
WHERE device_Browser IS NOT NULL
LIMIT 5;
```

<p></p>

```sql
%%sql 

ALTER TABLE clicks 
DROP COLUMN device_type;
```

<p></p>

```python
#  * mysql+mysqlconnector://root:***@localhost/clickstream
# 0 rows affected.
```

<p></p>

```python
%%sql 

select * from clicks 
limit 1;
```










