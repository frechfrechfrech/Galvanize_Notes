ig## Misc
- Lambda operator + map
    - format is lambda list_of_arguments: expression
        - f = lambda x,y: x+y
    - map r = map(func, seq)
            def fahrenheit(T):
                return ((float(9)/5)*T + 32)
            temp = (36.5, 37, 37.5,39)
            F = map(fahrenheit, temp)
    - Can Also do this other way
            Celsius = [39.2, 36.5, 37.3, 37.8]
            Fahrenheit = map(lambda x: (float(9)/5)*x + 32, Celsius)
- load python as ipython --pylab so that you can still use ipython if charts are open
- the care and keeping of ubuntu:
    Once a month, update all the stuff
    - sudo apt-get update
    - sudo apt-get upgrade
- Import pdb
    pdb.set_trace()
    Super helpful for debugging within functions and classes

## Chris' Galvanize Data Science Immersive Notes

#### General Advice
* You'll be given loads of resources (i.e. lecture slides, tutorials, textbook chapters, etc.). The best thing I did as a student was to put a great deal of effort into organizing my notes. My strategy was to create a comprehensive document (I used markdown), which I added to each night. Later on, when you need to find notes on a specific topic, you'll be able to do a quick Ctrl + F search into the file instead of aimlessly searching through random folders and files on your computer. For example, a typical day in my notes would look something like this:
<br>

    Tuesday, February 20th
    Topic: Linear Regression
    - Notes from Pre-Class Readings
    - Lecture Notes
    - Sections of the individual and pair assignment solutions that were most useful to me.
    - Additional Resources

    Additionally, you might not get through every reading or tutorial that is suggested. That's okay, but be sure to keep track of everything as later on you'll have laid out a convenient to-do list for future learning.

* Commit to a nightly routine and stick with it. When 6:00 PM rolls around, you might be tempted to sack out for the evening. However, it will pay great dividends to create a nightly routine throughout the program. My routine was to take a few hours off after class and then go through that day's pair assignment answer, copy lecture notes into my comprehensive notes file, and complete the pre-class readings for the next day.

* Struggling on an assignment or topic is a good thing! I learned the most when I got stuck on a problem, forced myself to search for an answer online, and solved the problem by myself. Going out of your comfort zone to learn new topics that might not come easily at first will help you in the long run. Having said that, my rule was that if I truly couldn't figure something out in 30 minutes then I'd ask for help. It's equally important to move past the little things and get an understanding of the big picture lessons for the day.

* Keep in mind that the program is designed to be a bit too fast and a bit too difficult. Take advantage of all of the great opportunities to learn, but also find some time for mental health breaks. If you're finishing half of the assignments every day, reviewing the solutions to learn what you missed, and getting at least 50% on the assessments, you're doing great!

* You are almost certain to feel overwhelmed at some point over the next few months. Be kind to yourself and know that if you made it this far into the program, you have what it takes! And chances are, if you're feeling overwhelmed, you're not the only one!

#### Helpful Atom Packages
Go to Packages -> Settings View -> Install Packages/Themes to find these.
* linter-pyflakes: This package is amazing. It highlights syntax errors and other small mistakes in your code before you even run it!
* local-history: This package keeps a history of all changes to your file so you can revert to whatever version you need. Prioritize using Github for version control, but when you inevitably forget to make a commit or two, this will definitely come in handy.
* click-link: So you can actually click on links in markdown files. Not essential but if you get angry having to copy & paste links, you may appreciate this.
* python-autopep8: Also not essential but it's a good habit to learn how to format your code according to best practices.
* minimap: A convenient overview of your entire document so you can navigate your code more easily.

#### Go-to Learning Resources (For Later)
* Markdown: https://www.markdowntutorial.com/
* Linear Algebra: https://www.youtube.com/watch?v=kjBOesZCoqc&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab
* Calculus: https://www.youtube.com/watch?v=WUvTyaaNkzM&list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr
* Python: https://www.youtube.com/watch?v=YYXdXT2l-Gg&list=PL-osiE80TeTt2d9bfVyTiXJA-UTHn6WwU
* Pandas: https://www.youtube.com/watch?v=yzIMircGU5I&list=PL5-da3qGB5ICCsgW1MxlZ0Hq8LL5U3u9y&index=1
* Scikit-Learn: https://www.youtube.com/watch?v=elojMnjn4kk&list=PL5-da3qGB5ICeMbQuqbbCOQWcS6OYBr5A&index=1


## 2018-02-20

### Day Overview:
- Topics:
    - welcome to the program
    - bash
    - git
    - unit testing
- Resources
    - Git Cheatsheet http://ndpsoftware.com/git-cheatsheet.html#loc=workspace
    - Unit testing https://jeffknupp.com/blog/2013/12/09/improve-your-python-understanding-unit-testing/


## bash

pwd - show current directory
ls list
  ls -l show details
cd

creating/removing files
touch filename - creates a file
rm filename - remove a file
mkdir
rmdir -r nonempty_dir #delete nonempty _dir and all contents
cp copy
mv move and rename files

mv *.txt ./inner_directory - move the files ending in .txt to the folder in the current directory called inner_directory
rm -r nonempty_dir # Delete nonempty_dir and all of its contents

bash redirection is just an arrow saying put this1 into this2

* 3.) Use grep to print all the lines having prices for Google (symbol: GOOG). Use bash file redirection to store these GOOG lines into a file named 2015_goog.csv.
grep "GOOG" 2015_sp100.csv > 2015_goog.csv

* 4.) Use sort to sort the lines in 2015_goog.csv. Use bash file redirection to store the sorted lines into a new file named 2015_goog_sorted.csv.
sort 2015_goog.csv > 2015_goog_sorted.csv


* 5.) The Python script plot_stock_prices.py knows how to plot stock price data files. Run that script, and use bash file redirection to input the file 2015_goog_sorted.csv into the script's stdin
python plot_stock_prices.py < 2015_goog_sorted.csv

* 6.) Combine steps 3, 4, and 5 above into one command using bash pipes
grep "GOOG" 2015_sp100.csv|sort| python plot_stock_prices.py

#Open a file in command line:
Go to the location where the file is
python filename.py

jupyter filename.pyinb
#Shortcuts
alt+tab - move between programs
ctrl+tab - move between pages in atom and firefox
ctrl+pgup/pgdn - move between tabs in CLI
ctrl+shift+t - add new tab in CLI
ctrl+w - close firefox tabs


## git stuff

git init - create a new git repository
git status - see the status of changes made
git add filename - add file to the staging area
git diff filename - list changes since the last version of the file
git commit -m "Complete first line of dialogue"
git log
  In the output, notice:

    A 40-character code, called a SHA, that uniquely identifies the commit. This appears in orange text.
    The commit author (you!)
    The date and time of the commit
    The commit message
git show HEAD - git currently on is HEAD, show HEAD to see most recent changes
git checkout HEAD filename - restore file in working directory to look exactly as it
                            did when you made the last commit
git reset HEAD filename - This command resets the file in the staging area to be the
    same as the HEAD commit. It does not discard file changes from the working directory,
    it just removes them from the staging area.
git reset commit_SHA - This command works by using the first 7 characters of the SHA of a previous commit.
      HEAD is now set to that previous commit.

Pair programming in Git
Pair work
- Define roles (A - working out of the branch in his/her repo, B - collaborator)
  - A and B fork and clone repo like normal.
  - A:
    - needs to add B as a collaborator for that repo on his/her Github.
    - adds a branch, e.g. $ git checkout -b pair_morning
    - starts coding (B navigating and helping)
    - Add, commit, push the branch to Github when it’s B’s turn to code.
        - e.g. $ git push origin pair_morning
  - B:
    - After A adds B as collaborator, adds A’s repo as a remote:
    $ git remote add <partner-name> <partner-remote-url>
    - Help A!
    - When B’s turn to code comes:
     - $ git fetch <partner-name>
     - $ git checkout --track <partner-name>/<branch-name>
     - starts coding (A navigating and helping)
    - When A’s turn comes again:
     - $ git push <partner-name> <branch-name>
        - e.g. $ git push <partner-name> pair_morning
- Continue switching back and forth
  - A :remote-name> <branch-name>` e.g. $ git push <partner-name> pair_morning


#anaconda python 2 and 3
python 3 is my default
source activate py2 - to activate python 2
source deactivate - to go back to python 3


# 2018-02-21

### Day Overview:
- Topics:
    - Object oriented programming
        - if __name__ == "__main__":
    - Unit testing
- Resources
    - OOP youtube lectures https://www.youtube.com/watch?v=ZDa-Z5JzLYM&list=PL-osiE80TeTt2d9bfVyTiXJA-UTHn6WwU&index=37
    - if __name__ == "__main__": https://stackoverflow.com/questions/419163/what-does-if-name-main-do


- The [pair_morning](pair_morning.md) assignment asks you to write functions using Pythonic syntax and write tests for those functions using Python's unittest module.

- The [pair_afternoon](pair_afternoon.md) assignment asks you to use object-oriented programming to modify and write some card games.


# 2018-02-22

### Day Overview:
- Topics:
    - SQL with postgreSQL
    - Interacting with postgreSQL through psycopg
- Resources

## Morning SQL Lecture

### RDBMS Terminology
- Schema defines the structure of table or database_name
- Data type - integer, text, or date
- First task when you get access to a database is typically to understand the schema

### SQL is
- a tool to interact with relational database management systems
- a declarative language, unlike python, which is imperative. With a declarative language, you tell the machine what you want, instead of how, and it figures out the best way to do it for you.



### Create your database

You need to import the data in the `degrees_data.csv`, `student_data.csv`, and `term_gpa_data.csv` into PostgreSQL.  A script that contains a schema has been provided (`make_sql_for_ds_script.sql`) to help you do this.  Make a PostgreSQL database called `sql_for_ds` using this script and the directions below.

1) From terminal in the same directory as this file, start postgres:  
   ```bash
   $ sudo -i -u postgres
   ```
2) Make the `sql_for_ds` database:  
   ```bash
   $ createdb sql_for_ds
   ```
3) Exit postgres:  
   ```bash
   $ exit
   ```
4) Open the /home/alex/Galvanize/dsi-sql/SQL_students/make_sql_for_ds_script.sql file with a text editor ( and inspect the contents.  Note how the schemas are created.  Find the paths to the `degrees_data.csv`, `student_data.csv`, and `term_gpa_data.csv` and change them to what they are on your machine. Save the file and close it.

5) Run the `make_sql_for_ds_script.sql` file and import results into the `sql_for_ds` database that you made.
   ```bash
   $ psql sql_for_ds < make_sql_for_ds_script.sql
   ```  
   At this point you should see a list of commands go by:  `BEGIN SET CREATE TABLE etc`  
6) Open the database you just made:  
   ```bash
   $ psql sql_for_ds
   ```
7) Try out some of the psql cheatsheet commands in `psql_commands.png`  

8) Try writing queries to answer the questions contained in `queries.sql`


### Write your query/run it
- Alias
```SELECT
    cust_name AS name
    (as is not required but is helpful)```

- CASE WHEN

- GROUP BY - any column in select clause that is not an aggregator must be in the group by clause

- WHERE only works on individual rows. Filter for aggregated values, use HAVING

- JOIN - default is inner. Also LEFT, FULL JOIN (aka outer join)

- Order of components vs order of evaluation

| Order of Components | Order of Evaluation |
| ------ | ------ |
| SELECT | 5 - Targeted list of columns evaluated and returned |
| FROM | 1 - Product of all tables is formed |
| JOIN / ON | |
| WHERE | 2 - Rows filtered out that do not meet condition |
| GROUP BY | 3 - Rows combined according to GROUP BY clause and aggregations | applied |
| HAVING | 4 - Aggregations that do not meet that HAVING criteria are removed |
| ORDER BY | 6 - Rows sorted by column(s) |
| LIMIT | 7 - Final table truncated based on limit size |
| ; | 8 - Semicolon included as reminder |

- Subqueries
    -

- Commands for psql
    - \q - quit
    - \c database - connect with a database
    - \d tablename - show table definition
    - \dt *.* - list tables from all schemas (if *.* omitted, will only show SEARCH_PATH ones)
    - \l - list databases
    - \dn - list schemas
    - \df - list functions
    - \dv - list views
    - \df+ - funciton
    - \x - pretty-format query results

### Self-prescribed Assignment
  - Try to set up my own database
  - go through the afternoon lecture
  - difference between distinct and group by
  - learn about CASE
  - multiple joins


## Afternoon Python interact with SQL through psycopg2
    - Useful for data pipelines, pre-cleaning, data Exploration
    - allows for dynamic query generation

### Pyscopg2
  - python library the connects and interacts with postgressql databases
  - General workflow
        - eestablish connection to postgres database using psycopg2
        - create a curspr
        - use the cursor to execute SQL queries and retrieve data
        - comit SQL actions
        - close cursor conenction

    - what is cur fetchall

    - ESTABLISH A CONNECTION
        conn = psycopg2.conect(dbname='socialmedia',user='alex', host=*'/var/run/postgresql'*) - the bolded host is true for my computer

        - figure out the host

        ```sudo -u postgres psql -c "SHOW unix_socket_directories;```

        - user for other people can be postgresql but for me(probs linux in general, it's my username, alex)


# 2018-02-23

### Day Overview:
- Topics:
    - NoSQL with Mongodb
    - Interacting with Mongodb through PyMongo
    - query an API (The New York Time Articles)
- Resources
 - Bulk Downloads/ FTP servers
    - Amazon S3 public [datasets](http://aws.amazon.com/publicdatasets/)
    - [InfoChimps](https://github.com/infochimps-labs/wukong-example-data/tree/f3c0820fb35cdb9c5f739ce46fafee1ebc3cc84c)
    - Academia -- [Stanford](http://snap.stanford.edu/data/) and [UCI](http://archive.ics.uci.edu/ml/)

 - APIs -- public and hidden
    - [Twitter](https://dev.twitter.com/)
    - [Foursquare](https://developer.foursquare.com/)
    -  [Facebook](https://developers.facebook.com/search/?q=apis&notfound=1)
    * [Tumblr](http://www.tumblr.com/docs/en/api/v2)
    * [Yelp](http://www.yelp.com/developers/documentation)
    * [Last.fm](http://www.last.fm/api)
    * [bitly](http://dev.bitly.com/)
    * [LinkedIn](https://developer.linkedin.com/apis)
    * [Yahoo Finance (hidden)](http://greenido.wordpress.com/2009/12/22/yahoo-finance-hidden-api/)
    * [Trulia](http://developer.trulia.com/)
    * [Evernote](http://dev.evernote.com/documentation/cloud/)
    * [Songkick](http://www.songkick.com/developer/)
    * [Zillow](http://www.zillow.com/howto/api/APIOverview.htm)

## NoSQL with Mongodb
 - SQL is good for
    - structured data
    - data where attributes stay the same over time aka schema
 - NoSQL aka "Not only SQL"
    - document-orientd - each object (row/file) is stored in one place
 - How to interact with mongo
    - initiate with command mongo
    - show dbs
    - use "name of db"
        - through this action, the database you choose to use becomes aliased as db
    - show collections - will show all the collections within the current database
    - then you can do things on the collections within the database
        - db.collection.method
        - db.collection.find({where}, {select})
 - practice
    - How many entries are in the log collection?: `db.log.find().count()`
    - limit the query: `db.log.find().limit(10)`
    - how many records are in san fran `db.log.find({cy:'San Francisco'}).count()=22`
    - how many distinct values are stored under field a? `db.log.distinct('a').length=559`
    - find all the documents where the value of field a contains Mozilla or Opera: `db.log.find({a: {$in: [/Mozilla/,/Opera/]}}).pretty`
    - Convert the timestamp field to the date type.
     - You will need to multiply the number by 1000 and then make it a Date object (you can create a Date object by using new Date()).
     - You can loop over each record using .forEach() and then .update() the record
     - `db.log.find({'t': {$exists: true}}).forEach(function(doc) {db.log.update({_id: doc._id},{$set:{t: new Date(doc.t*1000)}})})`
     - **Update:** Modify an entry in a collection

         * `db.users.update({name: "Jon"}, {$set: {friends: ["Phil"]}})`
         * `db.users.update({name: "Jon"}, {$push: {friends: "Susie"}})`

## PyMongo
- Start
        import pymongo
        client = pymongo.MongoClient[]
        db = client['whateveryoucallthedatabase']
        collection = db ['whateveryoucallthecollection']

- main differences between mongodb and pymongo are put '' around all fields in pymongo and pymongo returns generators
- in pymongo collection.find() returns a generator
- deal with generator through
- coll.insert({'name': 'Jon', 'age': '45', 'friends': ['Henry', 'Ashley']})
- "All entries: print list(coll.find())
- "Just one:" print coll.find_one()
- "Added Jon's car" - coll.update({'name': "Jon"}, {'$set': {'car': "Prius"}})

## practice



# 2018-02-26

### Day Overview:
- Topics:
    - Numpy
    - Pandas
- Resource
    - Readings:
        - Numpy https://docs.scipy.org/doc/numpy-dev/user/quickstart.html
        - Pandas http://pandas.pydata.org/pandas-docs/stable/10min.html
        - Matplotlib https://matplotlib.org/users/artists.html

### Reading Notes
- Numpy
        import numpy as np

        *make an array*
        a = np.arange(15).reshape(3, 5)
        b = np.array([1,2,3,4])
        c = np.zeros((3,4)) - will make a 3x4 matrix of zeros
        d = np.ones(2,3,4) - will make 2 3x4 matrices of zeros stacked on top of each other.

        *About the array*
        a.ndim - # of axes aka rank
        a.shape - dimensions
        a.size - total number of elements in the array
        a.dtype - describes type of elements in array

        *Generate list with steps*
        np.arange(start, end, step) will create an array with those elements
        np.linspace(start, end, # elements desired)

        *Multiplication*
        >>> A = np.array( [[1,1],
        ...             [0,1]] )
        >>> B = np.array( [[2,0],
        ...             [3,4]] )
        >>> A*B  # elementwise product
            array([[2, 0],
                    [0, 4]])
        >>> A.dot(B)    # matrix product
            array([[5, 4],
                    [3, 4]])
        >>> np.dot(A, B) # another matrix product
            array([[5, 4],
                    [3, 4]])

        *Operate on all elements of the matrix*
        a*3 = all elements of a times 3
        a+4 = all elements of a plus 4

        *Methods*
        a.sum() - sum all the elements in the array
        a.min() - min of elements in array

### Pandas
Overview:
Vectors = Series
Tables = Dataframes
You can label your columns, which is pretty great.

Most important functions

- import all the stuff
        import pandas as pd
        import numpy as np
        import matplotlib as plt
- Create a dataframe using csv `df = pd.read_csv('filepath/name')`
- Make new column
    - `df['Total Charges'] = pd.Series(df['Discharges']*df['Mean Charge'], index=df.index)`
    - or `df['Total Charges'] = df['Discharges']*df['Mean Charge']`

- Select data
    - using boolean mask
        - `df[ (df['Age'] > 24) & (df['EyeColor' == 'Blue']]` will select for all the rows where Age is greater than 24 and eye color is blue.
        - this works by creating a boolean table populated with trues and falses. Then overlaying that on the original dataframe, only retrieve reacords from the original data frame where the rows==True.
    - using loc - `df.loc[[row names or : for all], [column names or : ]]`
    - using filter `df.filter(['list of','column names to keep'])`
    - df.head() will show you the top 5
    - drop data. As a precaution, won't just drop data. Need to reassign to itself so that the data gets dropped.
        -`df = df.dropna(how = 'any')` - any will drop where any of fields are NA. 'all' will drop where all of the fields in a record are NA.
        -
- Functions
    - `correlation df.corr()`
    - `df.sort_values([field on which to sort], ascending = True default)`
    - `df.merge(df2, how='inner' default, on(['list of', 'columns to join on']`
    - `df.groupby('Field Name')` returns a generator until you put an aggregating function on it, then it returns a new dataframe. Other options are.
        - `df.groupby('Field1').sum()`
        - `df.groupby('Field1').sum()['field2']` will only pull out series corresponding to field2
    - df.info
    - df.describe() will give mean, median, max, etc of the dataframe
    - df.columns


- Colors
    - Hex is base 16
        - 3 categories or color
        - 6 characters can range from 0 to f
        - ff is most saturation for a given category of color
    - Different kinds of data
        - Nominal/Categorical - blue/red
        - Ordinal - has order
        - Interval - even spacing, ordered, measurable
        - Ratio - constant difference betwen
    - Different ways of differentiating data
        - color scaled or distinct colors
        - Recommended no more than 7-8 distinct colors at a time
        - texture
    - Resources
        - look at links in Notebook data_viz_lecture_elliot_cohen.ipynb


# 2018-02-27

### Day Overview:
    - Topics:
        - matplotlib
        - pandas + stats
    - Resources
        - https://matplotlib.org/gallery/style_sheets/style_sheets_reference.html

### Intro
    - matplotlib reading https://www.kdnuggets.com/2017/01/datascience-data-visualization-matplotlib.html
    - welcome repo https://github.com/gSchool/dsi-welcome/tree/18-01-DS-DP
    -


# 2018-02-28



### Morning stats lecture
- statistical inference means we are trying to understand something about the population from  sample
- descriptive statistics just describes the sample
- statistical model is a complex representation of the phenomena that made the data
- estimation is figuring population parameters based on model
    - point estimate - single value for parameter
    - interval estimate
- random variable
    - indepedentently drawn from identical distributions (iid)
- pmf, pdf, cdf, ppf
    - probability mass function is for discrete random variables
    - probability density function is for continuous random variables
    - cumulative distribution function cdf for both
    - pecerntile point function
        - x = np.arange(stats.poisson.ppf(0.001, mu),
              stats.poisson.ppf(0.999, mu)) - from 0.1% chance to 99.9% of area under curve

- one of 3 methods for estimating parameters
    - method of moments
        - mean
        - variance
        - skewness, etc
    - maximum likelihood estimation
        - max likelihood that of sample data given the distribution parameters (that we choose)
        - if P(X|theta1) > P(X|theta2)
        - assume dist
        - define likelihood function
        - choose params that maximize likelihood function
        - likelihood that the data gets pulled given that our selected parameters are true
    - maximum a posteriori
        - similar to ML but reverse
        - find parameter to maximize likelihood of parameter given data
            - f(theta | data) proportional to f(data | theta) g(theta) where g(theta) is prior belief
- Kernel density estimatation (KDE)
    - nonparametric method for when you don't know a distribution
    - center kernel function (usually guassian) over where the data occur
    - pick standard deviation of kernel - bandwidth
- Parametric vs nonparametrics
    - Parametric
        -based on assumptions about the dist of the underlying pop
        - if the data deviates strongly from assumptions, could lead to incorrect conclusions
    - nonparametrics
        - not based on assumptions about dis of underlying pop
        - generatlly not as powerful
        - interpretation can be difficult

### Afternoon lecture on sampling
- Random sampling tough
    - try your best
    - call out possible arguments to your sample
    - one other step

- Types of random sampling
    - Random sampling
        - simple random sampling 1/n chance of being selected
    - Stratified random sampling
        - each of k groups to be equally represented
        - select equal number of samples at random from each group
    - Cluster random sampling
        - prevent response bias (like only people who get jobs respond to galvanize job survey)
        - Target specific clusters and then try to get representative info for that cluster

- Central limit theorem
    - mean of a bunch of samples of any distribution  will be normal
    - any time you have something that is the sum of iid quantities
    - std deviation of the sample means decreases as sample size increases because harder and harder to get sample means that are far away from the true mean.

- Confidence Interval
    - Known relationship between sampling stats and their dist to calculate pop stats - ex CLT for means
        - normal distribution
            - stats.norm..ppf(% of area under curve to left)
            - mu = xbar +/- 1.96(stddevbar/rootn)
        - t dist when n is small(less than 30)
            - stats.t.ppf(% , df)
    - Bootstrapping
        - estimates the sampling distribution of an estimator (like mean) by sampling with replacement from the original sample
        - how
            - start with dataset of size n
            - sample with replacement to create 1 bootstrap sample of size n
            - repeat B times (200-2000 times)
            - each bootstrap sample can then be used as observations in a separate dataset for estimation and model fitting
            - throw away top 2.5% and bottom 2.5% and then we have our confidence interval
        - WHY DOES IT WORK?? https://stats.stackexchange.com/questions/26088/explaining-to-laypeople-why-bootstrapping-works
            - Basically because your sample is probably like the population sample so treating it like the population itself is a good way to approximate the axtual sample stats


# 2018-03-01

# Objectives
- understand theory behind hypo testing
- interpret p values
- type 1 and type 2 errors
- pdf, cdf, ppf
- scipy

- don't test a hypothesis directly. outline everything else and then disprove that.
- power calculation to determine how many observations you'll need to make the test statistically significant


### Procedure for hypo testing
    1. State your scientific question
    2. Define your null, your alternative and a level of $\alpha$
    3. Choose an appropriate hypothesis test
    4. **Collect data**
    5. Calculate the test statistic
    6. Calculate the *p*-value
    7. Compare your *p*-value to $\alpha$ and reject the null or fail to reject the null

- scipy.stats
    - stats.norm.cdf(x, loc = mu, scale = sd)
    - stats.binom.cdf(x, loc = n, scale = p)
    - stats.poission.cdf(x=k, mu, loc=0)
    - loc usually refers to mean, scale usuall refers to shape, std dev
- plot
        fig, ax = plt.subplots(1, figsize=(10, 4))

        p = 0.5
        experiment = np.array([1,1,1,0,1,1])
        h = experiment.sum()
        n = experiment.size
        x = np.arange(s.binom.ppf(0.01, n, p), s.binom.ppf(.99, n, p)+1)
        ax.plot(x, s.binom.pmf(x, n, p), 'bo', ms=8, label='pmf')
        ax.set_ylabel("probability mass")
        ax.set_xlabel("number of tea cups guessed correctly")
        lines = ax.vlines(x, 0, s.binom.pmf(x, n, p), colors='b', lw=7, alpha=0.5)
        target_pmf = [s.binom.pmf(5, n, p),s.binom.pmf(6, n, p)]
        ax.vlines([5,6],0,target_pmf,colors='r', lw=7, alpha=0.5)
        ax.plot([5,6], target_pmf, 'ro', ms=8, label='pmf');
- type 1 and type 2
    - typical vernacular - "insufficient evidence to reject the null"
    - type 1 error - falsely reject the null hypothesis
    - error accept it when
- chi squared
    - contingency table
    - if >20% of cells in contingency table <5 observations, then use fishers
- paired t test
    - samples are not independent. like sample of patients before and after treatment
- criteria for case study
    - 1. some kind of hypothesis testing
    - 2. some kind of visualization
    - 3. summarize data


# 2018-03-05

### Schedule
- 9:30-10:30
    - Power
    - Intro to R
    - Individual
- 2-3 Bayesian Stats
- 3-6 pair


Things we went over
    - /Galvanize/stats/power-bayes/html/index.html
        - power, r
    - r_and_python_cars_indiv_assignment.ipynb
    - EXTRA REFERENCE: Bayesian methods http://nbviewer.jupyter.org/github/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/blob/master/Chapter1_Introduction/Ch1_Introduction_PyMC3.ipynb
    - EXTRA REFERENCE: Differnce between t test and ANOVA https://keydifferences.com/difference-between-t-test-and-anova.html
    - EXTRA REFERENCE: Play around with power, alpha, n and d http://rpsychologist.com/d3/NHST/

Power calculation
    - 1-Beta
    - determine number of samples needed to power a given study
    - Things that affect statistical power are:
        1. significance level
        2. difference between mu0 and mu1 (effect size)
        3. variance
        4. sample size
    - R has great ways to
Probabalistic programming
    - How
        - Random variables are handled as primitive (like strings)
        - inference handled behind the scnees
        - memory and processor management is abstracted away
    - Pros
        - customization
        - propagation of uncertainty
        - intuition
    - Cons
        - assumes deep statistical understanding
        - [fill in from ppt]
    - Box
        - Data:
            Loop through infer -> criticize -> model -> infer
    - import pymc3 as pm
    - Basic workflow
        - define hyperpriors
        - open a model context
        - perform inference
        - get a trace = all sample from inference algorithm


2018-03-06
    - Reading:
        - ISLR pg 59-68 on linear regression
        - ISLR expert videos: https://www.r-bloggers.com/in-depth-introduction-to-machine-learning-in-15-hours-of-expert-videos/
    - Morning practice Sklearn
        - object-oriented-programming-in-sklearn.ipynb
    - Linear Regression all day pair
        - linear-regression-morning-lecture.ipynb
        - Linear_Regression_Pair.ipynb
        - linear-regression-sprint-solution.ipynb
        - assumptions of linear regression:
            0. Sample data is representative of the population.
            1. True relationship between the transformed features of X and y is linear in the coefficients.
            2. Feature matrix X has full rank (rows and columns are linearly independent).
            3. Residuals are independent.
            4. Residuals are normally distributed.
            5. Variance of the residuals is constant (homoscedastic).```


2018-03-07
    - No reading overnight
    - Schedule
        - Speaker Victor from SendGrid
        - Linear Regression warm up
            - jupnote
            - my questions:
                - why intercept?
                - how is spline actually functioning on the underbelly?
                - why not use logistic regression for binary variables?
        - Logistic regression
            - confusion matrices
            - horse_or_dog.ipynb
            - individual.md was the assignment we did in pair. PULL REQUEST
    - Additional resources
        - dummies for categorical variables
        - model selection


# 2018-03-08

No reading overnight
Schedule
    - repo dsi-regularized-regression
    - Derive normal solution to linear regression
        - aka the value of Beta vector to minimize sum of least squares. (see dead tree book)
        - problem is rank deficiency will prevent inverting of x so that the solution doesn't work
    - Regularization_Elliot.pdf
        - Outlier discussion
            - Outliers
            - high leverage point
            - Hat Matrix
        - Curse of dimensionality
        - shortcomings of unregularized regression
        - Ridge Regularization
        - LASSO regularization
    - Cross_Validation_Elliot.pdf
        - use it to choose coefficients and lambda (hyperparameter for
        -
Extra resources
    - Bias-variance infographic - https://elitedatascience.com/bias-variance-tradeoff
    - [Markdown CheatSheet](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet)
the


*** SPRING BREAK ***

# 2018-03-26
Learning Objectives
    Morning lecture - kNN Lecture.pdf
        ● Explain the difference between parametric and non-parametric models
        ● Know the hyperparameters for kNN and how the algorithm works    
        ● Compute common distance metrics used in kNN
        ● Understand the problems posed by high-dimensional data
        ● Know the advantages & disadvantages of using kNN
    Individual - https://github.com/frechfrechfrech/dsi-non-parametric-learners
    Afternoon Lecture - Decision Trees Lecture.pdf
        ● Understand how the algorithm for decision trees works
        ● Be able to calculate Shannon entropy and information gain
        ● Explain the concept of recursion and how it relates to decision
        trees
        ● Know advantages & disadvantages of decision trees


Overview of kNN Algorithm
1. Choose a value for the hyperparameter k - how many neighbors do you want to look at for a given data point?
    - common starting point is k = square root of n
    - cross validation to find k
    - k = 1, overfit, will be too complex
    - k = 50, underfit, won't be complex enough
2. Calculate the distance all data points are from each other - this is the other hyperparameter
    - Euclidean distance (kinda like the hypoteneuse)
    - Manhattan distance (sum of distance between likes)
    - If don't care about magnitude, but rather care about degree  (text analysis)
        - Cosine similarity (-1 same, 0 90 deg, 1 180 deg)
        - Cosine distance - opposite
    - MAKE SURE DATA IS SCALED!!!
3. Find the closest k points to each data point, i.e. its neighbors
    - point weighting
4. Make a prediction for each data point
    a. For classification, assign a data point’s category based on what category the majority of its neighbors are (e.g., if 2 neighbors are dogs and 1 neighbor is a horse, then you classify that point as a dog)
    b. For regression, calculate a data point’s value by taking the average value of its neighbors

Advantages & Disadvantages of kNN
    Advantages:
        Simple to implement
        The training phase is just storing the data in memory
        Works with any number of classes/categories
        Easy to add in additional data
        Only two hyperparameters - k & the distance metric
    Disadvantages:
        Poor performance in high dimensions (KNN does not work well in 5+ dimensions, review the curse of dimensionality)
        Very slow to run, especially with large datasets
        Categorical features don’t work well with kNN - tend to dominate because the distance between 0 and 1 when features are scaled is the furthest away two things can be

minimum x? Meh, not really a rule of thumb
signal in the data (little signal in the data)

Afternoon Lecture: Decision trees

Nodes are decision points
    - Root Node
    - Parent node
Concepts
    - Entropy - a measurement of the diversity in a sample
        - High entropy - a sample that is partly made up of dogs and partly made up of horses
        - Low entropy - a sample that is 100% dogs
    - Decision trees split data on features to decrease entropy
    - a way to measure how much we reduced the entropy by splitting the data in a particular way
        - If we decrease the entropy by a large
The Decision Tree Algorithm
    1. Consider all possible splits on all features
    2. Calculate & choose the “best” split
        a. Classification trees - the best split is the split that has the highest information gain when moving from parent to child nodes
            - Shannon entropy = sum over each category(-proportion in split*log2(proportion in split))
            - information gain = entropy of parent - prop_group_a(entropy a) - prop_group_b(entropy_b)
        b. Regression trees - the best split is the split that has the largest reduction in variance when moving from parent to child nodes    


Gini Index
    - Sklearn defaults to gini index
    - Measures the probability of misclassifying a single element if it was randomly labeled according to the distribution of classes in the sample
    - in practice very similar to shannon entropy most of the time

Decision Tree Pseudocode
function BuildTree:
    if every item in the dataset is in the same class or there is no feature left on which to split the data:
        return a leaf node with the class label
    else:
        find the best feature and value to split the data on
        split the dataset
        create a node
        for each split
            call BuildTree and add the result as a child of the node
        return node
Pruning to prevent overfitting
    - increase bias to decrease variance
    - pre-pruning
        - Leaf size: stop splitting when the number of samples left gets small enough
        - Depth: stop splitting at a certain depth (after a certain number of splits)
        - Purity: stop splitting if enough of the examples are the same class
        - Gain threshold: stop splitting when the information gain becomes too small
    - post-pruning
        - Merge terminal nodes (i.e., undo your split) if doing so decreases error in your test set
        - Set the maximum number of terminal nodes; this is a form of regularization
    - no automated pruning in sklearn
Algorithms for decision trees
    - ID3: category features only, information gain, multi-way splits
    - C4.5: continuous and categorical features, information gain, missing data okay, pruning
    - CART:
        - continuous and categorical features and targets, gini index, binary splits only
        - DEFAULT in sklearn
Advantages:
    Easy to interpret
    Non-parametric/more flexible model
    Can incorporate both numerical and
    categorical features*
    Prediction is computationally cheap
    Can handle missing values and outliers*
    Can handle irrelevant features and
    multicollinearity

Disadvantages:
    Computationally expensive to train
    Greedy algorithm - looks for the simplest, quickest model and may miss the best model (i.e., converges at local maxima instead of global maxima)
    Often overfits
    Deterministic - i.e., you’ll get the same model every time you run it

Decision Trees in sklearn
    ● Uses the CART algorithm (see http://scikit-learn.org/stable/modules/tree.html#tree, section
    1.10.6)
    ● Uses Gini Index by default (but you can change it to entropy if you’d like)
    ● You can prune by varying the following hyperparameters: max_depth, min_samples_split, min_samples_leaf, max_leaf_nodes
    ● Must use dummy variables for categorical features
    ○ E.g., a column with [‘Red’, ‘Green’, ‘Blue’] would need to be coded as separate variables -
    Is_it_red (Y/N), Is_it_green (Y/N), Is_it_blue (Y/N)
    ○ See ‘Feature Binarization and Encoding Categorical Features’ at
    http://scikit-learn.org/stable/modules/preprocessing.html
    ● Does not support missing values (even though CART typically does)
    ● Only supports binary splits

# 2018-03-27
Learning objectives

- Bagging
- Bumping
- Boosting
- Random forests

Mortgages jupyter notebook
randomization
PDF https://github.com/gSchool/DSI_Lectures/blob/master/random-forest/elliot_cohen/random_forests.pdf

Bootstrapping Aggregation (Bagging)
 - bootstrap data, fit a bunch of models, average the predictors
 - bootstrapped trees provides unbiased, high variance predictors
 - averaged estimators are lower variance than single predictors
 - average predictors are still unbiased
 - bootstrapping explores the data set/predictors space: then uses the average of predictors we might see under sample
 - Out of bag testing error - cross-validation
    - boostrap samples leave out, on average, 1/3 of the data.
    - each bootstrap tree can be trained on the data that was selected, tested on the rest

Random forest
    - randomization along the rows from bagging
    - only look at a subset of features at each node
    - temporary exclusion - features includable again at each split
    - classication use root p and regression use p/3 features
    - more features means stronger but also more correlated trees
    - Tuning
        - g(p) number of features considered at each split
        - m number of trees (default in sklearn 10 or 50, elliot recommends 500, or 1000)
        - nb sample size of each bootstrapped sample
        - x is tree characteristics (those are the things that you can tune on a tree, like max nodes and number of samples in a leaf)
    - sometimes training is faster/more manageable than storing 5000 trees at a time
        - fit + predict every time
    - randomization
    - can be parallelized independently

How do we interpret?
    - Partial dependency plots
        - change every instance of feature X1 to some value
            - find average prediction over all trees for the observations
            - increment value by 1
            - find average again
            - subtract original from this point partial dependence.
            - plot
        - do this for every feature you're interested in
    - Share of information gain
        - average the information gain for each feature across all trees
        - weighted by the number of samples going through node
    - Permutation test - error caused by losing V
        - compute average response across trees and accuracy
        - shuffle info for a given feature - compute average response across trees and accuracy
        - better to shuffle rather than leave the feature out when using cost function
    - Traffic method - how much traffic is going through this feature
        - calculate proportion of samples visiting feature V in each tree
        - average proportions visited


2018-03-28

Learning Objectives
    1. Discuss model averaging and other ensemble methods
    2. Adaboost
    3. Gradient boosting
    4. Summarize the important points with room for discussion

Combining models
    - Committees - train some number of models and use, for example, the average of the predictions
    - Boosting - a variant on committees, where models are trained in **sequence** and the error function for a given model depends on the previous one
        - sequential makes it difficult to parallelize
    - Decision trees - instead of averaging let the model choice be a function of the input variables
    - Mixture of experts - instead of hard partitioning of the input space we can move to a probabilistic framework for partitioning
    - Model stacking - a method to combine models of different types - http://blog.kaggle.com/2016/12/27/a-kagglers-guide-to-model-stacking-in-practice/

Ensemble learning or committees
    Use multiple predictive models and combine the predictions
    1. Bagging or Bootstrap aggregation [Breiman, 1996]
        - Bootstrap the training data and grow a tree from each bootstrap sample
        - Average the bootstrapped trees → reduces variance
        - Pruning adds bias so do not prune the trees just average them
    2. Random Forests [Breiman et al., 1999]
        - Bagging except we randomly select predictors at each split
        - Decorrelates the trees
    3. Boosting [Freund and Schapire, 1996]
        - Each subsequent tree is grown based on a reweighted version of the
        - training data
        - Decorrelates the trees
        - can use boosting on neural nets as well as decision trees works better when you have weak learners (weak learner would predict coin flip 52% of the time, rather than 99% of the time)
        - trees are actually stumps (have high bias, low variance). - elliot
        - very resilient to overfitting esp compared to neural nets - why? not sure


    ```
    import numpy as np
    from sklearn.model_selection import train_test_split
    data = train_test_split(X, y, test_size=0.20,random_state=1)
    X_train, X_test, y_train, y_test = data
    X_train = X_train[:,np.argsort(X_train.var(axis=0))[::-1][:200]] # select for the 200 datapoints that have the highest variance
    ```

    Deal with class imbalance
    ```
    from sklearn.model_selction import cross_val_score
    clf = RandomForestClassifier(n_estimators=100,
    max_features=’sqrt’,
    random_state=42)
    scores = cross_val_score(clf, X, y, cv=5,scoring=None)
    print("Accuracy: %0.2f (+/- %0.2f)"%(scores.mean(),scores.std()*2))
    scores = cross_val_score(clf, X, y, cv=5,scoring=’f1_weighted’)
    print("F1_weighted: %0.2f (+/- %0.2f)"%(scores.mean(),scores.std()*2))
    ```
    sklearn.model selection.grid search.GridSearchCV
    ```
    from sklearn.model_selection import GridSearchCV
    random_forest_grid = {’max_depth’: [3, None],
    ’max_features’: [’sqrt’, ’log2’, None],
    ’min_samples_split’: [1, 2, 4],
    ’min_samples_leaf’: [1, 2, 4],
    ’bootstrap’: [True, False],
    ’n_estimators’: [20, 40, 60, 80, 100, 120],
    ’random_state’: [42]}
    rf_gridsearch = GridSearchCV(RandomForestClassifier(),
    random_forest_grid,
    n_jobs=-1,verbose=True,
    scoring=’f1_weighted’)
    rf_gridsearch.fit(X_train, y_train)
    print("best parameters:", rf_gridsearch.best_params_)
    ```

Adaboost
- only uses the best models
- weight associated with observations (n)
    - initially 1/numberofobs
    -
- weight associated with model (m)

- optimization algorithms
    - SAMME
    - SAMME.R

2018-03-29 Support Vector Machines
    - additional
Learning Objectives

Support vector machines
    - Find a hyperplane that separates classes in feature space
    - If we cannot do this in a satisfying way:
    1. We introduce the concept of soft margins
    2. We enrich and enlarge the feature space to make separation possible
    Some of the neat aspects of SVMs
        Look for this hyperplane in a direct way
        SVMs are a special instance of kernel machines
        Kernel methods exploit information related to inner products
        a kernel trick helps make computation easy
        SVMs make implicit use of a L1 penalty

- Hyperplane
    - wTx + b  = weights(x) + intercept
        - positive: class = 1
        - negative: class = -1
        - w is a column vector. wT is a row vector np.array([[1,2,3])).shape = 1,3 aka row vector
    - A hyperplane in p dimensions is a flat affine subspace of dimension p − 1
    - An optimal separating hyperplane is a unique solution that separates two classes and maximizes the margin between the classes.
        - f (x) = b + w 1 x 1 + w 2 x 2 . . . w p x p = 0
        - f (x) = w x + b = 0
    - python package: SVM lib
    - points where data hits the margins,

- Linear algebra
    - dot product - angle and magnitude
    - norm: magnitude
    - cosine similarity - angle between the two vectors
    - w is the norm vector aka the vector perpendicular to the hyperplane

- mapping features into higher-dimensional space with a kernel function
    - kernel trick
- Discussion
    Do you think scaling is important?
        Yes, for
    What about class imbalance?
        - majority class will dictate the orientation of the decision boundary (bias toward majority class)
    What do you think a one-vs-rest approach means?
        - three different classes, end up with three different classifiers
    How about one-vs-one (K × K − 1/2)?
    Which kernels are the most flexible? Is flexible always good?
    LinearSVC uses the One-vs-All (also known as One-vs-Rest)
    multiclass reduction
    SVC uses the One-vs-One multiclass reduction.
    LinearSVC minimizes the squared hinge loss
    SVC minimizes the regular hinge loss.


2018-04-02 Dimension reduction PCA and tSNE
 - reading: chapter 10 islr (pp 373 - 399) http://www-bcf.usc.edu/~gareth/ISL/ISLR%20Seventh%20Printing.pdf
 - t-sne article https://www.analyticsvidhya.com/blog/2017/01/t-sne-implementation-r-python/
 - ppt dimension_reduction.pdf

Dimension reduction - can we summarize our data with fewer dimensions?
    - Why?
        - Remove multicolinearity
        - Deal with the curse of dimensionality
        - Remove redundant features
        - Interpretation & visualization
        - Make computations of algorithms easier
        - Identify structure for supervised learning
    - Always start by standardizing the dataset
        1. Center the data for each feature at the mean (so we have mean 0)
        2. Divide by the standard deviation (so we have std 1)
        Fit on train data, transform both train and test
    - Covariance is 1/n(xTx)
        import numpy as np
        from sklearn import preprocessing
        x1 = np.array([[10,10,40,60,70,100,100]]).T
        x2 = np.array([[3,4,7,6,9,7,8]]).T
        X = np.hstack([x1,x2]).astype(float)
        n,d = X.shape
        X = preprocessing.scale(X)
        print(X.mean(axis=0),X.std(axis=0))
        print(1.0/(n) * np.dot(X.T,X))
        print(np.cov(X.T,bias=True))
    - Correlation is cov/stddev
    - Correlation heatmap seaborn good for visualizing correlations between variables
    - PCA
        1. Create the design matrix
        2. Standardize your matrix
        3. Compute
        4. The principal components are the eigenvectors of the covariance matrix
        - The size of each eigenvector’s eigenvalue denotes the amount of variance captured by that eigenvector
        - Ordering the eigenvectors by decreasing corresponding eigenvalues, you get an uncorrelated and orthogonal basis capturing the directions of most-to-least variance in your data
        - Scree plot
            displays the eigenvalues associated with a component or factor in descending order versus the number of the component or factor. You can use scree plots in principal components analysis and factor analysis to visually assess which components or factors explain most of the variability in the data.
    - t-SNE: t-distributed Stochastic Neighbor Embedding
        - It converts similarities between data points to joint probabilities and tries to minimize the Kullback-Leibler divergence between the joint probabilities of the low-dimensional embedding and the high-dimensional data. t-SNE has a cost function that is not convex, i.e. with different initializations we can get different results.
        - advantage is that it's non-linear
        - We may want to use another dimension reduction technique first...
            - If the data are very dense → PCA to 50 dimensions (because in a couple million dimensions, will take forever to calculate)
            - If the data are very sparse → TruncatedSVD to 50 dimensions
        - tsne is pretty computationally intensive so probs want to try PCA first.
    - Explain if and how PCA/tSNE might be used in the following scenerios.
        1. Impressively large matrix of features, but we do not have the knowledge
        to interpret the features (metabolites and disease)
            Yes, can use PCA. can also use tnse (esp if relationships are nonlinear) AFTER PCA priming first. Because tnse doesn't work well in many dimensions.
        2 Impressively large matrix of features, but we really care only about which
        ones are most predictive of outcome?
            No, because PCA turns features into a linear combination of features, so lose interpretability. However, could use PCA to figure out which features are most important (in eigenvectors 1 and 2)
        3 Want to perform clustering on data set with say 100 dimensions?
            have run model, clustered in ALL dimensions and then use PCA to vidualize the results
        4 Want to perform clustering on data set with say 4 dimensions?
            Can use PCA to visualize. Could also use pair plots.
        5 Want to remove correlation among our features in a data set?
            Yes, PCA will remove correlation among features in a data set
        6 Want to investigate our target space or find some structure in your
        targets?
            Yes, that's exactly what the lung cancer example shows. Do we trust the labels that people give us in these disparate data sets?
        7 Want to identify observation outliers?
            Yes
    - Outlier dectection in sklearn
    - SVD
        - M = U*Sigma*vT
            - Sigma are loadings, like eigenvalues
            - sigma values always descending on the diagonal?

2018-04-03 Clustering

Resources
    - lecture repo: https://github.com/gSchool/DSI_Lectures/tree/master/clustering/elliot_cohen
    - Reading - chapter 10 islr
    - Morning pdf - KMeans.pdf
    - https://www.naftaliharris.com/blog/visualizing-k-means-clustering/
    - gap paper http://www.web.stanford.edu/~hastie/Papers/gap.pdf
Supervised learning - predict
    - P(Y|X)
    - minimize some cost function
Unsupervised learning - find patterns
    - density estimation(X)
    - similarity - distance, correlation, cos similarity
Kmeans Clustering
    - Algorithm
        - Initialize k centroids*
        - Until convergence**:
            - assign each data point to the nearest centroid
            - recompute the centroids as the mean of the data points
    - Weights - after standardization, features will be weighted by their variance
    - Centroid Initialization Methods
        1) Randomly choose k points from your data and make those your initial
        centroids (simplest)
        2) Randomly assign each data point to a number 1-k and initialize the kth centroid to the average of the points with the kth label (what happens as N becomes large?)
        3) k-means++ chooses well spread initial centroids. First centroid is chosen at random, with subsequent centroids chosen with probability proportional to the squared distance to the closest existing centroid. (default initialization in sklearn).
    - Choosing k
        - elbow plot - choose k at the elbow
        - maximize gap statistic (this diff on the elbow plot between)
        -
    - Stopping Criteria
        We can update for:
        1) A specified number of iterations (sklearn default : max_iter= 1000)
            - Elliot says kinda dumb but failsafe
        2) Until the centroids don’t change at all
            - this will take a long time
        3) Until the centroids don’t move by very much (sklearn default : tol= .0001)
    - Evaluating model
        - Minimize Intra-Cluster Variance or Within Cluster Variance (WCV)
            - where WCV for the kth cluster is the sum of all the pairwise Euclidean distances
            - also can think of as the sum of the distance from each point to the center.
        - Silhouette score
            - want to be close to my cluster center, far away from other cluster center
            - The Silhouette Coefficient is calculated using the mean intra-cluster distance (a) and the mean nearest-cluster distance (b) for each sample
                (b - a) / max(a, b)
                *only defined for 2 <= k < n
            Values range from -1 to 1, with 1 being optimal and -1 being the worst
    - Assumptions:
        - picked "correct" k
        - clusters have equal variance
        - clusters are isotropic (variance spherical)
        - clusters do NOT have to contain the same number of observations
    - Considerations
        - Because the algorithm is initialized randomly, could fall into a local minimum. So bootstrap cross-validate! Pick the one that minimizes the variance.
        - Should standardize features because we're talking about distance Where WCV for the kth cluster is the sum of all the pairwise Euclidean distances
        - in a rectangle with uniform distribution, splitting will always decrease variance
        - sensitive to outliers - can choose kmenoids because that algorithm chooses a point to define the group, rather than the average
        - One hot encoded categorical can overwhelm - look into k-modes
        - Try MiniBatchKMeans for large datasets (finds local minima, so be careful)
    - Exercise
        Use masking to grab the rows of features that correspond to the label of 1. Find the column-wise mean of that subset of features. Compute the euclidean distance from each data point in features to this mean.
            features = np.array([[3,4],[2,2],[5,4],[6,9],[-1,0]])
            labels = np.array([1,0,1,0,1])
            ones = features[labels==1]
            meanz = np.mean(ones, axis=0)
            distancez = np.linalg.norm(features-meanz, axis=1)
    - DBScan Algorithm = density-based scanning
        - Also computes clusters using distance metric, but decides the number of clusters for you
        - With DBScan, your main hyper-parameter is eps, which is the maximum distance between two points that are allowed to be in the same ‘neighborhood’
Hierarchical Algorithm
    - Hierarchical clustering algorithm
        1. Begin with n observations and a measure of dissimilarity (Euclidean dist, cosine similarity, etc.) of all pairs of points, treating each observation as its own
        cluster.*
        2. Fuse the two “clusters” that are most similar. The similarity of these two indicates the height on the dendrogram where the fusion should be recorded
        3. Compute the new pairwise similarities between the remaining clusters,
        4. rinse and repeat
    - Dendrogram
        - shows relationships in trees
        - “Height of fusion” on dendrogram quantifies the separation of clusters
    - Meaures of dissimilarity between groups
        - Single linkage - "Nearest neighbor"
            - Drawback: Chaining -- several clusters may merge together due to just a few close cases.
        - Complete Linkage - "Farthest neighbor"
            - Drawback: Cluster outliers prevent otherwise close clusters from merging.
        - Average Linkage - "Average neighbor"
            - Drawback: Computationally expensive.
    - Measure similarity between things
        - Sets - Jaccard Measure J(A,B) = |A int B]/|A union B|
                        = number of elements that
        - similarity between time series
            - same series translated over a little
            - dynamic time warp
        - strings
            - edit distance - how many edits are needed to transform one string into another

2018-04-04 NLP

Readings
    - [Introduction to NLP] (https://blog.algorithmia.com/introduction-natural-language-processing-nlp/
    - Scikit-learn's [Working with Text Data](http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html) tutorial.
  * Start at the *Loading the 20 newsgroups dataset* section and implement the code all the way through the *Evaluation of the performance on the test set* section.  Investigate code outputs along the way.  This should take you 30 minutes to an hour.  
  * As a bonus, complete the *Parameter tuning using grid search* at the end.
    - [Intro to NLP with spaCy](https://nicschrading.com/project/Intro-to-NLP-with-spaCy/)

NLP - ppt nlp.pdf
    Objectives
        ● Define NLP and describe several use cases
        ● Explain why NLP is hard - Or easy to do (with sklearn) but hard to master
        ● Describe (and be able to execute) a “typical” text processing workflow
        ● Calculate, by hand and in-code, tf matrices and tf-idf matrices
        ● tf-idf in Sklearn
        ● Introduce nltk and spaCy libraries
    NLP vocabulary and concepts
        - corpus: A collection of documents. Usually each row in the corpus is a document. (So each row in your X matrix is usually a document).
        - stop-words: Domain-specific words that are so common that they are not expected to help differentiate documents. sklearn’s stop-words list.
        - tokens: What documents are made of, and the columns in your X matrix.
            - You’d corpus: A collection of documents. Usually each row in the corpus is a document. (So each row in your X matrix is usually a document).
        - stop-words: Domain-specific words that are so common that they are not expected to help differentiate documents.
        - tokens: What documents are made of, and the columns in your X matrix.
            - You’d think tokens are words (and you’re right), but tokens are words that have been either stemmed or lemmatized to their root form: car, cars, car’s, cars’ ⇒ car
        - n-grams: How many tokens constitute a linguistic unit in the analysis?
            ● boy (n-gram = 1), little boy (n-gram=2), little boy blue (n-gram=3)
            ● computational complexity increases sign as n-gram increases
        - bag-of-words: A document is represented numerically by the set of its tokens, not preserving order and near-by tokens but preserving counts (frequency).
    NLP text processing workflow
        1. Lowercase all your text (unless for some words that are Part-Of-Speech (POS) the capitalization you decide is important.)
        2. Strip out miscellaneous spacing and punctuation.
        3. Remove stop words (careful they may be domain or use-case specific).
        4. Stem/Lemmatize the text into tokens.
        5. Convert text to numbers using a bag-of-words model and a term-frequency, inverse document frequency matrix (more later).
        6. Train / cluster your data in a machine learning model.
    tf-idf matrix
        - How to calculate term frequency (tf)
            - tf(t,d) = count of that word / norm of the vector of words in the document
                (can also use total # of words in the document as denom but with the norm version, the results have been found to be more useful)
            - t is the term (token)
            - d is the document
        - How to calculate inverse document frequency (idf)
            - df = |docs containing t|/|docs|
                = num docs containing t/num_docs
            - words with low document frequency are more useful in differentiating between documents
            - idf = log(|docs|/|1+docs containing t|) > makes the large values more important
            - tf-idf = tf * idf for each token in each document
    jupyter notebook: working_with_text_walkthrough.ipynb
    tools
        - nltk - older, less good, esp with details
        - spaCy - better, newer, details
Naive Bayes - ppt naive_bayes.pdf
    - Review of Bayes Rule
    - P(classA|word1,word2,word3)=P(class)*P(word1|classA)*             
                                P(word2|classA)*P(word3|classA)
                    P(class) = #docs in class/#docs
                    P(word1|classA) = #times word1 appeared in docs of class A/
                                    #words total in docs of class A
    - LaPlace Smoothing
        - add 1 to numerator and num_words to the denominator of conditional probability to make it so that the conditional probability will never be 0 and thus make the whole posterior 0.

2018-04-05 Gaussian Mixture Model
    - reading ESLII pp 272-279 (section 8.5)
    - sklearn documentation http://scikit-learn.org/stable/modules/mixture.html
    - morning jupyter notebook ~/Galvanize/mixture-models/ds-em-mixture-models/notebooks/em_intro.ipynb
    - Dirichlet process mixture model



cd into/cloned/fork-repo
git remote add upstream git://github.com/gSchool.git
git fetch upstream


?? gradient descent boosted trees

2018-04-09 Optimization + Gradient Descent
Gradient descent
    - dsi-optimization

    - rmse vs mse - if mean squared error isn't working, try root mean squared because it scales your cost function
    Formulating optimization problems
        ● Step 1 - Define an objective function to minimize or maximize.
            Minimize f(x) = sin(x)
        ● Step 2 - Define decision variables to optimize, e.g. x.
        ● Step 3 - Define constraints to be satisfied.
            Subject to 3 < x
            f(x) ≥ -0.5
    Newsvendor Problem
        ● Choose quantity, q, to minimize expected cost, (C(q, D)),
        given uncertain demand, D.
            C(q, D) = co * max(q - D, 0) + cu * max(D - q, 0)
            (C(q, D)) Subject to q ≥ 0
            c o - overage cost per unit of demand exceeded = cost - salvage
            c u - underage cost per unit of demand not met = price - cost
    Steepest Descent
        Step 1 - Direction
            d k = -∇ x f(x)
        Step 2 - Step - learning rate
            x k+1 = x k + d k
        Step 3 - Repeat
            k = k + 1
        Go to Step 1
    Why bother with an optimization approach when we have a closed form solution?
        - because doesn't work with discrete demand functions, requires continuous
        - cost function can get complicated
        - hard to add in constraints

Gradient Descent
    - Gradient Descent
        http://vis.supstat.com/2013/03/
        gradient-descent-algorithm-with-r/
        0. Choose step size alpha and precision threshold epsilon
        1. Select starting point x(0) , set i = 1
        2. Update x(t) = x(t-1) - alpha* gradient(t-1)
        3. If the % difference in gradient between t-1 and t gets below a certain threshold, stop
            |f(x(t-1))|-|f(x(t))| / |f(x(t-1))| < ‘, return min|f(x(t))| & argmin x(t)
        4. else, return to step 2.
    - select alpha and stopping criterion
        - alpha
            - a <= 1/C
            - adaptive: ((gradient at x_t - gradient at x_t-1) * (x_t - x_t-1))/||(gradient at x_t - gradient at x_t-1)||**2
        - epsilon
            - max num iterations
            - |f(x_(t-1))|-|f(x_t)| / |f(x_(t-1))|
            - |gradient(x)| < x
    - Potential Gradient Descent Drawbacks Solutions
        - Observations contribute equal weight to the gradient
        - Processor (cost function over all rows is expensive)
        - ways to deal with these
            - We could just use only a single data point at each iteration
            - the expected direction of the gradient would stay the same
            - we could also use a batch of data points at each iteration
            - This would address memory/processing limitations. It has been empirically proven to also often converge faster!
            - it does tend to oscillate as it nears the minimum
    - stochastic gradient descent is just gradient descent with a bunch of random starts

2018-04-10 Neural Networks: Multi-layer perceptron
    Resources
        - Neural net homework
    Objectives:
        - Neural net history
        - “Vanilla” neural networks: multilayer perceptrons
            - Parts of a neuron
            - Feed-forward
            - Backpropagation and gradient descent
        - Hyperparameters and Training
        - Keras introduction
        - References
    1957, Frank Rosenblatt invents the Perceptron
        - Initializes the weights* to random values (e.g. -1.0 to 1.0)
        - Weights change during supervised learning according to the delta rule, ~(yi - yp). After a certain number of training passes through the whole training set (a.k.a. the number of epochs) stop changing the weights.
        - Implements a learning rate that affects how quickly weights can change.
        - Adds a bias to the activation function that shifts the location of the activation threshold and allows this location to be a learned quantity (by multiplying it by a weight).
    Usually activation functions are non-linear
    Back-propagation is determining the gradient of your cost function with respect to the weights
    Gradient descent is updating weights using back-propogation and learning rate
    Vanilla neural network
        - activation functinos
            - sigmoid
            - tanh
            - ReLU max(0,x)
                - x<0, derivative is 0
                - x = 0 derivative is undefined cause corner
                - x>0 derivative is 1
            - tanh(x)
            - Karpathy: ReLU > Leaky ReLU > tanh
        - computations
            - Goal: Minimize the error or loss function - RSS (regression), misclassification rate (classification) - by changing the weights in the model.
            - Back propagation - the recursive application of the chain rule back along the computational network, allows the calculation of the local gradients, enabling...
            - Gradient descent to be used to find the changes in the weights required cto minimize the loss function.
        - batch, mini-batch, stochastic gradient descent
            - batch - look at all rows, calculate aggregate error, update weights
                - upside: outliers don't affect as much
                - downside: go through ALL datapoints before change weights ONCE
            - stochastic gradient descent - only see one row of data, calculate error, update weights
                - downside: outliers are super impactful
            - mini-batch - combo of batch and stochastic. see subsets of the data at a time
        - hyperparameters
            - structure - # of hidden layers, number of nodes in each layer.

            - training method: loss function, learning rate, batch size, number of epochs
            - regularization: likely needed!
                - weight decay (L1 and L2)
                - early stopping (stop when error on holdout cross-validation set starts going up again)
                - dropout
            - random seed
        - potential problems
            - vanishing (or exploding) gradient problem

    Autoencoders:
        - self-supervised learning
        - autoencoders encode (embed) the input into a lower dimensional space (loses info!)
        - can be used with many types of data: typical rows & columns, images, sequences
        - word2vec: contextualize word meaning based on what words are around that word.

2018-04-11 Image Processing + Convolutional Neural Network (CNN)

Resources:
    - http://karpathy.github.io/2015/05/21/rnn-effectiveness/
    - http://colah.github.io/posts/2015-08-Understanding-LSTMs/


Image Processing
    - Framework
        - read
            - read an image into a matrix of numbers
            - read in sub-directories of images
            - each sub-dir named after the label
        - resize
            - make image a specified shape
            - not cropping
            - uniform feature matrix downstream
        - transformations
            - color to grayscale
                - luminance of the RGB image. Y = 0.2125 R + 0.7154 G + 0.0721 B
                - Normalize to range from 0 to 1
                - 3D tensor matrix -> 2D matrix
            - denoise
            - vectorize an image
        - featurize
    - Python Libraries
        - Scikit-image
        - OpenCV-python wrapper
Convolutional Neural Network
    - kernel/filter - convolution
        - #pixels x #pixels x depth of image (3 at the begingging for rgb, then # of feature maps passed from last layer)
        - moves with some stride, stride won't work if dimensions don't align with dimensions of image.
        - often zero pad the border to make sure image out of conv layer same size as image into conv layer
        - acts on image to create new one (like blurred) which gets passed into the activation function. That is the new feature map.
        - activation function is useful because adds non-linear behavior
        - relu is particularly good because gradient always 1 or 0 so doesn't blow up.
        - potential issue is output image is smaller than input image
        - 4 hyperparams
            - number filters K (often multiple of 2)
            - spacial extent F
            - stride S
            - amount of zero padding P
    - downsampling
        - form of regularization
        - pooling
            - maxpool, looks at a block, puts a max value in that block
            - often use stride 2
    - often see conv layer, conv layer, pool -> fully connected layer,
    - CIFAR 10





2018-04-12 Recurrent Neural Networks

repo: /home/alex/Galvanize/Neural Nets/dsi-rnn
Karpathy's youtube lecture where he goes through many parts of the dr.seuss code line-by-line: https://www.youtube.com/watch?v=yCC09vCHzF8
Solving LSAT Puzzles with Parsey McParseface and Seq2Seq https://github.com/BillVanderLugt/LSAT
Goodfellow's Deep Learning Book here: https://github.com/janishar/mit-deep-learning-book-pdf
https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf
LSTM recommended by Chris: https://www.youtube.com/watch?v=WCUNPb-5EYI

- recurrent neural nets store weights from the last iteration of the net for the next one
- model architecture description:
    - number of hidden layers
    - how many nodes in each layer
    - activation functions
- Keras has lots of built-in functionality
- Google has a library that's really efficient (seq to seq)
- LSTM - long short-term memory
    - developed to get around problem of exploding gradient
    - input gate, forget gate, output gate - how quickly should we be forgetting these weights
    - well suited where there are time lags of unknown size and bound between important events
    - applications - speech recognition, translation, handwriting

SPRING BREAK 2 NOTES

Spark
- group of machines manages by cluster manager (YARN or Mesos).
- We submit Spark Applications to the cluster managers, which grant resources to applicaitons


- Driver process functions
    - maintain info about Spark Application
    - respond to user's program
    - analyze, distribute, and schedule work across executors

- Executor process functions
    - execute code assigned to it by the driver
    - report the state of the computation back to the drive node

- Executors will run in Spark code but driver can be "driven" from languages through Spark's Language APIs.



AWS

* About AWS (https://aws.amazon.com/about-aws/)
* Getting Started on AWS (https://aws.amazon.com/start-now/)
-- Read two of the 10 minute tutorials:  Launch a Linux VM (EC2), Store and Retrieve a File (S3)
EC2 virtual linux
    connect via command line
    - ssh -i ~/.ssh/MyKeyPair.pem ec2-user@<ip address here>
    - type yes
    terminate instance
    - EC2 Console -> select box next to instance -> Actions dropdown hit terminate
S3
    - go to AWS management console https://us-west-2.console.aws.amazon.com/console/home?region=us-west-2
    - type S3 to get to the S3 management console
    - to download files from S3, click the checkbox next to the file and then select "download" from the pop-up
    - to delete files, click the checkbox next to the file, from the "More" dropdown select delete


2018-04-30

Morning: Using AWS
Afternoon: High-performance python

Using AWS:
    - lecture /home/alex/Galvanize/AWS/me-aws https://github.com/gSchool/DSI_Lectures/tree/master/high-performance-python/miles_erickson
    - EC2 terminology
        Instance: a type of hardware you can rent, e.g., ‘m3.xlarge’ or ‘t2.micro’
        Amazon Machine Image (AMI), an OS you can run on an instance
        Region: a geograpic region, sometimes can't communicate with other regions
        Availability Zone (AZ): a specific subset of a region, often a specific building, such as ‘us-west-2a’
    - EBS provides disk-drive style storage:
    - S3 provides cheap, bulk storage
    - change permissions on files
        - chmod 400 ~/.ssh/aws-master.pem #Set permission on private key to 400
        - chmod 444 ~/.ssh/aws-master.pub
    - set up alias to containyour log in info for EC2
    - transfer files using scp
    - need to move credentials file up to ec2
        - alternative using IAM http://mikeferrier.com/2011/10/27/granting-access-to-a-single-s3-bucket-using-amazon-iam/

High Performance Computing
    - CPU can have multiple cores. Each core is a self-contained processor that can execute programs
    - process- an instance of a computer program that is being executed
    - Multiprocessing in python
        - works well with random forest
     ```python
        from multiprocessing import Pool
        import os
        # Count the number of words in a file
        def word_count(f):
        return len(open(f).read().split())
        # Use a regular loop to count words in files
        def sequential_word_count(lst_of_files):
        return sum([word_count(f) for f in lst_of_files])
        # Use a multiple cores to count words in files
        def parallel_word_count(lst_of_files):
        pool = Pool(processes=4)
        results = pool.map(word_count, lst_of_files)
        return sum(results)```
    - Multi-threading
        - Each process contains one or more threads of execution
        - useful when program has to wait on resources outside of python code - good for hitting an api when you have to wait for the response
    - What? | Library | Cores | Why?
        Parellelism | multiprocessing | multiple | CPU-bound problems

        Concurrency | threading | single | I/O-bound problems
2018-05-01
Spark

Spark binds to port 4040 by default for the user interface (access by 0.0.0.0:4040)
Spark is run in memory and can be up to 100times faster than Hadoop MapReduce, which writes data to disk after every map step.
Spark came out in 2014

Run spark ~/scripts/
    - From jupyter notebook
        - jupyspark
        - open a jupyter notebook and type
            import pyspark as ps
            spark = ps.sql.SparkSession.builder.getOrCreate()
    - From command line
        - Whenever you want to run a python script (called for instance script.py), you would do it by typing localsparksubmit script.py from the command line.

Afternoon Learning Objectives
• Define what an RDD is, by its properties and operations
• Explain MapReduce paradigm and do an example computation
• Explain the different between transformations and actions on an RDD
• Implement the different transformations through use cases
• Explain what persisting/caching an RDD means, and situations where this is useful

RDD - resilient distributed datasets
    - data itself is stored in a distributed manner
    - can recover from errors (node failure)
    - immutable
    - lazily evaluated - can give instructions and won't evaluate until you tell it to
    - cachable - useful if doing something repetitive like machine learning

MapReduce - the general idea and steps
    - Send the computation to the data rather than trying to bring the data to the computation.
    - Computation and communication are handled as (key, value) pairs.
    - In the “map” step, the mapper maps a function on the data that transforms it into (key, value) pairs. A local combiner may be run after the map to aggregate the results in (key, local aggregated values).
    - After the mapping phase is complete, the (key, value) or (key, local aggregated value) results need to be brought together, sorted by key. This is called the “shuffle and sort” step.
    - Results with the same key are all sent to the same MapReduce TaskTracker for aggregation in the reducer. This is the “reduce” step.
    - The final reduced results are communicated to the Client.

Transformation and action
    - Actions force the RDD to act
        - collect
        - count
        - reduce - Reduces RDD using given function
        - take - Return an array with the first n elements of the RDD
        - first - Return the first element in the RDD
        - saveAsTextFile
    - If you don't get an error when you call a transformation:
        - syntax is okay
        - still might not work
    - Transformations
        - .map(func) - 1:1 (a to alpha, b to bravo)
        - .flatmap(func) - 1 to many (split a sentence on whitespace, return a bunch of words, not a list of words, a bunch)
        - .filter(func)
        - .groupByKey()

Example
    data = sc.parallellize([1,2,3])
    flat_data = data.flatMap(lambda x: range(0, x))
    c = flat_data.count()
    r = flat_data.reduce(lambda a,b: a+b)

    flat_data = 0,0,1,0,1,2
    c=6, r=4

Persisting/Caching
    - rdd.cache() uses the default storage level MEMORY_ONLY, while
    - rdd.persist() gives you the option to specify the storage level
        - Random Forest, MUST persist
        - StorageLevel.MEMORY_AND_DISK is what taryn normally uses

Questions: How does it decide where to split
when use where vs filter
What is json?

jupyspark > output.txt will pipe to a text file

terminate spark in a jupyter notebook
spark.stop()


2018-05-02

SQL on Spark
    - python is an imperative language (do it this way), sql is declarative (I want this outcome, please do it)
    - spark dataframe is an RDD + schema
    - nullable
    - Actions
        .show(n) or .head(n)
        .printSchema() gives you the schema of the table (columns and datatypes, like df.info() in pandas)
        .collect() works the same as it does for RDDs, but is ugly. Use show instead!
        Aggregations (.sum(), .count(), .min(), .max(), etc.)
    - transformations
        - .describe() computes statistics for numeric and string columns
        - .sample() and .sampleBy() give you subsets of the data for easier development
    - do it the SQL way - all the regular SQL functions available without import
        df.registerTempTable('df')
        new_df = spark.sql('''
                            SELECT AVG(col), MAX(col4)
                            FROM df
                            WHERE col = some_val
                            GROUP BY col2
                            ''')
    - with dataframe APT, must import SQL functions
        - import pysark.sql.functions as F
    - sparkSQL (types of functions)
        • Mathematical (round, floor/ceil, trig functions, exponents, log, factorial, etc.)
        • Aggregations (count, average, min, max, first, last, collect_set, collect_list, etc.)
        • Datetime manipulations (change timezone, change string/datetime/unix time)
        • Hashing functions
        • String manipulations (concatenations, slicing)
        • Datatype manipulation (array certain columns together, cast to change datatype)
        http://spark.apache.org/docs/latest/api/python/pyspark.sql.html#module-pyspark.sql.functions
    -
