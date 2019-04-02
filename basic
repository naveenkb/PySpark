
# coding: utf-8

# In[1]:

# To run pyspark in jupyter notebook
import findspark
findspark.init()


# In[2]:

from pyspark import SparkConf


# In[3]:

# Setting pyspark config.
conf = SparkConf().setAll([('spark.executor.memory', '8g'), ('spark.driver.memory','1g')])


# In[4]:

import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import *


# In[5]:

#SparkSession: The entry point to programming Spark with the Dataset and DataFrame API.
spark = SparkSession.builder.appName('tutorial').config(conf=conf).getOrCreate()


# In[6]:

# checking spark version
spark.version


# In[7]:

# reading flat files
transactions = spark.read.csv("transaction.csv",inferSchema=True,header=True)


# In[8]:

transactions


# In[9]:

# print the dataframe
transactions.show()


# In[10]:

# print head of the data frame
transactions.head(2)


# In[11]:

# defining schema for a dataframe
from pyspark.sql.types import StructField,StringType,IntegerType,BooleanType,StructType


# In[12]:

schema = [StructField("customer_id",StringType(),True),
          StructField("product_id",IntegerType(),True),
          StructField("quantity",IntegerType(),True)]
schema = StructType(schema)


# In[13]:

# read the same file again
transactions = spark.read.csv("transaction.csv",inferSchema=True,header=True,schema=schema)


# In[14]:

# printing the schema of the file
transactions.printSchema()


# In[15]:

# create a pyspark dataframe
from pyspark.sql import Row


# In[16]:

df = spark.createDataFrame([Row(123,456),Row(234,567)],['col_a','col_b'])


# In[17]:

df.show()


# In[19]:

# write back to csv
#df.write.csv('df.csv')


# In[22]:

#df.coalesce(1).write.csv('df.csv',header=True)


# In[23]:

# convert to pandas dataframe


# In[24]:

df_pd = df.toPandas()


# In[25]:

df_pd


# In[26]:

# back to transactions data


# In[27]:

transactions.show()


# In[28]:

# selecting only a few columns from transaction table

transactions.select(['customer_id','product_id']).show()


# In[29]:

transactions.select('customer_id','product_id').show()


# In[30]:

# adding a new column to the transaction table


# In[31]:

transactions.withColumn("quantity_plus_1",transactions['quantity']+1).show()


# In[32]:

from pyspark.sql.functions import lit


# In[33]:

transactions = transactions.withColumn("is_sales_more_than_3",lit(True))


# In[34]:

transactions = transactions.withColumn("is_sales_more_than_3",when(transactions.quantity<=3,False).otherwise(True))


# In[35]:

transactions.show()


# In[36]:

# remane a column


# In[37]:

transactions.withColumnRenamed('is_sales_more_than_3','sales_more_than_3').show()


# In[38]:

# filter on a condition 


# In[39]:

transactions.filter(transactions['quantity']<3).show()


# In[40]:

# filter on multiple conditions


# In[41]:

transactions.filter((transactions['quantity']<3)&(transactions['customer_id']=='bcd')).show()


# In[42]:

transactions.show()


# In[71]:

# aggregation operations 
from pyspark.sql.functions import *


# In[75]:

# mean
transactions.agg({'quantity':'mean'}).collect()


# In[76]:

# count
transactions.agg({'quantity':'count'}).collect()


# In[80]:

# median
transactions.approxQuantile('quantity',[0.5],0.01)


# In[54]:

transactions.select(mean('quantity')).show()


# In[56]:

transactions.describe().show()


# In[59]:

transactions.select('quantity').describe().show()


# In[62]:

# groupby


# In[64]:

transactions.show()


# In[68]:

transactions.groupby('customer_id').agg({'quantity':'mean'}).show()


# In[69]:

# groupby on multiple columns
transactions.groupby(['customer_id','product_id']).agg({'quantity':'mean'}).show()


# In[81]:

# join operations


# In[82]:

# reading the customers file


# In[89]:

customer_df = spark.read.csv('customers.csv',inferSchema=True,header=True)


# In[91]:

customer_df.printSchema()


# In[95]:

transactions.join(customer_df,on='customer_id',how='right').show()


# In[96]:

# spark sql


# In[98]:

transactions.show()


# In[99]:

# Register the DataFrame as a SQL temporary view
transactions.createOrReplaceTempView("trans")


# In[103]:

spark.sql("select * from trans where quantity >=2").show()


# # Hands on

#  1. Find the average quantity bought by each gender.
#  2. Find the maximum quantity (and product) bought by every customer.

# In[ ]:



