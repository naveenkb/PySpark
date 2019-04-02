
# coding: utf-8

# # ML Through PySoark - Case Study - Loan Default Prediction

# #### Read train data

# In[ ]:

from pyspark import SQLContext
from pyspark.sql.functions import col, lower, regexp_replace, split
train_data = SQLContext(sc).read.csv('/data/publicdata/train.csv',header=True,inferSchema=True)
test_data = SQLContext(sc).read.csv('/data/publicdata/test.csv',header=True,inferSchema=True)


# Basic Data Exploration

# In[ ]:

train_data.printSchema()


# In[ ]:

test_data.printSchema()


# Train and Test Data Set Dimensions

# In[ ]:

def df_shape(df):
    return (df.count(),len(df.columns))
print("Train Data Dimensions:",df_shape(train_data))
print("Test Data Dimensions:",df_shape(test_data))


# Identify cat columns and num columns in Training Data and Test Data

# In[ ]:

def cat_num_cols_identify(df):
    cat_cols=[]
    num_cols=[]
    for i in range(0,len(df.dtypes)):
        if(df.dtypes[i][1]=='string'):
            cat_cols = cat_cols+[df.dtypes[i][0]]
        else:
            num_cols = num_cols+[df.dtypes[i][0]]
    return cat_cols,num_cols


# In[ ]:

train_cat_cols,train_num_cols = cat_num_cols_identify(train_data)
print("Train cat columns are:/n",train_cat_cols)
print("Train num columns are:",train_num_cols)
test_cat_cols,test_num_cols = cat_num_cols_identify(test_data)
print("Test cat columns are:",test_cat_cols)
print("Test num columns are:",test_num_cols)


# Keep only digits in "Years in current job" column 

# In[ ]:

train_data = train_data.withColumn("Years in current job",regexp_replace("Years in current job","\D+",""))
test_data = test_data.withColumn("Years in current job",regexp_replace("Years in current job","\D+",""))


# In[ ]:

train_data.printSchema()


# In[ ]:

test_data.printSchema()


# Convert All Int to Float type

# In[ ]:

int_or_cat_to_float_cols = ['Years in current job','Months since last delinquent','Bankruptcies','Tax Liens',
                           'Current Loan Amount','Credit Score','Annual Income','Number of Open Accounts',
                           'Number of Credit Problems','Current Credit Balance','Maximum Open Credit']
for c in int_or_cat_to_float_cols:
    train_data = train_data.withColumn(c,col(c).cast("float"))
    test_data = test_data.withColumn(c,col(c).cast("float"))


# In[ ]:

train_data.printSchema()


# In[ ]:

test_data.printSchema()


# All the columns are in correct data type - Let's Proceed with Data Quality checks

# In[ ]:

train_cat_cols,train_num_cols = cat_num_cols_identify(train_data)
print("Train cat columns are:/n",train_cat_cols)
print("Train num columns are:",train_num_cols)
test_cat_cols,test_num_cols = cat_num_cols_identify(test_data)
print("Test cat columns are:",test_cat_cols)
print("Test num columns are:",test_num_cols)


# Missing Value Analysis

# In[ ]:

def missing_values_count(df,cat_cols,num_cols):
    from pyspark.sql.functions import isnan
    num_missing_values = [df.filter((df[column]=="")|df[column].isNull()|isnan(df[column])).count() for column in num_cols]
    num_miss_dict = dict(zip(num_cols,num_missing_values))
    cat_missing_values = [df.filter((df[column]=="")|df[column].isNull()|isnan(df[column])).count() for column in cat_cols]
    cat_miss_dict = dict(zip(cat_cols,cat_missing_values))
    return cat_miss_dict,num_miss_dict


# Train Data Missing values

# In[7]:

train_cat_missing,train_num_missing = missing_values_count(train_data,train_cat_cols,train_num_cols)
print("Train Cat Missing values are:",train_cat_missing)
print("Train Num Missing values are:",train_num_missing)


# Look at the few missing values in train data

# In[ ]:

train_data.filter(train_data['Loan Status'].isNull()).show(n=5)


# Entire row is Null Values so removing the rows in which all rows are null

# In[ ]:

train_data = train_data.filter(train_data['Loan Status'].isNotNull())


# Let's observe the missing values again

# In[ ]:

train_cat_missing,train_num_missing = missing_values_count(train_data,train_cat_cols,train_num_cols)
print("Train Cat Missing values are:",train_cat_missing)
print("Train Num Missing values are:",train_num_missing)


# Test Data Missing Values

# In[ ]:

test_cat_missing,test_num_missing = missing_values_count(test_data,test_cat_cols,test_num_cols)
print("Train Cat Missing values are:",train_cat_missing)
print("Train Num Missing values are:",train_num_missing)


# In[ ]:

test_data.filter(test_data['Loan ID'].isNull())


# Loan ID is null and remaining columns are also null

# In[9]:

test_data = test_data.filter(test_data['Loan ID'].isNotNull())


# Calculating missing values after removing null values

# In[ ]:

test_cat_missing,test_num_missing = missing_values_count(test_data,test_cat_cols,test_num_cols)
print("Test Cat Missing values are:",test_cat_missing)
print("Test Num Missing values are:",test_num_missing)


# # Missing Value Imputation

# In[ ]:

from pyspark.ml.feature import Imputer
imputer = Imputer()
missingcols=['Bankruptcies','Tax Liens','Credit Score','Months since last delinquent','Years in current job','Annual Income','Maximum Open Credit']
imputer.setParams(strategy="median", missingValue=float("nan"),inputCols=missingcols,outputCols=missingcols)
imputer_fit = imputer.fit(train_data)
train_data=imputer_fit.transform(train_data)


# ### Use the same imputer object to fit on test data

# In[ ]:

test_data=imputer_fit.transform(test_data)


# Check the Missing Values again

# In[10]:

train_cat_missing,train_num_missing = missing_values_count(train_data,train_cat_cols,train_num_cols)
print("Train Cat Missing values are:",train_cat_missing)
print("Train Num Missing values are:",train_num_missing)
test_cat_missing,test_num_missing = missing_values_count(test_data,test_cat_cols,test_num_cols)
print("Test Cat Missing values are:",test_cat_missing)
print("Test Num Missing values are:",test_num_missing)


# There are no missing values so we can proceed with the furthur steps in the modelling

# # Feature Engineering

# Feature creation - One hot encoding of Categorical variables

# In[ ]:

# Target variable encoding
from pyspark.sql.functions import when
train_data=train_data.withColumn('label',when(col('Loan Status')=='Fully Paid',0).otherwise(1))


# In[ ]:

# One hot encoding
cat_cols = ['Term', 'Home Ownership', 'Purpose']
from pyspark.ml.feature import OneHotEncoderEstimator, StringIndexer, VectorAssembler
train_data.groupby('Home Ownership').count().show()
stages = []
for categoricalCol in cat_cols:
    stringIndexer = StringIndexer(inputCol = categoricalCol, outputCol = categoricalCol + 'Index')
    encoder = OneHotEncoderEstimator(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"])
    stages += [stringIndexer, encoder]
assemblerInputs = [c + "classVec" for c in cat_cols] + train_num_cols
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
stages += [assembler]


# In[ ]:

# Pipe line for running the stages
from pyspark.ml import Pipeline
cols = train_data.columns
pipeline = Pipeline(stages = stages)
pipelineModel = pipeline.fit(train_data)
train_data = pipelineModel.transform(train_data)
test_data=pipelineModel.transform(test_data)
train_data.select('features').show()


# Split to train and validation datasets

# In[ ]:

train,val_df = train_data.randomSplit([0.8,0.2],seed=42)
print("Train Data Rows:",train.count())
print("Validation Data Rows:",val_df.count())


# # Logistic Regression

# Fit Logistic Regression on the train data

# In[ ]:

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
LR = LogisticRegression(featuresCol='features',labelCol='label',maxIter=20)
LR_Model = LR.fit(train)
evaluator = BinaryClassificationEvaluator()
print("Training AUC is:",evaluator.evaluate(LR_Model.transform(train),{evaluator.metricName:"areaUnderROC"}))


# Evalute Logistic Regression on the vlaidation data set

# In[ ]:

prediction_LR = LR_Model.transform(val_df)
print("Validation AUC is :",evaluator.evaluate(prediction_LR,{evaluator.metricName:"areaUnderROC"})


# # Random Forest Classifier

# Fit Random Forest classfier

# In[ ]:

from pyspark.ml.classification import RandomForestClassifier
rf=RandomForestClassifier(featuresCol='features',labelCol="label", numTrees=300, maxDepth=6,seed=42)
rf_model = rf.fit(train)
print("Train AUC is:")
evaluator.evaluate(rf_model.transform(train),{evaluator.metricName:"areaUnderROC"})


# Evaluate Random Forest classfier on the validation data set

# In[ ]:

prediction_RF = rf_model.transform(val_df)
print("Validation AUC is:")
evaluator.evaluate(prediction_RF,{evaluator.metricName:"areaUnderROC"})


# # Gradient Boosting Trees

# Fit Gradient Boosting Trees

# In[ ]:

from pyspark.ml.classification import GBTClassifier
gbt=GBTClassifier(featuresCol='features',labelCol="label", maxIter=300, maxDepth=7,seed=42)
gbt_model = gbt.fit(train)
print("Train AUC:")
evaluator.evaluate(gbt_model.transform(train),{evaluator.metricName:"areaUnderROC"})


# Evaluate GBT on the validation datasets

# In[ ]:

prediction_gbt = gbt_model.transform(val_df)
print("Validation AUC:")
evaluator.evaluate(prediction_gbt,{evaluator.metricName:"areaUnderROC"})


# # Parameter Tuning or Grid search

# Paramet Tuning for logistic regression

# In[ ]:

from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.tuning import CrossValidator
grid = ParamGridBuilder().addGrid(LR.maxIter, [10, 20]).build()
cv = CrossValidator(estimator=LR, estimatorParamMaps=grid, evaluator=evaluator,numFolds=5)
cvModel = cv.fit(train)
print("CV Best Model AUC is:")
evaluator.evaluate(cvModel.transform(val_df),{evaluator.metricName:"areaUnderROC"})


# Parameter Tuning for Random Forest classifier

# In[11]:

grid = ParamGridBuilder().addGrid(rf.numTrees, [100, 200]).addGrid(rf.maxDepth,[5,6,7]).build()
cv = CrossValidator(estimator=rf, estimatorParamMaps=grid, evaluator=evaluator,numFolds=3)
cvModel = cv.fit(train)
evaluator.evaluate(cvModel.transform(val_df),{evaluator.metricName:"areaUnderROC"})


# In[ ]:



