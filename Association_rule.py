import pandas as pd
import os
from pyspark.ml.fpm import FPGrowth
from pyspark.sql import functions as F


os.chdir('/data/2/gcgmyasb/data/work/Citiphones/IVR_2/Feat_Engg')

d = pd.read_csv('ivr_calls_in_oct_nov_with_prev_events_r_new.csv', nrows = 10000)

df = spark.createDataFrame(d)
# get a warning that scheduler.TaskSetManager: Stage 0 contains a task of very large size (369 KB). The maximum recommended task size is 100 KB.
# Need to understand more about that 

cust_event_set = df.groupby('customerid').agg(F.collect_set('prev_event'))

data = cust_event_set.select('customerid', col('collect_set(prev_event)').alias('event_set'))

fpGrowth = FPGrowth(itemsCol="event_set", minSupport=0.2, minConfidence=0.6)
model = fpGrowth.fit(data)

# Display frequent itemsets.
model.freqItemsets.show()

# Display generated association rules.
model.associationRules.show()

# transform examines the input items against all the association rules and summarize the
# consequents as prediction
model.transform(df).show()
