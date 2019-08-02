"""

https://rpubs.com/danilomartinezdatascientist/384354
https://github.com/clumdee/Python-and-Spark-for-Big-Data-master

"""
import pyspark.sql.functions
from pyspark.sql import SparkSession

#Create a Spark Session
SpSession = SparkSession \
    .builder \
    .master("local") \
    .appName("py_spark") \
    .getOrCreate()

#Load the CSV file into a RDD
SpContext = SpSession.sparkContext
df = SpContext.textFile("Bank/UniversalBank.csv")
df.cache()
df.take(3)

#Remove the first line (contains headers)
datalines=df.filter(lambda x: 'Age' not in x)
datalines.count()

#Cleanup data
# drop ID and Zip Code
from pyspark.sql import Row
#Create a Data Frame from the data
parts = datalines.map(lambda l: l.split(","))
bankMap = parts.map(lambda p: Row(Age=int(p[1]), \
                                Experience=int(p[2]), \
                                Income=float(p[3]), \
                                Family=int(p[5]), \
                                CCAvg=float(p[6]), \
                                Education=int(p[7]), \
                                Mortgage=int(p[8]), \
                                PersonalLoan=int(p[9]), \
                                SecuritiesAccount=int(p[10]), \
                                CDAccount=int(p[11]), \
                                online=int(p[12]), \
                                CreaditCard=int(p[13])  ))

# Infer the schema, and register the DataFrame as a table.
bankDf = SpSession.createDataFrame(bankMap)
bankDf.cache()
bankDf.describe().printSchema()

#######################Data exploratory########################
# Drop any row that contains missing data
print 'count non-missing data:'
print bankDf.na.drop().count()

print 'describe Income and Mortgage: \n'
bankDf.select("Income","Mortgage").describe().show()

print 'Correlation of Income and Moartage? '
from pyspark.sql.functions import corr
bankDf.select(corr("Income","Mortgage")).show()

print 'Correlation of Age and Experience? '
from pyspark.sql.functions import corr
bankDf.select(corr("Age","Experience")).show()


print 'mean of Income'
from pyspark.sql.functions import mean
bankDf.select(mean("Income")).show()

print 'Min and Max of Income'
from pyspark.sql.functions import max,min
bankDf.select(max("Income"),min("Income")).show()

print 'how many people have income less than 40k?'
from pyspark.sql.functions import count
result = bankDf.filter(bankDf['Income'] < 40)
result.select(count('Income')).show()


print 'what is % of people that have income less than 40k?'

print (bankDf.filter(bankDf["Income"]<40).count()*1.0/bankDf.count())*100

print 'what is % of people that have income more than 100k?'

print (bankDf.filter(bankDf["Income"]>100).count()*1.0/bankDf.count())*100


bankDf.groupBy("Experience").count().show(200)
bankDf.groupBy("Age").count().show(100)  #we can replace count with any agg function

grouped = bankDf.groupBy("Experience")
Exp_Mor=grouped.agg({"Mortgage":'mean'})
Exp_Mor=Exp_Mor.show(200)

"""
#not working
#Exp_Mor.coalesce(1).write.format('csv').save("hdfs:///user/nzarifi/Bank/myresults.csv", header='true')
#Exp_Mor.write.csv('test.csv')
"""
print 'people under 25 years old that could receive over 200k loan'
bankDf.filter( (bankDf["Age"] < 25) & (bankDf['Mortgage'] > 200) ).show()


print 'people with experience below 2 that could receive over 200k loan'
bankDf.filter( (bankDf["Experience"] < 2) & (bankDf['Mortgage'] > 200) ).show()

print 'number of people with high experience rejected to get loan: '
print bankDf.filter( (bankDf["Experience"] > 35) & (bankDf['Mortgage'] ==0) ).count()
print 'number of people with high experience received over 200k: '
print bankDf.filter( (bankDf["Experience"] > 35) & (bankDf['Mortgage'] >200) ).count()

print 'number of people who received over 200k: '
print bankDf.filter( (bankDf['Mortgage'] >200) ).count()

print 'Education and Mortgage'
grouped = bankDf.groupBy("Education")
grouped.agg({"Mortgage":'mean'}).show()



#####################################################################
#Find correlation between predictors and target
for i in bankDf.columns:
    if not( isinstance(bankDf.select(i).take(1)[0][0], unicode)) :
        print( "Correlation to Mortgage for ", i, bankDf.stat.corr('Mortgage',i))
print 'There is no significant correlation between mortgage and other features,\n'

print 'describe Experience and Age: \n'
bankDf.select("Experience","Age").describe().show()

#Find correlation between predictors and target
for i in bankDf.columns:
    if not( isinstance(bankDf.select(i).take(1)[0][0], unicode)) :
        print( "Correlation to Experience for ", i, bankDf.stat.corr('Experience',i))


#####################################################################



#Transform to a Data Frame for input to Machine Learing
#Drop columns that are not required (low correlation)

from pyspark.ml.linalg import Vectors
def transformToLabeledPoint(row) :
    lp = ( row["Experience"], Vectors.dense([row["CCAvg"],\
                        row["Education"], \
                        row["Age"], \
                        row["Family"], \
                        row["Income"]]))
    return lp

bankLp = bankMap.map(transformToLabeledPoint)
bankDF = SpSession.createDataFrame(bankLp,["label", "features"])
print "label and features:"
bankDF.select("label","features").show(10)



"""--------------------------------------------------------------------------
Perform Machine Learning
-------------------------------------------------------------------------"""
#Split into training and testing data
(trainingData, testData) = bankDF.randomSplit([0.9, 0.1])
trainingData.count()
testData.count()

#Build the model on training data
from pyspark.ml.regression import LinearRegression
lr = LinearRegression(maxIter=10)
lrModel = lr.fit(trainingData)

#Print the metrics
print("Coefficients: " + str(lrModel.coefficients))
print("Intercept: " + str(lrModel.intercept))

#Predict on the test data
predictions = lrModel.transform(testData)
##predictions.select("prediction","label","features").show()

import pyspark.sql.functions as F
predictions.select(F.bround('prediction', 0).alias('r'),"label","features").show()



#Find R2 for Linear Regression
from pyspark.ml.evaluation import RegressionEvaluator
r2 = RegressionEvaluator(predictionCol="prediction", \
                 labelCol="label",metricName="r2")
R2=r2.evaluate(predictions)


mse = RegressionEvaluator(predictionCol="prediction", \
                 labelCol="label",metricName="mse")
MSE= mse.evaluate(predictions)

rmse = RegressionEvaluator(predictionCol="prediction", \
                 labelCol="label",metricName="rmse")
RMSE= rmse.evaluate(predictions)

mae = RegressionEvaluator(predictionCol="prediction", \
                 labelCol="label",metricName="mae")
MAE= mae.evaluate(predictions)

print("R2_Values: " + str(R2))
print("MSE_Values: " + str(MSE))
print("RMSE_Values: " + str(RMSE))
print("MAE_Values: " + str(MAE))



######################---------------------------------
#Evaluate accuracy worth to try!
############################-----------------------------------------
print 'Since the target only has integer numbers let s check its accuracy just like classification!!!'
##update prediction column to rounded integer number otherwise the accuracy comes Zero!

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

updated_pred=predictions.withColumn("prediction", F.bround('prediction', 0).alias('r'))
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", \
                    labelCol="label",metricName="accuracy")

acc=evaluator.evaluate(updated_pred)
print ("Accuracy of LR_regressor: " + str(acc))




#################RandomForestRegressor#############################
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import VectorIndexer

RF = RandomForestRegressor()
RFModel = RF.fit(trainingData)

#Predict on the test data
predictions = RFModel.transform(testData)


import pyspark.sql.functions as F
predictions.select(F.bround('prediction', 0).alias('r'),"label","features").show()



#Find R2 for Linear Regression
from pyspark.ml.evaluation import RegressionEvaluator
r2 = RegressionEvaluator(predictionCol="prediction", \
                 labelCol="label",metricName="r2")
R2=r2.evaluate(predictions)


mse = RegressionEvaluator(predictionCol="prediction", \
                 labelCol="label",metricName="mse")
MSE= mse.evaluate(predictions)

rmse = RegressionEvaluator(predictionCol="prediction", \
                 labelCol="label",metricName="rmse")
RMSE= rmse.evaluate(predictions)

mae = RegressionEvaluator(predictionCol="prediction", \
                 labelCol="label",metricName="mae")
MAE= mae.evaluate(predictions)

print("RF_R2_Values: " + str(R2))
print("RF_MSE_Values: " + str(MSE))
print("RF_RMSE_Values: " + str(RMSE))
print("RF_MAE_Values: " + str(MAE))





print("RF_R2_Values: " + str(R2))
import pyspark.sql.functions as F
predictions.select(F.bround('prediction', 0).alias('r'),"label","features").show()

print 'Adding new column:rounded_prediction'
predictions.withColumn("rounded_prediction", F.bround('prediction', 0).alias('r')).show()


#Evaluate accuracy
############################
print 'Since the target only has integer numbers let s check its accuracy just like classification!!!'
##update prediction column to rounded integer number otherwise the accuracy comes Zero!

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

updated_pred=predictions.withColumn("prediction", F.bround('prediction', 0).alias('r'))

evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", \
                    labelCol="label",metricName="accuracy")
predictions.withColumn("prediction", F.bround('prediction', 0).alias('r')).show()
acc=evaluator.evaluate(updated_pred)
print ("Accuracy of RF_regressor: " + str(acc))

