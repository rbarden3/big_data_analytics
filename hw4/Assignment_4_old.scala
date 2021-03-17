```
import org.apache.spark.ml.regression.LinearRegression

val schema = StructType.fromDDL(country: String, iso_code: String, date: String,	total_vaccinations: Integer,	people_vaccinated: Integer,	people_fully_vaccinated: Integer,	daily_vaccinations_raw: Integer,	daily_vaccinations: Integer,	total_vaccinations_per_hundred: Integer,	people_vaccinated_per_hundred: Integer,	people_fully_vaccinated_per_hundred: Integer,	daily_vaccinations_per_million: Integer,	vaccines: String,	source_name	source_website: String)

val training = spark.read.option("header",true)
   .csv("country_vaccinations.csv")

val lr = new LinearRegression()
  .setMaxIter(10)
  .setRegParam(0.3)
  .setElasticNetParam(0.8)

// Fit the model
val lrModel = lr.fit(training)

// Print the coefficients and intercept for linear regression
println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")

// Summarize the model over the training set and print out some metrics
val trainingSummary = lrModel.summary
println(s"numIterations: ${trainingSummary.totalIterations}")
println(s"objectiveHistory: [${trainingSummary.objectiveHistory.mkString(",")}]")
trainingSummary.residuals.show()
println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
println(s"r2: ${trainingSummary.r2}")```