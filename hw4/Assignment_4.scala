import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.feature.VectorAssembler
import spark.implicits._
import org.apache.spark.sql.types._

val data = spark.read.format("csv").option("header", "true").load("country_vaccinations.csv")
val usData = data.filter(data("country") === "United States").withColumn("people_vaccinated", 'people_vaccinated.cast(DoubleType)).withColumn("date", unix_timestamp($"date", "yyyy-MM-dd"))
//usData.printSchema

val model_data = usData.select($"people_vaccinated".alias("label"), $"date")
//training.printSchema

val assembler = new VectorAssembler().setInputCols(Array("date")).setOutputCol("features")

val vectorized_data = assembler.transform(model_data)

val nulls_replaced = vectorized_data.na.fill(0.0,Array("label"))
val training = nulls_replaced.select("label", "features")

val lr = new LinearRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8)

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
println(s"r2: ${trainingSummary.r2}")

val predictions = trainingSummary.predictions
predictions.show()

val preds_expanded = predictions.
                    withColumn("xyProd",col("label")*col("prediction")).
                    withColumn("label_sqrd",col("label")*col("label"))
preds_expanded.show()

val act_sum = preds_expanded.select(col("label")).rdd.map(_(0).asInstanceOf[Double]).reduce(_+_)
val pred_sum = preds_expanded.select("prediction").rdd.map(_(0).asInstanceOf[Double]).reduce(_+_)
val xyProd_sum = preds_expanded.select("xyProd").rdd.map(_(0).asInstanceOf[Double]).reduce(_+_)
val xsqrd_sum= preds_expanded.select("label_sqrd").rdd.map(_(0).asInstanceOf[Double]).reduce(_+_)
val num_rows = preds_expanded.count()
val act_sum_sqrd = act_sum*act_sum
val act_avg = act_sum/num_rows
val pred_avg = pred_sum/num_rows

val mse = trainingSummary.rootMeanSquaredError
val B1 = (xyProd_sum - ((act_sum*pred_sum)/num_rows))/(xsqrd_sum-(act_sum_sqrd/num_rows))
val B0 = pred_avg - (B1*act_avg)

println()
println(s"MSE: ${mse}")
println(s"B1: ${B1}")
println(s"B0: ${B0}")
