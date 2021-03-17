import org.apache.spark.mllib.linalg._ 
import org.apache.spark.mllib.regression._
val rawData = sc.textFile("covtype.data")
val data = rawData.map{ line => 
val values = line.split(',').map(_.toDouble)
val featureVector = Vectors.dense(values.init)
val label = values.last - 1
LabeledPoint(label, featureVector)
}

val Array(trainData, cvData, testData) = data.randomSplit(Array(0.8, 0.1, 0.1))
trainData.cache()
cvData.cache()
testData.cache()
import org.apache.spark.mllib.evaluation._
import org.apache.spark.mllib.tree._
import org.apache.spark.mllib.tree.model._
import org.apache.spark.rdd._

def getMetrics(model: DecisionTreeModel, data: RDD[LabeledPoint]):
MulticlassMetrics = {
val predictionAndLabels = data.map(example => 
(model.predict(example.features), example.label)
)
new MulticlassMetrics(predictionAndLabels)
}
val model = DecisionTree.trainClassifier(
trainData, 7, Map[Int,Int](), "gini", 4, 100)
val metrics = getMetrics(model, cvData)

println()
println("Labels:")
for (i <- 0 to 6) {println("Index: " + i + " --- " + metrics.labels(i))}
println()
println("Precision:")
for (i <- 0 to metrics.labels.length - 1) {println("Index: " + i + " --- " + metrics.precision(metrics.labels(i)))}
println()
println("Recall:")
for (i <- 0 to metrics.labels.length - 1) {println("Index: " + i + " --- " + metrics.recall(metrics.labels(i)))}
println()
println("Accuracy:")
metrics.accuracy
println()
println("Confusion Matrix:")
metrics.confusionMatrix