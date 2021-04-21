
import org.apache.spark.ml.feature.{StopWordsRemover, Tokenizer, CountVectorizer, CountVectorizerModel}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel}
// import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg.SparseVector
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.sql.SparkSession
import spark.implicits._
import org.apache.spark.sql.types._

val custom_schema = StructType(Array( // create custom schema for formatting csv
    StructField("id", IntegerType, false),
    StructField("label", StringType, false),
    StructField("text", StringType, false),
    StructField("label_num", IntegerType, false)
))
// read in data from csv
val data = spark.read.format("csv").schema(custom_schema).option("header", "true").option("multiLine", "true").option("escape","\"").load("spam_ham_dataset.csv")
// val usData = data.filter(data("country") === "United States").withColumn("people_vaccinated", 'people_vaccinated.cast(DoubleType)).withColumn("date", unix_timestamp($"date", "yyyy-MM-dd"))
//usData.printSchema

val input_data = data.select($"text", $"label_num") // select only necessary columns
//training.printSchema

val tokenizer = new Tokenizer().setInputCol("text").setOutputCol("words") // create tokenizer 
val tokenized = tokenizer.transform(input_data) // tokenize data
tokenized.select("text", "words")

val remover = new StopWordsRemover().setInputCol("words").setOutputCol("tokens") // remove unnecessary words (stop words) from  tokens
val removed = remover.transform(tokenized)

// create count vectorizer for vectorization
val counter: CountVectorizerModel = new CountVectorizer().setInputCol("tokens").setOutputCol("features").fit(removed)
val counted = counter.transform(removed) // turn dataset into vectors
counted.show()

val model_data = counted.select("label_num", "features", "words").withColumn("label_num", 'label_num.cast(DoubleType)) // convert labeled_num to doubles

val Array(train_in, test_in) = model_data.randomSplit(Array(0.7, 0.3)) //split dataset into train and test datasets

// Create training and Test RDDs
val trainingLabeled : RDD[LabeledPoint] = train_in.rdd.map(row => LabeledPoint(row.getAs[Double]("label_num"), 
  SparseVector.fromML(row.getAs[org.apache.spark.ml.linalg.SparseVector]("features"))))
val testLabeled : RDD[LabeledPoint] = test_in.rdd.map(row => LabeledPoint(row.getAs[Double]("label_num"), 
  SparseVector.fromML(row.getAs[org.apache.spark.ml.linalg.SparseVector]("features"))))




//create a sparkSession object for use in converting RDD to Dataset
val sparkSession =  SparkSession.builder().getOrCreate() 

// val trainingSet =  sparkSession.createDataset(training_in) sparkSession.createDataset(testLabeled).show()
// 
val counter_metrics = counted.withColumn("words", explode($"words")).groupBy($"words").agg(count($"words").as("counts"))
val training_metrics = train_in.withColumn("words",  explode($"words")).groupBy($"words", $"label_num").agg(count($"words").as("counts"))
val testing_metrics = test_in.withColumn("words", explode($"words")).groupBy($"words").agg(count($"words").as("counts"))

//compute train data metrics:
val train_len = train_in.count()
val train_spam = train_in.select(col("label_num")).rdd.map(_(0).asInstanceOf[Double]).reduce(_+_)
val train_ham = train_len - train_spam

val p_s = train_spam / train_len
val p_h = train_ham / train_len

// val spam_words = training_metrics.filter("label_num > 0.0")
// spam_words.show()
// val ham_words = training_metrics.filter("label_num < 1.0")
// ham_words.show()
// val spam_words_2 = spam_words.withColumn("p_ws", col("counts")/train_len)
// val ham_words_2 = ham_words.withColumn("p_wh", col("counts")/train_len)
val training_metrics_expanded = training_metrics.withColumn("p_w_hs", col("counts")/train_len)
val training_spam = train_in.filter("label_num > 0.0").withColumn("words",  explode($"words")).groupBy($"words", $"label_num").agg(count($"words").as("counts"))
training_spam.show()
// training_metrics.joinWith(spam_words_2, training_metrics("words") === spam_words_2("words"), "fullouter").map{ case (w, l, c, null) => (w, l, c, 0)}

// val columnDataTypes : Array[String] = model_data.schema.fields.map(x=>x.dataType).map(x=>x.toString)
val words_p = training_metrics_expanded.select("words", "label_num", "p_w_hs").withColumn("logged_p", log($"p_w_hs")).groupBy($"words").agg(sum($"logged_p").as("p_sM"))

// compare p_ws and p_wh to p_sM
training_metrics_expanded.filter('words === "words").show()
words_p.filter('words === "words").show()

// training_metrics_expanded.select("words", "label_num", "p_w_hs").withColumn("logged_p", log($"p_w_hs")).filter('words === "words").show()
// val spam_words_p = training_metrics_expanded.select($"words", $"label_num", $"counts", $"p_w_hs".alias("p_ws")).filter('label_num === 1.0)
// spam_words_p.show()
// val ham_words_p = training_metrics_expanded.select($"words", $"label_num", $"counts", $"p_w_hs".alias("p_wh")).filter('label_num === 0.0)
// ham_words_p.show()

words_p.show()
val model = NaiveBayes.train(trainingLabeled, lambda = 1.0, modelType = "multinomial") // train model
val predict_label = testLabeled.map(p => (model.predict(p.features), p.label)) // group predictions and labels into RDD
val results = sparkSession.createDataset(predict_label).select($"_1".alias("predicted"), $"_2".alias("labels")) // convert predict_label RDD to Dataset

val accuracy = 1.0 * predict_label.filter(x => x._1 == x._2).count() / testLabeled.count() // calculate accuracy
// results.filter(results("labels") =!= results("predicted")).show() // check incorrect predictions
