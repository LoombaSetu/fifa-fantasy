import org.apache.spark._
import org.apache.spark.SparkContext._
import org.apache.spark.sql.functions.{col, _}
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.regression.DecisionTreeRegressor
import org.apache.spark.ml.evaluation.RegressionEvaluator

object ScalaBatchLayer {
  def main(args: Array[String]) {
    val conf = new SparkConf()
      .setAppName("Build FIFA Players Table")
      .setMaster("local[2]")

    val spark = SparkSession.builder()
      .appName("FIFA Players Table")
      .config("spark.hadoop.metastore.catalog.default", "hive")
      .enableHiveSupport()
      .getOrCreate()

    try {
      // Read data from CSV file
      val playerData = spark.sql("SELECT * FROM playerdatahive")
      //val inputPlayerData = Read from Kafka topic

      // Feature selection
      val selectedFeatures = Array(
        "pace", "shooting", "passing", "dribbling", "defending", "physic", "club_name",
        "height_cm", "nationality_name", "preferred_foot", "player_positions"
      )

      // Select relevant features and target variables
      val data = playerData.select((selectedFeatures.map(col) :+ col("value_eur")): _*)

      // StringIndexer for categorical variables
      val categoricalCols = Array("nationality_name", "preferred_foot", "player_positions", "club_name")
      val indexers = categoricalCols.map(col => new StringIndexer().setInputCol(col).setOutputCol(col + "_index"))

      // Assemble features into a vector
      val assembler = new VectorAssembler()
        .setInputCols((selectedFeatures ++ categoricalCols.map(_ + "_index")).distinct)
        .setOutputCol("features")

      // Decision Tree Regression Model
      val dt = new DecisionTreeRegressor()
        .setLabelCol("overall") // Change this to "value_eur" for different target variables
        .setFeaturesCol("features")

      // Create a pipeline with stages
      val pipeline = new Pipeline().setStages(indexers ++ Array(assembler, dt))

      // Split the data into training and testing sets
      val dataArray = data.randomSplit(Array(0.8, 0.2))
      val Array(trainingData, testData) = dataArray

      // Train the model
      val model = pipeline.fit(trainingData)

      // Make predictions on the test set
      val predictions = model.transform(testData)
      // val inputPredictions = model.transform(inputPlayerData)

      // Evaluate the model
      val evaluator = new RegressionEvaluator()
        .setLabelCol("value_eur")
        .setPredictionCol("prediction")
        .setMetricName("rmse")

      val rmse = evaluator.evaluate(predictions)
      println(s"Root Mean Squared Error (RMSE) on test set: $rmse")

      // Save the model
      model.write.overwrite().save("hdfs:///setu-project/")

      // Creating views and saving to Hbase via Hive
      spark.sql("insert overwrite table playerdataprocessed " +
        "select concat(short_name, fifa_version)," +
        "  fifa_version, short_name, long_name, height_cm," +
        "  player_positions, nationality_name, age, club_name," +
        "  preferred_foot, overall, pace, shooting, passing, dribbling, defending, physic," +
        "  value_eur from playerdatahive;")


    } catch {
      case e: Exception =>
        // Log or handle the exception
        e.printStackTrace()
    } finally {
      spark.stop()
    }
  }
}
