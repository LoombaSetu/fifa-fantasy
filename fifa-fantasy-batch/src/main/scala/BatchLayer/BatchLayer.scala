import org.apache.spark._
import org.apache.spark.SparkContext._
import org.apache.spark.sql.functions.{col, _}
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.regression.DecisionTreeRegressor
import org.apache.spark.ml.evaluation.RegressionEvaluator

object scala {
  def main(args: Array[String]) {
    val conf = new SparkConf()
      .setAppName("Build FIFA Players Table")
      .setMaster("local[2]")


    val spark = SparkSession.builder()
      .appName("FIFA Players Table")
      .config("spark.hadoop.metastore.catalog.default", "hive")
      .enableHiveSupport()
      .getOrCreate()

    val PlayerData: DataFrame = spark.read.format("csv")
      .option("header", "true")
      .load("hdfs:///setu-project/male_players.csv")


//    val longPlayerData = PlayerData.selectExpr(
//      "player_id", "fifa_version", "short_name", "long_name", "player_positions", "height_cm",
//      "nationality_name", "preferred_foot",
//      "stack(9, 'overall', overall, 'pace', pace, 'shooting', shooting, 'passing', passing, 'dribbling', dribbling," +
//        " 'defending', defending, 'value_eur', value_eur, 'wage_eur', wage_eur, 'physic', physic) as (Attribute, Value)"
//    )

    // Spark ML bit

    val selectedFeatures = Array(
      "pace", "shooting", "passing", "dribbling", "defending", "physic",
      "height_cm", "nationality_name", "preferred_foot", "player_positions"
    )

    // Select relevant features and target variables
    val data = PlayerData.select((selectedFeatures.map(col) :+ col("overall") :+ col("value_eur") :+ col("wage_eur")): _*)


    // StringIndexer for categorical variables
    val categoricalCols = Array("nationality_name", "preferred_foot", "player_positions")
    val indexers = categoricalCols.map(col => new StringIndexer().setInputCol(col).setOutputCol(col + "_index"))

    // Assemble features into a vector
    val assembler = new VectorAssembler()
      .setInputCols((selectedFeatures ++ categoricalCols.map(_ + "_index")).distinct)
      .setOutputCol("features")

    // Decision Tree Regression Model
    val dt = new DecisionTreeRegressor()
      .setLabelCol("overall") // Change this to "value_eur" or "wage_eur" for different target variables
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

    // Evaluate the model
    val evaluator = new RegressionEvaluator()
      .setLabelCol("overall") // Change this to "value_eur" or "wage_eur" for different target variables
      .setPredictionCol("prediction")
      .setMetricName("rmse")

    val rmse = evaluator.evaluate(predictions)
    println(s"Root Mean Squared Error (RMSE) on test set: $rmse")

    // Save the model
    model.write.overwrite().save("hdfs:///setu-project/")



  }

}
