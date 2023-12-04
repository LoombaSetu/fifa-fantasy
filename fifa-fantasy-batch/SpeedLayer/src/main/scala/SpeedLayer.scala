import org.apache.kafka.common.serialization.StringDeserializer
import org.apache.spark.SparkConf
import org.apache.spark.streaming._
import org.apache.spark.streaming.kafka010.ConsumerStrategies.Subscribe
import org.apache.spark.streaming.kafka010.LocationStrategies.PreferConsistent
import org.apache.spark.streaming.kafka010._
import com.fasterxml.jackson.databind.{ DeserializationFeature, ObjectMapper }
import com.fasterxml.jackson.module.scala.experimental.ScalaObjectMapper
import com.fasterxml.jackson.module.scala.DefaultScalaModule
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.hbase.TableName
import org.apache.hadoop.hbase.HBaseConfiguration
import org.apache.hadoop.hbase.client.ConnectionFactory
import org.apache.hadoop.hbase.client.Put
import org.apache.hadoop.hbase.util.Bytes

object FifaSpeedLayer {
  val mapper = new ObjectMapper()
  mapper.registerModule(DefaultScalaModule)

  //Check next 5 rows
  val hbaseConf: Configuration = HBaseConfiguration.create()
  hbaseConf.set("hbase.zookeeper.property.clientPort", "2181")
  hbaseConf.set("hbase.zookeeper.quorum", "localhost")
  val hbaseConnection = ConnectionFactory.createConnection(hbaseConf)
  val table = hbaseConnection.getTable(TableName.valueOf("newplayerdata"))
  
  def main(args: Array[String]) {
    if (args.length < 1) {
      System.err.println(s"""
        |Usage: FifaSpeedLayer <brokers>
        |  <brokers> is a list of one or more Kafka brokers
        | 
        """.stripMargin)
      System.exit(1)
    }

    val Array(brokers) = args

    // Create context with 2 second batch interval
    val sparkConf = new SparkConf().setAppName("FifaSpeedLayer")
    val ssc = new StreamingContext(sparkConf, Seconds(2))

    // Create direct kafka stream with brokers and topics
    val topicsSet = Set("fifa-player-updates")
    // Create direct kafka stream with brokers and topics
    val kafkaParams = Map[String, Object](
      "bootstrap.servers" -> brokers,
      "key.deserializer" -> classOf[StringDeserializer],
      "value.deserializer" -> classOf[StringDeserializer],
      "group.id" -> "use_a_separate_group_id_for_each_stream",
      "auto.offset.reset" -> "latest",
      "enable.auto.commit" -> (false: java.lang.Boolean)
    )
    val stream = KafkaUtils.createDirectStream[String, String](
      ssc, PreferConsistent,
      Subscribe[String, String](topicsSet, kafkaParams)
    )

    // Get the lines, split them into words, count the words and print
    val serializedRecords = stream.map(_.value);

    val newPlayers = serializedRecords.map(record => mapper.readValue(record, classOf[NewPlayerData]))

    newPlayers.foreachRDD { rdd =>
      rdd.foreach { player =>
        val uniqueIdentifier = s"${player.fifa_version}-${player.short_name}"

        // Create a new Put object with the unique identifier as the row key
        val put = new Put(Bytes.toBytes(uniqueIdentifier))

        // Add player attributes to the Put object
        put.addColumn(Bytes.toBytes("attributes"), Bytes.toBytes("overall"), Bytes.toBytes(player.overall))
        put.addColumn(Bytes.toBytes("attributes"), Bytes.toBytes("pace"), Bytes.toBytes(player.pace))
        put.addColumn(Bytes.toBytes("attributes"), Bytes.toBytes("shooting"), Bytes.toBytes(player.shooting))
        put.addColumn(Bytes.toBytes("attributes"), Bytes.toBytes("passing"), Bytes.toBytes(player.passing))
        put.addColumn(Bytes.toBytes("attributes"), Bytes.toBytes("dribbling"), Bytes.toBytes(player.dribbling))
        put.addColumn(Bytes.toBytes("attributes"), Bytes.toBytes("defending"), Bytes.toBytes(player.defending))
        put.addColumn(Bytes.toBytes("attributes"), Bytes.toBytes("physic"), Bytes.toBytes(player.physic))
        put.addColumn(Bytes.toBytes("attributes"), Bytes.toBytes("nationality_name"), Bytes.toBytes(player.nationality_name))
        put.addColumn(Bytes.toBytes("attributes"), Bytes.toBytes("preferred_foot"), Bytes.toBytes(player.preferred_foot))
        put.addColumn(Bytes.toBytes("attributes"), Bytes.toBytes("val_eur"), Bytes.toBytes(player.val_eur))

        // Perform the HBase insertion
        table.put(put)
      }
    }


    // Start the computation
    ssc.start()
    ssc.awaitTermination()
  }

}
