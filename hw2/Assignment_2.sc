/* --------------------- #1 --------------------- */
val block_1 = spark.read.csv("block_1.csv")
val dataRDD = block_1.rdd

/* --------------------- #2 --------------------- */
dataRDD.first
val header = dataRDD.first()
val noheader = dataRDD.filter(x => x != header)
noheader.first

/* ------------------- #3 & 4 ------------------- */
def toDouble(s: String) = {
    if ("?".equals(s)) Double.NaN else s.toDouble
    }

def parse(line: String) = {
    val pieces = line.split(',')  
    val id1 = pieces(0).toInt
    val id2 = pieces(1).toInt
    val scores = pieces.slice(2, 11).map(toDouble)  
    val matched = pieces(11).toBoolean
    (id1, id2, scores, matched)
    }

val parsed = noheader.map(line => parse(line.mkString(",")))

/* --------------------- #5 --------------------- */
import java.lang.Double.isNaN
val grouped = parsed.groupBy(md => md._4)
val scores = parsed.map(md => md._3(0)).filter(!isNaN(_)).stats()

/* --------------------- #6 --------------------- */