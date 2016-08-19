wget https://repo1.maven.org/maven2/org/apache/commons/commons-csv/1.4/commons-csv-1.4.jar -O commons-csv-1.4.jar   
wget https://repo1.maven.org/maven2/com/databricks/spark-csv_2.11/1.4.0/spark-csv_2.11-1.4.0.jar -O spark-csv_2.11-1.4.0.jar

#./bin/pyspark --jars "spark-csv_2.11-1.4.0.jar,commons-csv-1.4.jar"

export PACKAGES="com.databricks:spark-csv_2.11:1.4.0"
export PYSPARK_SUBMIT_ARGS="--packages ${PACKAGES} pyspark-shell"
sudo pip install -r requirements.txt
