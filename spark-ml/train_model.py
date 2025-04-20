"""
Spotify Recommendation System - Model Training Script
This script trains an ALS recommendation model using Spark and stores the results in Redis.
"""
import os
import time
import json
import pandas as pd
import numpy as np
import redis
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.sql.functions import col, explode
from pyspark.sql.types import StructType, StructField, StringType, FloatType, IntegerType
from pyspark.ml.feature import StringIndexer


def main():
    print("Starting Spotify recommendation model training...")

    # Update Spark configuration to use Arrow
    spark = SparkSession.builder \
        .appName("SpotifyRecommendationTraining") \
        .master("local[*]") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .config("spark.sql.execution.arrow.pyspark.fallback.enabled", "true") \
        .getOrCreate()

    redis_host = os.environ.get('REDIS_HOST', 'localhost')
    r = wait_for_redis(redis_host)
    if not r:
        print("Exiting due to Redis connection failure")
        return

    try:
        tracks_df = pd.read_csv("/workspaces/SpotiRec/data/data.csv")
        tracks_df = tracks_df.rename(columns={'id': 'track_id'})
        print(f"Tracks dataset loaded with shape: {tracks_df.shape}")

        all_tracks = []
        for _, row in tracks_df.iterrows():
            track_info = {
                'id': row['track_id'],
                'name': row['name'],
                'artist': row['artists'],
                'album': row.get('album', ''),
                'genre': row.get('genre', '')
            }
            all_tracks.append(track_info)

        r.set("all_tracks", json.dumps(all_tracks))
        print(f"Stored {len(all_tracks)} tracks in Redis")

    except Exception as e:
        print(f"Error loading tracks dataset: {e}")
        return

    interactions = generate_synthetic_interactions(tracks_df)

    schema = StructType([
        StructField("user_id", StringType(), True),
        StructField("track_id", StringType(), True),
        StructField("rating", FloatType(), True)
    ])

    # Convert to explicit Python types before creating the Spark DataFrame
    interactions_py = [(str(row.user_id), str(row.track_id), float(row.rating)) 
                       for _, row in interactions.iterrows()]
    
    spark_interactions = spark.createDataFrame(interactions_py, schema=schema)

    # Create string indexers
    user_indexer = StringIndexer(inputCol="user_id", outputCol="user_index")
    track_indexer = StringIndexer(inputCol="track_id", outputCol="track_index")

    # Apply the indexers and keep the original columns for mapping back
    spark_interactions = user_indexer.fit(spark_interactions).transform(spark_interactions)
    spark_interactions = track_indexer.fit(spark_interactions).transform(spark_interactions)

    # Ensure the output columns are properly cast to the right types
    spark_interactions = spark_interactions.withColumn("user_index", col("user_index").cast(IntegerType()))
    spark_interactions = spark_interactions.withColumn("track_index", col("track_index").cast(IntegerType()))
    spark_interactions = spark_interactions.withColumn("rating", col("rating").cast(FloatType()))

    (training, test) = spark_interactions.randomSplit([0.8, 0.2])

    print("Training ALS model...")
    als = ALS(
        userCol="user_index",
        itemCol="track_index",
        ratingCol="rating",
        coldStartStrategy="drop",
        nonnegative=True,
        rank=10,
        maxIter=5,
        regParam=0.1
    )

    model = als.fit(training)

    print("Generating recommendations...")
    user_recs = model.recommendForAllUsers(10)

    # Use user_index instead of user_id which doesn't exist in the result dataframe
    flat_recs = user_recs.select(
        col("user_index"),  # Changed from user_id to user_index
        explode(col("recommendations")).alias("recommendation")
    ).select(
        col("user_index"),  # Changed from user_id to user_index
        col("recommendation.track_index").alias("track_index"),  # Changed from track_id to track_index
        col("recommendation.rating").alias("predicted_rating")
    )

    # Now you need to map the indices back to the original IDs
    # First get the mapping from indices to original IDs
    user_index_to_id = spark_interactions.select("user_id", "user_index").distinct()
    track_index_to_id = spark_interactions.select("track_id", "track_index").distinct()

    # Join with the mapping tables to get the original IDs
    user_recs_with_ids = flat_recs.join(
        user_index_to_id, 
        on="user_index", 
        how="left"
    ).join(
        track_index_to_id, 
        on="track_index", 
        how="left"
    )

    # Convert to pandas DataFrame for Redis storage
    user_recs_pd = user_recs_with_ids.select(
        "user_id", "track_id", "predicted_rating"
    ).toPandas()

    print("Storing recommendations in Redis...")
    for user_id, group in user_recs_pd.groupby('user_id'):
        top_tracks = group.sort_values('predicted_rating', ascending=False)['track_id'].tolist()
        r.set(f"prefs:{user_id}", json.dumps(top_tracks))
        if top_tracks:
            r.set(f"rec:{user_id}", top_tracks[0])

    print("Generating fallback recommendations...")
    popular_tracks = tracks_df.sort_values('popularity', ascending=False)['track_id'].tolist()
    r.set("fallback_tracks", json.dumps(popular_tracks))

    print("Model training and recommendation storage complete!")

    while True:
        print("Model training job active. Press Ctrl+C to terminate.")
        time.sleep(300)
        print("Refreshing recommendations...")

def wait_for_redis(host, max_retries=30, retry_interval=1):
    print(f"Attempting to connect to Redis at {host}...")
    for i in range(max_retries):
        try:
            r = redis.Redis(host=host, port=6379, db=0)
            r.ping()
            print(f"Successfully connected to Redis at {host}")
            return r
        except redis.exceptions.ConnectionError:
            print(f"Waiting for Redis... (attempt {i+1}/{max_retries})")
            time.sleep(retry_interval)

    print(f"Failed to connect to Redis at {host} after {max_retries} attempts")
    return None

def generate_synthetic_interactions(tracks_df):
    print("Generating synthetic user-track interactions...")

    num_users = 100
    min_interactions = 5
    max_interactions = 15
    interactions = []

    track_ids = tracks_df['track_id'].tolist()

    for user_id in range(1, num_users + 1):
        num_interactions = np.random.randint(min_interactions, max_interactions + 1)
        sample_size = min(num_interactions, len(track_ids))
        user_tracks = np.random.choice(track_ids, size=sample_size, replace=False)

        ratings = []
        for track in user_tracks:
            track_popularity = tracks_df[tracks_df['track_id'] == track]['popularity'].values[0]
            rating_bias = min(track_popularity / 20, 3)
            base_rating = np.random.random() * 2
            rating = min(base_rating + rating_bias, 5)
            ratings.append(float(rating))

        for i, track_id in enumerate(user_tracks):
            # Store as Python native types, not numpy types
            interactions.append((f"U{user_id}", str(track_id), float(ratings[i])))

    return pd.DataFrame(interactions, columns=['user_id', 'track_id', 'rating'])

if __name__ == "__main__":
    main()
