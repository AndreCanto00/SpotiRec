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

def main():
    print("Starting Spotify recommendation model training...")
    
    # Initialize Spark session
    spark = SparkSession.builder \
        .appName("SpotifyRecommendationTraining") \
        .master("local[*]") \
        .getOrCreate()
    
    # Connect to Redis
    redis_host = os.environ.get('REDIS_HOST', 'localhost')
    
    # Wait for Redis to be available
    r = wait_for_redis(redis_host)
    if not r:
        print("Exiting due to Redis connection failure")
        return
    
    # Load the synthetic data
    try:
        tracks_df = pd.read_csv("/workspaces/SpotiRec/data/data.csv")
        print(f"Tracks dataset loaded with shape: {tracks_df.shape}")
        
        # Store all tracks in Redis for reference
        all_tracks = []
        for _, row in tracks_df.iterrows():
            track_info = {
                'id': row['id'],
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
    
    # Generate synthetic user-track interactions
    interactions = generate_synthetic_interactions(tracks_df)
    
    # Convert to Spark DataFrame
    schema = StructType([
        StructField("user_id", StringType(), True),
        StructField("track_id", StringType(), True),
        StructField("rating", FloatType(), True)
    ])
    
    spark_interactions = spark.createDataFrame(interactions, schema=schema)
    
    # Split the data
    (training, test) = spark_interactions.randomSplit([0.8, 0.2])
    
    # Build and train the ALS model
    print("Training ALS model...")
    als = ALS(
        userCol="user_id",
        itemCol="track_id",
        ratingCol="rating",
        coldStartStrategy="drop",
        nonnegative=True,
        rank=10,
        maxIter=5,
        regParam=0.1
    )
    
    model = als.fit(training)
    
    # Generate recommendations for all users
    print("Generating recommendations...")
    user_recs = model.recommendForAllUsers(10)
    
    # Flatten recommendations
    flat_recs = user_recs.select(
        col("user_id"),
        explode(col("recommendations")).alias("recommendation")
    ).select(
        col("user_id"),
        col("recommendation.track_id").alias("track_id"),
        col("recommendation.rating").alias("predicted_rating")
    )
    
    # Collect recommendations and store in Redis
    user_recs_pd = flat_recs.toPandas()
    
    print("Storing recommendations in Redis...")
    # Group by user_id
    for user_id, group in user_recs_pd.groupby('user_id'):
        # Get top recommendations for this user
        top_tracks = group.sort_values('predicted_rating', ascending=False)['track_id'].tolist()
        
        # Store in Redis
        r.set(f"prefs:{user_id}", json.dumps(top_tracks))
        
        if top_tracks:
            r.set(f"rec:{user_id}", top_tracks[0])
    
    # Generate fallback recommendations (most popular tracks)
    print("Generating fallback recommendations...")
    popular_tracks = tracks_df.sort_values('popularity', ascending=False)['track_id'].tolist()
    r.set("fallback_tracks", json.dumps(popular_tracks))
    
    print("Model training and recommendation storage complete!")
    
    # Keep the script running to simulate a long-running service
    # In a real system, this would periodically retrain the model
    while True:
        print("Model training job active. Press Ctrl+C to terminate.")
        time.sleep(300)  # Sleep for 5 minutes
        print("Refreshing recommendations...")
        # In a real system, we would check for new data and retrain if needed

def wait_for_redis(host, max_retries=30, retry_interval=1):
    """Wait for Redis to be available"""
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
    """Generate synthetic user-track interactions for demonstration"""
    print("Generating synthetic user-track interactions...")
    
    # Create 100 users
    num_users = 100
    # Each user interacts with 5-15 tracks
    min_interactions = 5
    max_interactions = 15
    
    interactions = []
    
    # Get track IDs
    track_ids = tracks_df['track_id'].tolist()
    
    # Generate interactions
    for user_id in range(1, num_users + 1):
        num_interactions = np.random.randint(min_interactions, max_interactions + 1)
        
        # Ensure we don't try to sample more than available
        sample_size = min(num_interactions, len(track_ids))
        user_tracks = np.random.choice(track_ids, size=sample_size, replace=False)
        
        # Generate ratings (1-5) with bias towards higher ratings for popular tracks
        ratings = []
        for track in user_tracks:
            track_popularity = tracks_df[tracks_df['track_id'] == track]['popularity'].values[0]
            # Higher popularity increases chance of higher rating
            rating_bias = min(track_popularity / 20, 3)  # Maps 0-100 popularity to 0-3 bias
            base_rating = np.random.random() * 2  # Random value between 0-2
            rating = min(base_rating + rating_bias, 5)  # Combine for final rating, cap at 5
            ratings.append(float(rating))
        
        for i, track_id in enumerate(user_tracks):
            interactions.append((f"U{user_id}", track_id, ratings[i]))
    
    return pd.DataFrame(interactions, columns=['user_id', 'track_id', 'rating'])

if __name__ == "__main__":
    main()