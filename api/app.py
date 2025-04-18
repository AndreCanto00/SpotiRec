from flask import Flask, request, jsonify
import redis
import json
import os
import random
import time
from prometheus_flask_exporter import PrometheusMetrics

app = Flask(__name__)
metrics = PrometheusMetrics(app)

# Define custom metrics
cache_hits = metrics.counter(
    'cache_hits', 'Number of cache hits'
)
cache_misses = metrics.counter(
    'cache_misses', 'Number of cache misses'
)
fallbacks_used = metrics.counter(
    'fallbacks_used', 'Number of fallback recommendations used'
)

# Connect to Redis
redis_host = os.environ.get('REDIS_HOST', 'localhost')
r = redis.Redis(host=redis_host, port=6379, db=0)

# Wait for Redis to be available
def wait_for_redis():
    max_retries = 30
    retry_interval = 1
    for _ in range(max_retries):
        try:
            r.ping()
            print("Successfully connected to Redis")
            return True
        except redis.exceptions.ConnectionError:
            print(f"Waiting for Redis... (retry in {retry_interval}s)")
            time.sleep(retry_interval)
    print("Failed to connect to Redis after multiple attempts")
    return False

# Default fallback tracks if Redis is not available
FALLBACK_TRACKS = [
    "Bohemian Rhapsody - Queen",
    "Hotel California - Eagles", 
    "Imagine - John Lennon",
    "Billie Jean - Michael Jackson",
    "Sweet Child O' Mine - Guns N' Roses"
]

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    redis_status = "healthy" if r.ping() else "unhealthy"
    return jsonify({
        "status": "healthy",
        "redis": redis_status
    })

@app.route('/', methods=['GET'])
def home():
    """Home endpoint"""
    return jsonify({
        "message": "Welcome to SpotiRec API!"
    })

@app.route('/listened', methods=['POST'])
def track_listened():
    """Record that a user listened to a track"""
    data = request.json
    if not data or 'user_id' not in data or 'track_id' not in data:
        return jsonify({"error": "Invalid request. Required fields: user_id, track_id"}), 400
    
    user_id = data['user_id']
    track_id = data['track_id']
    
    # Check if we have personalized recommendations in the cache
    user_preferences = r.get(f"prefs:{user_id}")
    
    if user_preferences:
        # Use existing preferences to generate a recommendation
        prefs = json.loads(user_preferences)
        # Simple logic: recommend something similar to what they've already liked
        predicted_next = prefs[0] if prefs else FALLBACK_TRACKS[0]
    else:
        # No preferences found, use a fallback
        fallback_data = r.get("fallback_tracks")
        if fallback_data:
            fallbacks = json.loads(fallback_data)
            predicted_next = random.choice(fallbacks)
        else:
            predicted_next = random.choice(FALLBACK_TRACKS)
    
    # Store the prediction in Redis
    r.set(f"rec:{user_id}", predicted_next)
    
    # Store listen history (limited to last 100 tracks)
    r.lpush(f"history:{user_id}", track_id)
    r.ltrim(f"history:{user_id}", 0, 99)
    
    return jsonify({
        "user_id": user_id,
        "listened": track_id,
        "predicted_next": predicted_next
    })

@app.route('/recommend/<user_id>', methods=['GET'])
def recommend(user_id):
    """Get recommendation for a user"""
    # Check if we have a cached recommendation
    cached_rec = r.get(f"rec:{user_id}")
    
    if cached_rec:
        # We have a cached recommendation
        cache_hits.inc()
        recommendation = cached_rec.decode('utf-8')
        cache_hit = True
        fallback_used = False
    else:
        # No cached recommendation
        cache_misses.inc()
        
        # Try to get fallback tracks from Redis
        fallback_data = r.get("fallback_tracks")
        if fallback_data:
            fallbacks = json.loads(fallback_data)
            recommendation = random.choice(fallbacks)
        else:
            recommendation = random.choice(FALLBACK_TRACKS)
        
        fallbacks_used.inc()
        cache_hit = False
        fallback_used = True
    
    return jsonify({
        "user_id": user_id,
        "recommendation": recommendation,
        "cache_hit": cache_hit,
        "fallback_used": fallback_used if not cache_hit else False
    })

@app.route('/history/<user_id>', methods=['GET'])
def get_history(user_id):
    """Get listening history for a user"""
    # Get listening history from Redis
    history = r.lrange(f"history:{user_id}", 0, -1)
    
    # Convert from bytes to strings
    history = [item.decode('utf-8') for item in history]
    
    return jsonify({
        "user_id": user_id,
        "history": history
    })

@app.route('/popular', methods=['GET'])
def get_popular():
    """Get popular tracks (fallbacks)"""
    fallback_data = r.get("fallback_tracks")
    
    if fallback_data:
        popular_tracks = json.loads(fallback_data)
    else:
        popular_tracks = FALLBACK_TRACKS
    
    return jsonify({
        "popular_tracks": popular_tracks[:10]  # Return top 10
    })

if __name__ == '__main__':
    # Wait for Redis before starting the app
    if wait_for_redis():
        app.run(host='127.0.0.1', port=5000)
    else:
        print("Exiting due to Redis connection failure")