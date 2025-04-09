import json
import math
import urllib3
from datetime import datetime, timedelta

# Initialize HTTP client
http = urllib3.PoolManager(timeout=urllib3.Timeout(connect=2.0, read=2.0))

def fetch_player_stats(player_id):
    """Fetch player stats from API"""
    try:
        response = http.request(
            'GET',
            f"http://localhost:8081/api/player-stats/by-player-id/{player_id}"
        )
        if response.status == 200:
            stats = json.loads(response.data.decode('utf-8'))
            return {
                'runs': int(stats['runs']),
                'balls': int(stats['balls']),
                'fours': int(stats['fours']),
                'sixes': int(stats['sixes']),
                'average': float(stats['average']),
                'strikeRate': float(stats['strikeRate']),
                'matches': int(stats['matches'])
            }
        return None
    except Exception as e:
        print(f"Error fetching stats: {str(e)}")
        return None

def calculate_cosine_similarity(prediction, actual):
    """Calculate similarity between prediction and actual stats"""
    if not actual:
        return 0.7  # Default if no stats available
    
    all_keys = set(prediction.keys()).union(set(actual.keys()))
    dot_product = sum(prediction.get(key, 0) * actual.get(key, 0) for key in all_keys)
    magnitude_pred = math.sqrt(sum(prediction.get(key, 0) ** 2 for key in all_keys))
    magnitude_actual = math.sqrt(sum(actual.get(key, 0) ** 2 for key in all_keys))
    
    epsilon = 1e-9
    raw_similarity = dot_product / (magnitude_pred * magnitude_actual + epsilon)
    return 0.5 + (raw_similarity * 0.5)  # Scale to 0.5-1.0 range

def calculate_multiplier(prediction, player_stats):
    """Calculate payout multiplier based on prediction quality"""
    if not player_stats:
        return 1.5  # Default multiplier if no stats
    
    # Normalize stats to per-match basis
    matches = max(1, player_stats['matches'])
    actual_stats = {
        'runs': player_stats['runs'] / matches,
        'balls': player_stats['balls'] / matches,
        'fours': player_stats['fours'] / matches,
        'sixes': player_stats['sixes'] / matches
    }
    
    # Calculate prediction quality
    similarity = calculate_cosine_similarity(prediction, actual_stats)
    
    # Calculate player consistency (0.5-1.0 range)
    consistency = min(1.0, max(0.5, 
        (player_stats['average'] / 50 + player_stats['strikeRate'] / 150) / 2
    ))
    
    # Final multiplier calculation (1.0-3.0 range)
    base_multiplier = 1.0 + (similarity * consistency)
    return min(3.0, max(1.0, base_multiplier))

def lambda_handler(event, context):
    """Handle Lambda invocation"""
    try:
        # Parse input
        if 'body' in event:
            body = json.loads(event['body'])
        else:
            body = event
        
        # Validate input
        required_fields = ['player_id', 'bet_amount', 'prediction']
        missing_fields = [field for field in required_fields if field not in body]
        
        if missing_fields:
            return {
                'statusCode': 400,
                'body': json.dumps({
                    'error': 'Missing required fields',
                    'missing_fields': missing_fields
                })
            }
        
        # Fetch stats and calculate
        player_stats = fetch_player_stats(body['player_id'])
        multiplier = calculate_multiplier(body['prediction'], player_stats)
        estimated_return = round(body['bet_amount'] * multiplier, 2)
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'player_id': body['player_id'],
                'bet_amount': body['bet_amount'],
                'estimated_return': estimated_return,
                'multiplier': round(multiplier, 2),
                'stats_used': {
                    'runs_per_match': player_stats['runs']/player_stats['matches'] if player_stats else None,
                    'balls_per_match': player_stats['balls']/player_stats['matches'] if player_stats else None,
                    'average': player_stats['average'] if player_stats else None,
                    'strikeRate': player_stats['strikeRate'] if player_stats else None
                }
            })
        }
        
    except json.JSONDecodeError:
        return {
            'statusCode': 400,
            'body': json.dumps({'error': 'Invalid JSON format'})
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
