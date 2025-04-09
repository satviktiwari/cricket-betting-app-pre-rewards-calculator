import json
import math

def calculate_euclidean_distance(prediction, actual):
    """Calculate Euclidean distance."""
    squared_diff = sum((prediction[key] - actual.get(key, 0)) ** 2 for key in prediction)
    return math.sqrt(squared_diff)

def calculate_manhattan_distance(prediction, actual):
    """Calculate Manhattan distance."""
    return sum(abs(prediction[key] - actual.get(key, 0)) for key in prediction)

def calculate_minkowski_distance(prediction, actual, p):
    """Calculate Minkowski distance."""
    return sum(abs(prediction[key] - actual.get(key, 0)) ** p for key in prediction) ** (1 / p)

def calculate_cosine_similarity(prediction, actual):
    """Calculate Cosine Similarity."""
    if prediction == actual:
        return 1.0
    
    # Get all unique keys from both prediction and actual
    all_keys = set(prediction.keys()).union(set(actual.keys()))
    
    dot_product = sum(prediction.get(key, 0) * actual.get(key, 0) for key in all_keys)
    magnitude_pred = math.sqrt(sum(prediction.get(key, 0) ** 2 for key in all_keys))
    magnitude_actual = math.sqrt(sum(actual.get(key, 0) ** 2 for key in all_keys))
    
    epsilon = 1e-9
    return dot_product / (magnitude_pred * magnitude_actual + epsilon)

def calculate_combined_score(prediction, actual, weights, p=3):
    """Calculate a combined score based on weighted distances and cosine similarity."""
    euclidean = calculate_euclidean_distance(prediction, actual)
    manhattan = calculate_manhattan_distance(prediction, actual)
    minkowski = calculate_minkowski_distance(prediction, actual, p)
    cosine = calculate_cosine_similarity(prediction, actual)
    
    cosine_distance = 1 - cosine
    
    combined_score = (
        euclidean * weights["Euclidean"] +
        manhattan * weights["Manhattan"] +
        minkowski * weights["Minkowski"] +
        cosine_distance * weights["Cosine"]
    )
    
    combined_score = 1 / (1 + combined_score)
    
    return combined_score

def calculate_rewards(predictions, actual, investments, weights, p=3):
    """Calculate rewards for each user based on their combined score and investment."""
    user_scores = {}
    
    for user, user_prediction in predictions.items():
        total_score = 0
        count = 0
        
        # For each player the user predicted
        for player, prediction in user_prediction.items():
            if player in actual:  # Only consider players that actually played
                score = calculate_combined_score(prediction, actual[player], weights, p)
                total_score += score
                count += 1
        
        # Average score across all predicted players
        user_scores[user] = total_score / count if count > 0 else 0
    
    total_budget = sum(investments.values())
    
    weighted_scores = {
        user: user_scores[user] * investments.get(user, 0)
        for user in predictions
    }
    
    total_weighted_score = sum(weighted_scores.values())
    
    if total_weighted_score == 0:
        # Handle edge case where all scores are 0
        reward_weights = {user: 1/len(predictions) for user in predictions}
    else:
        reward_weights = {
            user: weighted_scores[user] / total_weighted_score
            for user in predictions
        }
    
    rewards = {
        user: reward_weights[user] * total_budget
        for user in predictions
    }
    
    return rewards

def lambda_handler(event, context):
    """AWS Lambda handler function for reward calculation."""
    try:
        print("Incoming Event:", json.dumps(event))
        
        if "body" in event:
            body = json.loads(event["body"])
        else:
            body = event
        
        predictions = body["predictions"]
        actual = body["actual"]
        investments = body["investments"]
        weights = body.get("weights", {"Euclidean": 0.4, "Manhattan": 0.3, "Minkowski": 0.2, "Cosine": 0.1})
        p = body.get("p", 3)
        
        rewards = calculate_rewards(predictions, actual, investments, weights, p)
        
        return {
            "statusCode": 200,
            "body": json.dumps({
                "rewards": rewards,
                "message": "Rewards calculated successfully"
            })
        }
    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            "statusCode": 500,
            "body": json.dumps({
                "error": str(e),
                "message": "Error calculating rewards"
            })
        }
