import math
import json
from collections import Counter
from typing import Dict, Any

def extract_timeline_features(raw_emotions: Dict[str, list], total_duration: float = 5.0, slice_duration: float = 2.5) -> Dict[str, Any]:
    """
    Converts asynchronous raw emotion arrays into chronologically aligned time-slices.
    """
    num_slices = math.ceil(total_duration / slice_duration)
    
    # Initialize empty buckets for each time slice
    buckets = {i: {"facial": [], "speech": [], "body": []} for i in range(num_slices)}
    
    # 1. Distribute raw data into time buckets based on sample midpoints
    for modality, emotions in raw_emotions.items():
        if not emotions:
            continue
            
        sample_interval = total_duration / len(emotions)
        
        for i, emotion in enumerate(emotions):
            # Calculate the exact middle timestamp of this specific sample
            midpoint_time = (i + 0.5) * sample_interval
            
            # Determine which time slice this midpoint falls into
            slice_index = int(midpoint_time // slice_duration)
            slice_index = min(slice_index, num_slices - 1) # Prevent out-of-bounds
            
            buckets[slice_index][modality].append(emotion)

    # Helper function to get the most common emotion (handles ties like happiness/surprise)
    def get_majority(items):
        if not items:
            return "none"
        counts = Counter(items)
        most_common = counts.most_common(2)
        if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
            return f"{most_common[0][0]}/{most_common[1][0]}"
        return most_common[0][0]

    # Helper function to show emotion shifts (e.g., neutral -> happy)
    def get_shift(items):
        if not items:
            return "none"
        unique_ordered = []
        for item in items:
            if not unique_ordered or item != unique_ordered[-1]:
                unique_ordered.append(item)
        return " -> ".join(unique_ordered) if len(unique_ordered) > 1 else unique_ordered[0]

    # 2. Aggregate the buckets into the final structured schema
    timeline_slices = []
    for i in range(num_slices):
        start_time = i * slice_duration
        end_time = min((i + 1) * slice_duration, total_duration)
        
        slice_data = {
            "window": f"{start_time:.1f}s - {end_time:.1f}s",
            "facial_majority": get_majority(buckets[i]["facial"]),
            "speech_shift": get_shift(buckets[i]["speech"]),
            "body_state": get_majority(buckets[i]["body"])
        }
        timeline_slices.append(slice_data)

    return {
        "total_duration_seconds": total_duration,
        "timeline_slices": timeline_slices
    }

# --- Example Usage ---
if __name__ == "__main__":
    raw_data = {
      "speech": [
        "neutral",
        "happy",
        "fear",
        "neutral"
        ],
        "facial": [
            "neutral",
            "neutral",
            "neutral",
            "neutral",
            "happiness",
            "neutral",
            "neutral",
            "happiness",
            "surprise",
            "surprise",
            "neutral"
        ],
        "body": [
            "fear",
            "sadness"
        ]
    }

    processed_features = extract_timeline_features(raw_data, total_duration=5.0, slice_duration=2.5)
    
    print(json.dumps(processed_features, indent=2))