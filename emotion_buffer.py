# emotion_buffer.py

import json
from datetime import datetime, timedelta
from collections import Counter
import os

# clear history
cleared_once = False

def save_emotion_to_json(emotion_label, file_path="emotion_log.json", max_entries=15000):   # 25 frame/s ->10min: 15000
    global cleared_once

    # clear history when start
    if not cleared_once and os.path.exists(file_path):
        with open(file_path, "w") as f:
            json.dump({"emotions": []}, f, indent=2)
        cleared_once = True

    # record new emotion
    entry = {
        "emotion": emotion_label,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {"emotions": []}
    else:
        data = {"emotions": []}

    data["emotions"].append(entry)

    if len(data["emotions"]) > max_entries:
        data["emotions"] = data["emotions"][-max_entries:]

    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)


def load_latest_emotion(file_path="emotion_log.json", lookback_minutes=10, window_seconds=10):
    try:
        with open(file_path, "r") as f:
            data = json.load(f)

        now = datetime.now()
        time_threshold = now - timedelta(minutes=lookback_minutes)

        # only load the latest 10 minutes emotion
        emotion_entries = [
            {
                "emotion": e["emotion"],
                "time": datetime.strptime(e["time"], "%Y-%m-%d %H:%M:%S")
            }
            for e in data.get("emotions", [])
            if datetime.strptime(e["time"], "%Y-%m-%d %H:%M:%S") >= time_threshold
        ]

        if not emotion_entries:
            return ["neutral"]

        result_sequence = []
        buffer = []
        start_time = emotion_entries[0]["time"]

        for entry in emotion_entries:
            if (entry["time"] - start_time).total_seconds() <= window_seconds:
                buffer.append(entry)
            else:
                # Process the current window
                non_neutral = [e["emotion"] for e in buffer if e["emotion"].lower() != "neutral"]
                if non_neutral:
                    top = Counter(non_neutral).most_common(1)[0][0]
                    if not result_sequence or top != result_sequence[-1]:
                        result_sequence.append(top)
                buffer = [entry]
                start_time = entry["time"]

        # Final window
        if buffer:
            non_neutral = [e["emotion"] for e in buffer if e["emotion"].lower() != "neutral"]
            if non_neutral:
                top = Counter(non_neutral).most_common(1)[0][0]
                if not result_sequence or top != result_sequence[-1]:
                    result_sequence.append(top)

        return result_sequence if result_sequence else ["neutral"]

    except Exception as e:
        print(f"Error loading emotion log: {e}")
        return ["neutral"]
