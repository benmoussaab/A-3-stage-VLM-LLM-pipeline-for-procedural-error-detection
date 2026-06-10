import os
import time
import json
from google import genai
from google.genai import types

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# IMPORTANT: Provide your Gemini API key via environment variable.
API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_API_KEY_HERE")
MODEL_ID = "gemini-3-flash-preview"

# Input JSONL from Stage 2 (LLM Refiner)
INPUT_JSONL = "stage2_refined_actions.jsonl"
# Output JSONL for Stage 3
OUTPUT_JSONL = "stage3_error_detection_results.jsonl"

def load_refined_actions():
    video_items = []
    if not os.path.exists(INPUT_JSONL):
        print(f"Input file {INPUT_JSONL} not found.")
        return video_items
        
    with open(INPUT_JSONL, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            if not line.strip():
                continue
            data = json.loads(line)
            v_name = data.get('video_name')
            actions = data.get('cleaned_actions')
            if v_name and actions:
                video_items.append((v_name, actions))
    return video_items

# ------------------------------------------------------------------------------
# PERFECT TASK GRAPH
# ------------------------------------------------------------------------------
# NOTE: This graph is for a Quesadilla cooking task.
# Replace this with the canonical sequence of steps for your own task.
# ------------------------------------------------------------------------------
TASK_GRAPH = """PERFECT TASK GRAPH:
1. Place tortilla on cutting board
2. Use knife to scoop Nutella
3. Spread Nutella onto tortilla
4. Put banana slices on tortilla
5. Sprinkle cinnamon onto tortilla
6. Fold tortilla
7. Slice using knife
8. Place tortilla wedges on plate"""

# ------------------------------------------------------------------------------
# FEW-SHOT EXAMPLES (IN-CONTEXT LEARNING)
# ------------------------------------------------------------------------------
# NOTE: These examples perfectly align with the Quesadilla Task Graph.
# You MUST replace these examples with your own JSON actions -> JSON error 
# mappings for your specific task!
# ------------------------------------------------------------------------------
FEW_SHOT_EXAMPLES = """
### EXAMPLE 1:
[{"step": "Place tortilla on cutting board", "start": 2.5, "end": 11.2}, {"step": "Use knife to scoop Nutella", "start": 12.4, "end": 20.8}, {"step": "Spread Nutella onto tortilla", "start": 20.8, "end": 40.3}, {"step": "Sprinkle cinnamon onto tortilla", "start": 41.5, "end": 51.2}, {"step": "Put chocolate chips, cookies, and cereal on tortilla", "start": 52.4, "end": 63.8}, {"step": "Fold tortilla", "start": 65.1, "end": 72.4}, {"step": "Slice using knife", "start": 73.2, "end": 85.1}, {"step": "Place tortilla wedges on plate", "start": 86.4, "end": 91.2}]

OUTPUT:[
    {
      "step": "Place tortilla on cutting board",
      "error_type": "Normal",
      "start": 2.4,
      "end": 7.8
    },
    {
      "step": "Use knife to scoop Nutella",
      "error_type": "Normal",
      "start": 7.8,
      "end": 22.4
    },
    {
      "step": "Spread Nutella onto tortilla",
      "error_type": "Normal",
      "start": 22.4,
      "end": 35.9
    },
    {
      "step": "Sprinkle cinnamon onto tortilla",
      "error_type": "Normal",
      "start": 36.8,
      "end": 47.6
    },
    {
      "step": "Put banana slices on tortilla",
      "error_type": "Normal",
      "start": 48.1,
      "end": 63.8
    },
    {
      "step": "Fold tortilla",
      "error_type": "Normal",
      "start": 63.8,
      "end": 69.2
    },
    {
      "step": "Slice using knife",
      "error_type": "Normal",
      "start": 69.2,
      "end": 82.1
    },
    {
      "step": "Place tortilla wedges on plate",
      "error_type": "Normal",
      "start": 82.1,
      "end": 86.4
    }
  ]
### EXAMPLE 2:
INPUT: [{"step": "Place tortilla on cutting board", "start": 2.1, "end": 7.5}, {"step": "Use knife to scoop Nutella", "start": 8.4, "end": 16.2}, {"step": "Spread Nutella onto tortilla", "start": 16.2, "end": 31.5}, {"step": "Sprinkle cinnamon onto tortilla", "start": 32.7, "end": 41.2}, {"step": "Put banana slices on tortilla", "start": 42.1, "end": 52.3}, {"step": "Fold tortilla", "start": 53.4, "end": 59.1}, {"step": "Slice using knife", "start": 59.1, "end": 67.8}, {"step": "Place tortilla wedges on plate", "start": 68.2, "end": 71.1}]

OUTPUT: [
    {
      "step": "Place tortilla on cutting board",
      "error_type": "Normal",
      "start": 2.4,
      "end": 8.3
    },
    {
      "step": "Use knife to scoop Nutella",
      "error_type": "Normal",
      "start": 11.2,
      "end": 16.3
    },
    {
      "step": "Spread Nutella onto tortilla",
      "error_type": "Normal",
      "start": 16.3,
      "end": 29.2
    },
    {
      "step": "Sprinkle cinnamon onto tortilla",
      "error_type": "Normal",
      "start": 31.0,
      "end": 38.7
    },
    {
      "step": "Put banana slices on tortilla",
      "error_type": "Normal",
      "start": 39.3,
      "end": 46.8
    },
    {
      "step": "Fold tortilla",
      "error_type": "Normal",
      "start": 46.8,
      "end": 53.1
    },
    {
      "step": "Slice using knife",
      "error_type": "Normal",
      "start": 53.1,
      "end": 62.9
    },
    {
      "step": "Place tortilla wedges on plate",
      "error_type": "Normal",
      "start": 62.9,
      "end": 68.5
    }
  ]
### EXAMPLE 3:
INPUT:[{"step": "Place tortilla on cutting board", "start": 2.1, "end": 9.4}, {"step": "Use knife to scoop Nutella", "start": 11.5, "end": 20.8}, {"step": "Spread Nutella onto tortilla", "start": 20.8, "end": 58.2}, {"step": "Put banana slices on tortilla", "start": 62.4, "end": 74.1}, {"step": "Sprinkle coffee grounds onto tortilla", "start": 75.8, "end": 86.3}, {"step": "Fold tortilla", "start": 88.5, "end": 96.7}, {"step": "Slice using knife", "start": 98.2, "end": 112.5}, {"step": "Place tortilla wedges on plate", "start": 113.1, "end": 116.8}]

OUTPUT:[
    {
      "step": "Place tortilla on cutting board",
      "error_type": "Normal",
      "start": 2.3,
      "end": 11.8
    },
    {
      "step": "Use knife to scoop Nutella",
      "error_type": "Normal",
      "start": 13.6,
      "end": 44.2
    },
    {
      "step": "Spread Nutella onto tortilla",
      "error_type": "Normal",
      "start": 44.2,
      "end": 61.7
    },
    {
      "step": "Put banana slices on tortilla",
      "error_type": "Normal",
      "start": 61.7,
      "end": 76.2
    },
    {
      "step": "Sprinkle cinnamon onto tortilla",
      "error_type": "Normal",
      "start": 77.4,
      "end": 84.3
    },
    {
      "step": "Fold tortilla",
      "error_type": "Normal",
      "start": 85.0,
      "end": 91.4
    },
    {
      "step": "Slice using knife",
      "error_type": "Normal",
      "start": 91.4,
      "end": 109.0
    },
    {
      "step": "Place tortilla wedges on plate",
      "error_type": "Normal",
      "start": 109.0,
      "end": 113.1
    }
  ]
"""

# ==============================================================================
# 3. MISTAKE DETECTION LOOP
# ==============================================================================
def detect_errors_batched():
    video_items = load_refined_actions()
    if not video_items:
        return
        
    client = genai.Client(api_key=API_KEY)
    
    # --- SMART RESUME LOGIC ---
    processed_videos = set()
    if os.path.exists(OUTPUT_JSONL):
        with open(OUTPUT_JSONL, "r", encoding="utf-8") as file:
            for l in file:
                if l.strip():
                    try:
                        record = json.loads(l)
                        for key in record.keys():
                            # we expect {"video_name": ...} or the direct map from the prompt
                            if "video_name" in record:
                                processed_videos.add(record["video_name"])
                            else:
                                processed_videos.add(key)
                    except:
                        pass
    print(f"Found {len(processed_videos)} videos already audited. Resuming...")

    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_level="HIGH"),
        temperature=1.0,
        system_instruction=f"""You are a high-precision Cooking Procedure Auditor analyzing egocentric cooking videos.

You will evaluate the video against this {TASK_GRAPH}

STRICT AUDIT RULES:
1. NORMAL STEPS: If a step perfectly matches the task graph without deviation, label its "error_type" as "Normal".
2. ERRORS (Modification, Addition, Slip, Correction): If the person attempts a step but does it wrong , or does something extra, describe what they ACTUALLY did in the "step" field and label its "error_type" as "Error".
3. OMISSIONS: If a canonical step from the task graph is completely skipped and never attempted, you must output a segment for it using the EXACT canonical step name, label its "error_type" as "Omission", and set the start and end times to the exact same second (e.g., the time when the step should have happened). 

OUTPUT FORMAT:
You will receive up to 3 videos. For each video, return a JSON object containing an array of segments in chronological order. Each segment must have:
- "step": (string) The canonical step name (if Normal/Omission) or a description of the error (if Error).
- "error_type": (string) Strictly one of: ["Normal", "Error", "Omission"].
- "start": (float) Start time in seconds.
- "end": (float) End time in seconds.

Ensure that every single step from the Perfect Task Graph is accounted for, either as Normal, Error, or Omission!"""
    )

    for i in range(0, len(video_items), 3):
        batch = video_items[i : i + 3]
        
        # Check if all videos in this batch are already processed
        if all(name in processed_videos for name, _ in batch):
            print(f"  -> Batch {i//3 + 1} already audited, skipping.")
            continue
            
        name_map = {f"Video_{idx+1}": name for idx, (name, _) in enumerate(batch)}
        batch_input = "\n".join([f"Video_{idx+1}: {json.dumps(actions)}" for idx, (_, actions) in enumerate(batch)])

        print(f"Auditing Batch {i//3 + 1} ({len(batch)} videos)...")
        prompt = f"{FEW_SHOT_EXAMPLES}\n\nAudit these videos:\n{batch_input}\nReturn JSON with keys 'Video_1', 'Video_2', 'Video_3' (if applicable)."

        try:
            response = client.models.generate_content(model=MODEL_ID, contents=prompt, config=config)
            raw = response.text.strip().replace("```json", "").replace("```", "").strip()
            res = json.loads(raw)
            
            # Map "Video_1" back to its actual filename and save individually
            for k, actions in res.items():
                if k in name_map:
                    actual_name = name_map[k]
                    final_record = {"video_name": actual_name, "audited_actions": actions}
                    with open(OUTPUT_JSONL, "a", encoding="utf-8") as f:
                        f.write(json.dumps(final_record) + "\n")
                        
            print(f"✅ Saved batch {i//3 + 1}")
            time.sleep(2)
        except Exception as e:
            print(f"❌ Error on batch {i//3 + 1}: {e}")
            continue

    print(f"\nAll done! Output saved to {OUTPUT_JSONL}")

if __name__ == "__main__":
    detect_errors_batched()
