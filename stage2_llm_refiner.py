import os
import time
import json
import re
from google import genai
from google.genai import types

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# IMPORTANT: Provide your Gemini API key via environment variable.
API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_API_KEY_HERE")
MODEL_ID = "gemini-2.5-flash"

# Input JSON from Stage 1 (VLM Describer)
INPUT_JSON = "stage1_vlm_descriptions.json"
# Output JSONL for Stage 2
OUTPUT_JSONL = "stage2_refined_actions.jsonl"

def load_transcripts():
    try:
        with open(INPUT_JSON, 'r', encoding='utf-8') as f:
            video_data = json.load(f)
        
        # Convert dictionary to a list of tuples (video_name, segments_list)
        return list(video_data.items())
    except Exception as e:
        print(f"Error loading data from {INPUT_JSON}: {e}")
        return []

# ------------------------------------------------------------------------------
# IN-CONTEXT EXAMPLES (FEW-SHOT PROMPT)
# ------------------------------------------------------------------------------
# NOTE: These examples are heavily tailored for a "Quesadilla" cooking task.
# If you are adapting this for a different task, you MUST replace these examples
# with your own noisy VLM transcript -> clean JSON ground-truth pairings!
# ------------------------------------------------------------------------------
EXAMPLES_TEXT = """I am going to provide you with examples of how to clean my noisy data. Study these patterns carefully before I give you the real data to process.

### EXAMPLE 1
TRANSCRIPT:
"[0.0s - 10.0s] The person is taking out a tortilla from a bag and placing it on a red cutting board.
[5.0s - 15.0s] A person is placing a tortilla on a red cutting board, then opening a jar of Nutella and using a knife to spread it on the tortilla.
[10.0s - 20.0s] The person is opening a jar, taking out a spoon, and spreading a chocolate spread onto a tortilla on a red cutting board.
[15.0s - 25.0s] The person is spreading Nutella onto a tortilla using a knife.
[20.0s - 30.0s] A person is spreading chocolate spread onto a tortilla using a knife on a red cutting board.
[25.0s - 35.0s] The person is spreading a brown spread onto a tortilla using a knife, then adding banana slices on top.
[30.0s - 40.0s] The person is using their hands to place banana slices on a tortilla spread with Nutella on a red cutting board.
[35.0s - 45.0s] The person is placing banana slices on a tortilla spread with Nutella, then sprinkling cocoa powder on top.
[40.0s - 50.0s] The person is using a pepper mill to sprinkle ground pepper onto a tortilla that already has chocolate spread and banana slices on it, then folds the tortilla in half.
[45.0s - 55.0s] The person is using their hands to fold a tortilla into a taco shape, then using a knife to cut it in half.
[50.0s - 60.0s] The person is pressing a tortilla on a red cutting board and then using a knife to cut it into two pieces.
[55.0s - 65.0s] The person is using a knife to cut a sandwich in half on a red cutting board.
[60.0s - 66.5s] The person is using a red cutting board to cut a sandwich into two halves, then placing the halves on a white plate.
[65.0s - 66.5s] The person is using a whiteboard marker to write on a whiteboard."
REFINED ACTIONS JSON:
[{"step": "Place tortilla on cutting board", "start": 2.4, "end": 7.1}, {"step": "Use knife to scoop Nutella", "start": 11.8, "end": 17.1}, {"step": "Spread Nutella onto tortilla", "start": 17.1, "end": 29.9}, {"step": "Put banana slices on tortilla", "start": 31.4, "end": 39.0}, {"step": "Sprinkle cinnamon onto tortilla", "start": 39.8, "end": 46.6}, {"step": "Fold tortilla", "start": 46.6, "end": 53.2}, {"step": "Slice using knife", "start": 53.2, "end": 60.1}, {"step": "Place tortilla wedges on plate", "start": 60.1, "end": 63.4}]

### EXAMPLE 2
TRANSCRIPT:
"[0.0s - 10.0s] The person is taking out a tortilla from a bag and placing it on a red cutting board.
[5.0s - 15.0s] The person is placing a tortilla on a red cutting board.
[10.0s - 20.0s] The person is using a knife to cut a tortilla on a red cutting board.
[15.0s - 25.0s] The person is using a knife to open a jar of Nutella and spread it onto a tortilla on a red cutting board.
[20.0s - 30.0s] The person is using a knife to spread chocolate hazelnut spread onto a tortilla on a red cutting board.
[25.0s - 35.0s] The person is spreading a dark-colored spread onto a tortilla using a knife.
[30.0s - 40.0s] The person is spreading chocolate hazelnut spread onto a tortilla using a knife.
[35.0s - 45.0s] A person is spreading chocolate hazelnut spread onto a tortilla using a knife.
[40.0s - 50.0s] A person is spreading chocolate spread onto a tortilla using a knife.
[45.0s - 55.0s] A person is spreading chocolate spread onto a tortilla using a knife.
[50.0s - 60.0s] A person is spreading chocolate spread onto a tortilla using a knife.
[55.0s - 65.0s] The person is spreading a chocolate spread onto a tortilla using a knife.
[60.0s - 70.0s] The person is spreading chocolate hazelnut spread onto a tortilla using a knife.
[65.0s - 75.0s] The person is spreading chocolate hazelnut spread on a tortilla using a knife and then adding chopped hazelnuts on top.
[70.0s - 80.0s] The person is using a spoon to add chopped hazelnuts onto a tortilla spread with Nutella.
[75.0s - 85.0s] The person is pouring honey from a jar onto the tortilla using a spoon.
[80.0s - 90.0s] The person is using a pepper mill to sprinkle ground pepper onto the tortilla, which already has chocolate spread and sliced bananas on it.
[85.0s - 95.0s] The person is using a chocolate spreader to apply chocolate spread onto a tortilla.
[90.0s - 100.0s] The person is sprinkling a dark brown powder onto a tortilla, then folding it into a half-moon shape.
[95.0s - 105.0s] The person is using their hands to fold a tortilla into a half-moon shape on a red cutting board, with a knife and a jar of chocolate spread nearby.
[100.0s - 110.0s] The person is using their hands to press down on a tortilla on a red cutting board, then picks up a knife to cut the tortilla.
[105.0s - 115.0s] A person is using a knife to cut a tortilla on a red cutting board.
[110.0s - 120.0s] A person is using a knife to cut a tortilla into two pieces on a red cutting board.
[115.0s - 125.0s] The person is using a knife to cut a tortilla into two pieces on a red cutting board.
[120.0s - 130.0s] A person is cutting a tortilla into two pieces using a knife on a red cutting board.
[125.0s - 135.0s] The person is using their hands to fold a tortilla into a triangle, then placing it on a plate.
[130.0s - 136.1s] The person is using a knife to cut a quesadilla on a red cutting board.
[135.0s - 136.1s] No action."
REFINED ACTIONS JSON:
[{"step": "Place tortilla on cutting board", "start": 2.0, "end": 11.3}, {"step": "Use knife to scoop Nutella", "start": 17.4, "end": 37.6}, {"step": "Spread Nutella onto tortilla", "start": 37.6, "end": 63.4}, {"step": "Put banana slices on tortilla", "start": 67.6, "end": 74.5}, {"step": "Sprinkle cinnamon onto tortilla", "start": 80.1, "end": 92.6}, {"step": "Fold tortilla", "start": 95.6, "end": 105.1}, {"step": "Slice using knife", "start": 106.7, "end": 120.5}, {"step": "Place tortilla wedges on plate", "start": 121.4, "end": 129.6}]

### EXAMPLE 3
TRANSCRIPT:
"[0.0s - 10.0s] The person is taking out a tortilla from a package and placing it on a red cutting board.
[5.0s - 15.0s] The person is placing a tortilla on a red cutting board and then opening a jar of Nutella.
[10.0s - 20.0s] The person is using a knife to spread Nutella onto a tortilla on a red cutting board.
[15.0s - 25.0s] The person is spreading Nutella onto a tortilla using a knife.
[20.0s - 30.0s] The person is spreading Nutella on a tortilla using a knife.
[25.0s - 35.0s] The person is spreading Nutella on a tortilla using a knife, then sprinkling chopped bananas and sprinkles on top.
[30.0s - 40.0s] The person is sprinkling a spice onto a tortilla using a spice shaker.
[35.0s - 45.0s] The person is spreading Nutella on a tortilla using a knife.
[40.0s - 50.0s] The person is spreading sliced bananas onto a tortilla that has chocolate spread on it, using their hands.
[45.0s - 55.0s] The person is spreading a brown spread on a tortilla using a knife, then folding the tortilla in half.
[50.0s - 60.0s] The person is folding a tortilla in half using their hands.
[55.0s - 65.0s] The person is using a knife to cut a tortilla into two pieces on a red cutting board.
[60.0s - 70.0s] The person is using a knife to cut a tortilla into four pieces on a red cutting board.
[65.0s - 73.6s] The person is using a knife to cut a tortilla into four pieces on a red cutting board.
[70.0s - 73.6s] The person is cutting a tortilla into two pieces using a knife."
REFINED ACTIONS JSON:
[{"step": "Place tortilla on cutting board", "start": 2.0, "end": 8.0}, {"step": "Use knife to scoop Nutella", "start": 12.2, "end": 17.7}, {"step": "Spread Nutella onto tortilla", "start": 17.7, "end": 28.1}, {"step": "Sprinkle cinnamon onto tortilla", "start": 33.2, "end": 40.8}, {"step": "Put banana slices on tortilla", "start": 41.8, "end": 49.3}, {"step": "Fold tortilla", "start": 50.2, "end": 57.0}, {"step": "Slice using knife", "start": 57.9, "end": 67.0}, {"step": "Place tortilla wedges on plate", "start": 67.0, "end": 70.4}]

### EXAMPLE 4
TRANSCRIPT:
"[0.0s - 10.0s] The person is using a red cutting board to place a tortilla on it.
[5.0s - 15.0s] The person is placing a tortilla on a red cutting board and then using a knife to spread Nutella on it.
[10.0s - 20.0s] The person is spreading Nutella onto a tortilla using a knife.
[15.0s - 25.0s] The person is spreading Nutella onto a tortilla using a knife.
[20.0s - 30.0s] A person is spreading Nutella onto a tortilla using a knife.
[25.0s - 35.0s] The person is spreading a chocolate spread on a tortilla using a knife.
[30.0s - 40.0s] The person is sprinkling a spice onto the tortilla using a spice shaker.
[35.0s - 45.0s] The person is spreading a dark brown spread on a white tortilla using a knife, then adding sliced bananas on top.
[40.0s - 50.0s] The person is using their hands to place banana slices onto a tortilla spread with Nutella.
[45.0s - 55.0s] The person is using a knife to cut a tortilla into a half-moon shape on a red cutting board.
[50.0s - 60.0s] The person is using a knife to cut a tortilla into two pieces on a red cutting board.
[55.0s - 65.0s] A person is using a knife to cut a quesadilla on a red cutting board.
[60.0s - 70.0s] The person is using a knife to cut a quesadilla into two halves on a red cutting board.
[65.0s - 71.1s] The person is using a knife to cut a quesadilla into four pieces and then placing them on a plate.
[70.0s - 71.1s] The person is stirring a pot on the stove using a wooden spoon."
REFINED ACTIONS JSON:
[{"step": "Place tortilla on cutting board", "start": 2.4, "end": 8.3}, {"step": "Use knife to scoop Nutella", "start": 11.2, "end": 16.3}, {"step": "Spread Nutella onto tortilla", "start": 16.3, "end": 29.2}, {"step": "Sprinkle cinnamon onto tortilla", "start": 31.0, "end": 38.7}, {"step": "Put banana slices on tortilla", "start": 39.3, "end": 46.8}, {"step": "Fold tortilla", "start": 46.8, "end": 53.1}, {"step": "Slice using knife", "start": 53.1, "end": 62.9}, {"step": "Place tortilla wedges on plate", "start": 62.9, "end": 68.5}]

### EXAMPLE 5
TRANSCRIPT:
"[0.0s - 10.0s] The person is using their hands to open a plastic bag containing tortillas.
[5.0s - 15.0s] The person is opening a bag of tortillas, taking out one tortilla, and placing it on a red cutting board.
[10.0s - 20.0s] The person is placing a tortilla on a red cutting board and then using a knife to cut it.
[15.0s - 25.0s] The person is using a knife to spread chocolate spread onto a tortilla on a red cutting board.
[20.0s - 30.0s] The person is spreading chocolate hazelnut spread onto a tortilla using a knife.
[25.0s - 35.0s] A person is spreading chocolate spread onto a tortilla using a spatula.
[30.0s - 40.0s] The person is spreading chocolate spread on a tortilla using a knife.
[35.0s - 45.0s] The person is spreading chocolate spread on a tortilla using a knife.
[40.0s - 50.0s] The person is spreading chocolate spread on a tortilla using a knife.
[45.0s - 55.0s] The person is sprinkling cocoa powder onto a tortilla using a small container, then adding banana slices on top.
[50.0s - 60.0s] The person is using a knife to cut a piece of bread into smaller pieces, then placing the pieces on a tortilla spread with chocolate spread, and finally rolling up the tortilla.
[55.0s - 65.0s] The person is folding a tortilla into a half-moon shape and then using a knife to cut it into two pieces.
[60.0s - 70.0s] The person is using a knife to cut a quesadilla on a red cutting board.
[65.0s - 75.0s] The person is using a knife to cut a quesadilla on a red cutting board.
[70.0s - 76.2s] The person is using a red cutting board to cut a quesadilla into two pieces, then placing them on a white plate.
[75.0s - 76.2s] The person is using a knife to cut something on a cutting board."
REFINED ACTIONS JSON:
[{"step": "Place tortilla on cutting board", "start": 2.8, "end": 15.0}, {"step": "Use knife to scoop Nutella", "start": 15.6, "end": 23.9}, {"step": "Spread Nutella onto tortilla", "start": 23.9, "end": 39.1}, {"step": "Sprinkle cinnamon onto tortilla", "start": 42.5, "end": 48.5}, {"step": "Put banana slices on tortilla", "start": 48.5, "end": 57.0}, {"step": "Fold tortilla", "start": 57.0, "end": 62.2}, {"step": "Slice using knife", "start": 62.2, "end": 70.6}, {"step": "Place tortilla wedges on plate", "start": 70.6, "end": 74.0}]

I will now provide the transcript I want you to clean. Please follow the style above:
"""

# ==============================================================================
# 3. REFINEMENT LOOP
# ==============================================================================
def clean_transcripts():
    client = genai.Client(api_key=API_KEY)
    transcripts = load_transcripts()
    if not transcripts:
        return

    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=-1),
        temperature=1.0,
        system_instruction="""You are a Transcript Refinement Assistant.
Your task is to take a raw, noisy VLM transcript and convert it into a clean, segmented list of actions.

FIDELITY PROTOCOL:
- DO NOT assume the person followed the recipe correctly.
- KEEP every action that occurs, even if it seems wrong, unusual, or out of order.
- KEEP any mention of spills, slips, drops, or corrections.
- KEEP any unusual ingredients or extra steps mentioned (e.g., adding honey, oats, or using a different tool).
- DO NOT skip or "correct" mistakes to make the output look like a normal recipe execution.
- Your output should be a factual representation of the observed behavior, including all errors and noise that represent real actions.

Noise to filter out:
- Repetitive phrases (e.g., "The person is...")
- Camera motion descriptions.
- Irrelevant background objects unless they are being used.
- Redundant overlapping segments (merge them into a single clean span)."""
    )

    # Process in batches of 2 to fit RPD limits
    for i in range(0, len(transcripts), 2):
        batch_num = (i // 2) + 1
        total_batches = (len(transcripts) + 1) // 2
        
        # Get names and contents separately (Anonymization)
        name_a, desc_list_a = transcripts[i]
        
        # Format the noisy descriptions (start - end: text)
        desc_a = "\n".join([f"[{d.get('start', 0.0)}s - {d.get('end', 0.0)}s] {d.get('vlm_description', '')}" for d in desc_list_a])
        
        if (i + 1) < len(transcripts):
            name_b, desc_list_b = transcripts[i+1]
            desc_b = "\n".join([f"[{d.get('start', 0.0)}s - {d.get('end', 0.0)}s] {d.get('vlm_description', '')}" for d in desc_list_b])
        else:
            name_b, desc_b = "None", "None (End of list)"

        print(f"Processing Batch {batch_num}/{total_batches} ({name_a} and {name_b})...")
        
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=EXAMPLES_TEXT),
                    types.Part.from_text(text=f'Now, refine these TWO transcripts. Return a single JSON object with keys "Video_A" and "Video_B".\n\nTRANSCRIPT A:\n{desc_a}\n\nTRANSCRIPT B:\n{desc_b}')
                ]
            )
        ]

        try:
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=contents,
                config=config,
            )
            
            clean_text = response.text
            
            # Re-associate with original filenames to reserve them in the output
            json_match = re.search(r'\{.*\}', clean_text, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group(0))
                    
                    # Record for Video A
                    record_a = {"video_name": name_a, "cleaned_actions": data.get("Video_A", [])}
                    with open(OUTPUT_JSONL, "a", encoding="utf-8") as f:
                        f.write(json.dumps(record_a) + "\n")
                    
                    # Record for Video B
                    if name_b != "None":
                        record_b = {"video_name": name_b, "cleaned_actions": data.get("Video_B", [])}
                        with open(OUTPUT_JSONL, "a", encoding="utf-8") as f:
                            f.write(json.dumps(record_b) + "\n")
                    
                    print(f"Batch {batch_num} saved to {OUTPUT_JSONL}.")
                except Exception as parse_err:
                    print(f"JSON Parse Error on batch {batch_num}: {parse_err}")
            else:
                print(f"No JSON found in response for batch {batch_num}")

            # Sleep to respect rate limits
            time.sleep(6) 
            
        except Exception as e:
            print(f"Error on batch {batch_num}: {e}")
            break

    print(f"All done! Results saved in {OUTPUT_JSONL}")

if __name__ == "__main__":
    clean_transcripts()
