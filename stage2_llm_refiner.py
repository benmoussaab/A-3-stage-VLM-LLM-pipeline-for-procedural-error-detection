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
MODEL_ID = "gemini-3-flash-preview"

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
"[0.0s - 10.0s] The person is using their hands to remove a tortilla from a bag and place it on a red cutting board.\n[5.0s - 15.0s] The person is taking out a tortilla from a package and placing it on a red cutting board.\n[10.0s - 20.0s] The person is spreading a chocolate spread onto a tortilla using a knife.\n[15.0s - 25.0s] The person is using a knife to spread chocolate hazelnut spread onto a tortilla.\n[20.0s - 30.0s] The person is spreading chocolate hazelnut spread onto a tortilla using a knife.\n[25.0s - 35.0s] A person is spreading chocolate spread on a tortilla using a knife.\n[30.0s - 40.0s] A person is spreading a chocolate spread on a tortilla using a knife.\n[35.0s - 45.0s] The person is spreading a chocolate spread onto a tortilla using a knife.\n[40.0s - 50.0s] A person is spreading chocolate spread on a tortilla using a knife on a red cutting board.\n[45.0s - 55.0s] The person is using a black object to spread a brown substance onto a tortilla.\n[50.0s - 60.0s] The person is using a chocolate spreader to spread chocolate hazelnut spread onto a tortilla.\n[55.0s - 65.0s] The person is spreading a chocolate spread onto a tortilla using a spoon.\n[60.0s - 70.0s] The person is spreading chocolate spread on a tortilla and then adding quick oats from a container onto the tortilla.\n[65.0s - 75.0s] The person is pouring quick oats from a container onto a tortilla spread with chocolate spread, using a spoon to scoop the oats.\n[70.0s - 80.0s] The person is spreading a chocolate spread on a tortilla and then folding it into a half-moon shape using their hands.\n[75.0s - 85.0s] The person is folding a tortilla into a triangle using their hands.\n[80.0s - 90.0s] The person is using their hands to fold a tortilla into a triangle, then unfolding it and folding it again into a triangle.\n[85.0s - 95.0s] The person is tearing a tortilla into two pieces using their hands.\n[90.0s - 100.0s] The person is cutting a tortilla into four pieces using a knife on a red cutting board.\n[95.0s - 105.0s] The person is using their hands to pick up tortillas from a table and place them into a bowl.\n[100.0s - 105.0s] The person is using a knife to cut a tortilla."
REFINED ACTIONS JSON:
[
    {
      "step": "Place tortilla on table",
      "start": 2.4,
      "end": 10.6
    },
    {
      "step": "Use knife to scoop Nutella",
      "start": 13.2,
      "end": 23.1
    },
    {
      "step": "Spread Nutella onto tortilla",
      "start": 23.1,
      "end": 39.2
    },
    {
      "step": "Sprinkle cinnamon onto tortilla",
      "start": 48.0,
      "end": 57.4
    },
    {
      "step": "Pour a handful of oatmeals on tortilla",
      "start": 60.0,
      "end": 73.1
    },
    {
      "step": "Fold tortilla",
      "start": 73.8,
      "end": 85.4
    },
    {
      "step": "Rip tortilla by hands",
      "start": 85.4,
      "end": 94.3
    },
    {
      "step": "Place tortilla wedges into bowl",
      "start": 94.3,
      "end": 102.0
    }
  ]
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
"[0.0s - 10.0s] The person is using their hands to take out a tortilla from a bag of Mission flour tortillas.\n[5.0s - 15.0s] The person is placing a tortilla on the floor, then picking it up and putting it back on the table.\n[10.0s - 20.0s] The person picks up a tortilla from the floor, places it on a cutting board, and uses a knife to cut it.\n[15.0s - 25.0s] The person is taking out a tortilla from a plastic bag, placing it on a red cutting board, and then cutting it into four equal pieces using a knife.\n[20.0s - 30.0s] A person is placing a tortilla on a red cutting board and then reaching for a box of raisins.\n[25.0s - 35.0s] The person is using their hands to open a red box of raisins and then sprinkling the raisins onto a tortilla on a red cutting board.\n[30.0s - 40.0s] The person is pouring raisins onto a tortilla using a red box of raisins.\n[35.0s - 45.0s] The person is sprinkling raisins onto a tortilla using a red container, then adding banana slices onto the tortilla.\n[40.0s - 50.0s] The person is using a knife to cut a tortilla on a red cutting board, and they are adding sliced bananas to the tortilla.\n[45.0s - 55.0s] The person is using their hand to place banana slices onto a tortilla on a red cutting board.\n[50.0s - 60.0s] The person is using their hands to fold a tortilla into a triangle on a red cutting board.\n[55.0s - 65.0s] The person is folding a tortilla into a triangle and then using a knife to cut it in half.\n[60.0s - 70.0s] A person is using a knife to cut a tortilla on a red cutting board.\n[65.0s - 75.0s] A person is using a knife to cut a tortilla into two pieces on a red cutting board.\n[70.0s - 80.0s] The person is using a knife to cut a tortilla into two pieces, then placing the pieces on a plate.\n[75.0s - 81.6s] The person is using a knife to cut a tortilla into four pieces and then placing them on a plate.\n[80.0s - 81.6s] The person is using a knife to cut a tortilla on a red cutting board."
REFINED ACTIONS JSON:
[
    {
      "step": "Drop tortilla",
      "start": 2.3,
      "end": 9.4
    },
    {
      "step": "Discard tortilla and place a new one",
      "start": 9.4,
      "end": 19.4
    },
    {
      "step": "Add a handful of raisins to tortilla",
      "start": 26.3,
      "end": 39.0
    },
    {
      "step": "Put banana slices on tortilla",
      "start": 41.1,
      "end": 50.4
    },
    {
      "step": "Fold tortilla",
      "start": 53.7,
      "end": 61.6
    },
    {
      "step": "Slice using knife",
      "start": 61.6,
      "end": 71.8
    },
    {
      "step": "Place tortilla wedges on plate",
      "start": 73.5,
      "end": 79.5
    }
  ]
### EXAMPLE 4
TRANSCRIPT:
"[0.0s - 10.0s] The person is taking out a tortilla from a plastic bag using their hands.\n[5.0s - 15.0s] The person is spreading chocolate hazelnut spread onto a tortilla using a knife.\n[10.0s - 20.0s] The person is spreading a chocolate spread onto a tortilla using a knife.\n[15.0s - 25.0s] The person is spreading a chocolate spread onto a tortilla using a knife.\n[20.0s - 30.0s] The person is spreading a chocolate spread on a tortilla using a knife.\n[25.0s - 35.0s] The person is spreading a brown substance onto a tortilla using a spoon and then adding pieces of a yellow and black bag of chips onto the tortilla.\n[30.0s - 40.0s] The person is pouring raisins from a yellow and black container onto a tortilla, then adding a dark brown spread from a bottle with a black cap.\n[35.0s - 45.0s] The person is sprinkling a spice onto the tortilla using a shaker with a black lid.\n[40.0s - 50.0s] The person is sprinkling cinnamon from a shaker onto a tortilla that has been spread with a brown substance, then folding the tortilla into a triangle.\n[45.0s - 55.0s] The person is sprinkling cinnamon from a bottle onto a tortilla, then folding the tortilla into a half-moon shape and cutting it with a knife.\n[50.0s - 60.0s] The person is using a knife to cut a tortilla into two pieces on a red cutting board.\n[55.0s - 65.0s] The person is using a knife to cut a tortilla into two pieces on a red cutting board.\n[60.0s - 65.5s] The person is wrapping a sandwich with a tortilla using their hands, then placing it on a white plate.\n[65.0s - 65.5s] A person is pointing at a wooden cabinet."

REFINED ACTIONS JSON:
[
    {
      "step": "Place tortilla on cutting board",
      "start": 2.3,
      "end": 7.9
    },
    {
      "step": "Use knife to scoop Nutella",
      "start": 7.9,
      "end": 16.1
    },
    {
      "step": "Spread Nutella onto tortilla",
      "start": 16.1,
      "end": 24.7
    },
    {
      "step": "Put banana slices on tortilla",
      "start": 26.0,
      "end": 36.1
    },
    {
      "step": "Sprinkle cinnamon onto tortilla",
      "start": 36.1,
      "end": 47.0
    },
    {
      "step": "Fold tortilla",
      "start": 47.0,
      "end": 50.6
    },
    {
      "step": "Slice using knife",
      "start": 50.6,
      "end": 58.3
    },
    {
      "step": "Place tortilla wedges on plate",
      "start": 58.3,
      "end": 62.1
    }
  ]
                                         
### EXAMPLE 5
TRANSCRIPT:
"[0.0s - 10.0s] The person is taking out a tortilla from a bag and placing it on a red cutting board.\n[5.0s - 15.0s] A person is placing a tortilla on a red cutting board and then opening a jar of Nutella.\n[10.0s - 20.0s] The person is spreading Nutella onto a tortilla using a knife.\n[15.0s - 25.0s] The person is spreading Nutella onto a tortilla using a knife.\n[20.0s - 30.0s] The person is spreading Nutella onto a tortilla using a knife.\n[25.0s - 35.0s] The person is spreading Nutella onto a tortilla using a knife.\n[30.0s - 40.0s] The person is spreading Nutella on a tortilla using a knife and then adding banana slices on top.\n[35.0s - 45.0s] The person is using their hands to place banana slices on a tortilla spread with Nutella, while other ingredients and tools are visible on the table.\n[40.0s - 50.0s] The person is using a knife to spread chocolate spread on a tortilla, which is placed on a red cutting board.\n[45.0s - 55.0s] The person is sprinkling cocoa powder onto a tortilla using a small container with a black lid.\n[50.0s - 60.0s] The person is sprinkling cocoa powder onto a tortilla, then folding it into a burrito.\n[55.0s - 65.0s] The person is using their hands to fold a tortilla into a half-moon shape on a red cutting board.\n[60.0s - 70.0s] The person is pressing a folded tortilla with their hands on a red cutting board, then using a knife to cut the tortilla into two halves.\n[65.0s - 75.0s] The person is using a knife to cut a tortilla into two pieces on a red cutting board.\n[70.0s - 79.3s] The person is spreading a white substance on a tortilla using a knife, then folding the tortilla in half and placing it on a plate.\n[75.0s - 79.3s] The person is placing a jar on the table, which appears to be a chocolate spread, next to a plate with a tortilla and a spoon."
                                         
REFINED ACTIONS JSON:
[
    {
      "step": "Place tortilla on cutting board",
      "start": 2.9,
      "end": 7.4
    },
    {
      "step": "Use knife to scoop Nutella",
      "start": 12.1,
      "end": 18.6
    },
    {
      "step": "Spread Nutella onto tortilla",
      "start": 18.6,
      "end": 32.8
    },
    {
      "step": "Put banana slices on tortilla",
      "start": 32.8,
      "end": 41.6
    },
    {
      "step": "Sprinkle cinnamon onto tortilla",
      "start": 44.7,
      "end": 54.3
    },
    {
      "step": "Fold tortilla",
      "start": 56.4,
      "end": 62.4
    },
    {
      "step": "Slice using knife",
      "start": 63.6,
      "end": 71.4
    },
    {
      "step": "Place tortilla wedges on plate",
      "start": 71.4,
      "end": 75.6
    }
  ]

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
        thinking_config=types.ThinkingConfig(thinking_level="HIGH"),
        temperature=0.0,
        system_instruction="""You are a Transcript Refinement Assistant.
Your task is to take a raw, noisy VLM transcript and convert it into a clean, segmented list of actions.

FIDELITY PROTOCOL:
- DO NOT assume the person followed the recipe correctly.
- KEEP every action that occurs, even if it seems wrong, unusual, or out of order.
- KEEP any mention of spills, slips, drops, or corrections.
- KEEP any unusual ingredients or extra steps mentioned .
- DO NOT skip or "correct" mistakes to make the output look like a normal recipe execution.
- Your output should be a factual representation of the observed behavior, including all errors and noise that represent real actions.

Noise to filter out:
- Repetitive phrases 
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
