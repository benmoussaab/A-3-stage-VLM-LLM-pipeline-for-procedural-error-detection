import os
import glob
import json
import torch
import cv2
import numpy as np
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# IMPORTANT: Replace these with your own paths.
VIDEO_DIR = "path/to/your/videos" 
OUTPUT_JSON = "stage1_vlm_descriptions.json"

CHUNKS_DURATION_SEC = 20.0  # Length of each video chunk
STRIDE_SEC = 10.0           # Overlap: step forward by 10 seconds (Sliding Window)
TARGET_FPS = 5.0            # Frames per second to extract

# ==============================================================================
# 1. VIDEO PROCESSING (SLIDING WINDOW)
# ==============================================================================
def get_video_chunks(video_path, chunk_sec=20.0, fps=5.0):
    """
    Splits the video into chunks (e.g., 20 seconds).
    Extracts frames at the target FPS for each chunk.
    Yields (start_time, end_time, list_of_pil_images).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    
    if total_frames == 0 or video_fps == 0:
        return
        
    duration_sec = total_frames / video_fps
    
    # Iterate through the video in overlapping chunks
    current_time = 0.0
    while current_time < duration_sec:
        end_time = min(current_time + chunk_sec, duration_sec)
        
        # Calculate which frames to pull for this specific chunk
        chunk_duration = end_time - current_time
        num_frames = int(chunk_duration * fps)
        
        frames = []
        for i in range(num_frames):
            frame_time = current_time + (i / fps)
            frame_idx = int(frame_time * video_fps)
            
            if frame_idx >= total_frames:
                break
                
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame)
                pil_img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                frames.append(pil_img)
                
        if frames:
            yield current_time, end_time, frames
            
        current_time += STRIDE_SEC  # Use STRIDE instead of chunk_sec for overlapping
        
    cap.release()

# ==============================================================================
# 2. MODEL INITIALIZATION
# ==============================================================================
print("Loading Qwen2-VL-7B model into GPU...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto" 
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

# ------------------------------------------------------------------------------
# SYSTEM PROMPT
# ------------------------------------------------------------------------------
# NOTE: This prompt is currently tailored for a Quesadilla-making task.
# Change the task context (e.g., "cooking") and the guessing logic ("ingredients/recipes")
# to fit your own specific task domain (e.g., assembly, repair, medical procedure).
# ------------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an expert AI observing a person cooking.
Watch this short 20-second video clip.

Instructions:
Your ONLY goal is to provide a purely literal, physical description of what the person is doing.
1. You MUST explicitly state WHAT tool the person is using to perform the action.
2. Only if ingredients are clear try to guess the exact names of ingredients or recipes.
3. Describe objects by their color, shape, or container.
4. Use informations mentioned in jars or bags being used to identify ingredients if possible.
5. Focus entirely on the physical motion and the objects involved.
6. If the person is doing nothing, standing still, or off-screen, output exactly "No action".

Output ONLY your single-sentence description. Do not add conversational filler or bullet points."""

# ==============================================================================
# 3. PROCESSING LOOP
# ==============================================================================
def process_videos():
    video_files = glob.glob(os.path.join(VIDEO_DIR, "*.mp4"))
    video_files.sort()
    print(f"Found {len(video_files)} videos in {VIDEO_DIR}.")

    results = {}

    for idx, video_path in enumerate(video_files):
        video_name = os.path.basename(video_path)
        print(f"\n[{idx+1}/{len(video_files)}] Processing {video_name} ...")
        
        video_segments = []
        
        for start_t, end_t, chunk_frames in get_video_chunks(video_path, chunk_sec=CHUNKS_DURATION_SEC, fps=TARGET_FPS):
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": chunk_frames,
                            "fps": TARGET_FPS,
                        },
                        {"type": "text", "text": SYSTEM_PROMPT},
                    ],
                }
            ]
            
            try:
                text_prompt = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                
                # Use processor with images/video
                image_inputs, video_inputs = processor.image_processor(messages, return_tensors="pt")
                
                inputs = processor(
                    text=[text_prompt],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to("cuda")

                # Generate output
                with torch.no_grad():
                    generated_ids = model.generate(**inputs, max_new_tokens=128)
                
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                
                description = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0].strip()
                
            except Exception as e:
                print(f"  [ERROR] {start_t:.1f}s - {end_t:.1f}s: {e}")
                description = "Error generating description."
            
            segment_data = {
                "start": round(start_t, 1),
                "end": round(end_t, 1),
                "vlm_description": description
            }
            video_segments.append(segment_data)
            
            print(f"  {start_t:.1f}s - {end_t:.1f}s -> {description[:60]}...")

        results[video_name] = video_segments
        
        # Incremental save
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)
            
    print(f"\nAll done! Output saved to {OUTPUT_JSON}")

if __name__ == "__main__":
    process_videos()
