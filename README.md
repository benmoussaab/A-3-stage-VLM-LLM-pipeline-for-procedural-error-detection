# EgoPER 3-Stage Pipeline: Egocentric Procedural Error Recognition

This repository contains the inference code for a **3-Stage Video-Language Model (VLM) + Large Language Model (LLM) pipeline** designed to detect normal execution, procedural modifications, errors, and omitted steps in egocentric (first-person) cooking/assembly videos.

## Overview of the 3 Stages

1. **Stage 1 (VLM Describer)**: 
   Uses a Vision-Language Model (e.g., `Qwen2-VL-7B-Instruct`) to process video chunks and generate dense, purely literal textual descriptions of the physical actions occurring in the scene.
2. **Stage 2 (LLM Refiner)**: 
   Uses a powerful LLM (e.g., `Gemini 2.5 Flash`) to refine the noisy VLM descriptions. It filters out irrelevant camera motion and object details, structuring the output into a chronological, segmented list of JSON actions with start and end times.
3. **Stage 3 (Mistake Detector)**: 
   Uses an advanced LLM (e.g., `Gemini 3 Flash Preview`) as a high-precision auditor. It compares the chronological actions from Stage 2 against a predefined **Perfect Task Graph** to accurately classify steps as `Normal`, `Error`, or `Omission`.

## Setup and Requirements

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Set your Google Gemini API Key as an environment variable:
   ```bash
   export GEMINI_API_KEY="your_api_key_here"
   ```

## Usage Instructions

> **IMPORTANT**: The prompt templates, few-shot examples, and task graph included in these scripts are currently tailored for a **Quesadilla cooking task**. You **MUST** adapt the prompts, provide your own few-shot examples, and define your own Task Graph before running this on a different dataset.

### 1. Stage 1: VLM Describer
1. Open `stage1_vlm_describer.py`.
2. Edit `VIDEO_DIR` to point to the directory containing your raw `.mp4` video clips.
3. Update the `SYSTEM_PROMPT` to fit your specific domain (e.g., replace ingredients logic if doing a repair task).
4. Run the script:
   ```bash
   python stage1_vlm_describer.py
   ```
   *Output:* `stage1_vlm_descriptions.json`

### 2. Stage 2: LLM Refiner
1. Open `stage2_llm_refiner.py`.
2. The script reads from `stage1_vlm_descriptions.json` by default.
3. **CRITICAL**: Update the `EXAMPLES_TEXT` variable with your own noisy transcript vs. clean JSON few-shot pairings.
4. Run the script:
   ```bash
   python stage2_llm_refiner.py
   ```
   *Output:* `stage2_refined_actions.jsonl`

### 3. Stage 3: Mistake Detector
1. Open `stage3_mistake_detector.py`.
2. The script reads from `stage2_refined_actions.jsonl` by default.
3. **CRITICAL**: Update `TASK_GRAPH` with your own chronological sequence of expected steps.
4. **CRITICAL**: Update `FEW_SHOT_EXAMPLES` to map your Stage 2 JSON actions to the expected Final Audit Output (`Normal`, `Error`, `Omission`).
5. Run the script:
   ```bash
   python stage3_mistake_detector.py
   ```
   *Output:* `stage3_error_detection_results.jsonl`

## Citation
If you use this pipeline in your research, please cite the original EgoPER framework as well as the underlying foundation models (Qwen2-VL, Gemini) as appropriate.
