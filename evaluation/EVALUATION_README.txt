================================================================================
                    CHESS BOARD EVALUATION PIPELINE
                         Technical Documentation
================================================================================

OVERVIEW
--------
This evaluation folder contains a complete pipeline for measuring the accuracy
of chess board image generation models. The goal is to compare generated images
against ground truth FEN and compute
meaningful metrics for piece detection and placement accuracy.


================================================================================
                              THE PROBLEM
================================================================================

When generating chess board images, we need to verify:
1. Are all pieces present in the generated image?
2. Are they in the correct positions?
3. Are white and black pieces correctly distinguished?

The challenge: We have FEN strings as ground truth (e.g., "rnbqkbnr/pppppppp/...")
but we need to extract piece positions from raw images.


================================================================================
                           SOLUTION ARCHITECTURE
================================================================================

The pipeline works in these stages:

    [Generated Image] --> [SAM Detection] --> [Grid Extraction] --> [Metrics]
                                                     |
    [FEN String] ------> [Grid Conversion] ----------+


================================================================================
                              MODULE BREAKDOWN
================================================================================

1. FEN TO GRID CONVERSION (fen_to_grid.py)
------------------------------------------
Purpose: Convert FEN notation to a 2-channel 8x8 grid representation.

Input:  FEN string, e.g., "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
Output: Tensor/Array of shape (2, 8, 8)
        - Channel 0: White pieces (1.0 where white piece exists, 0.0 elsewhere)
        - Channel 1: Black pieces (1.0 where black piece exists, 0.0 elsewhere)

How it works:
- Parses FEN placement string (first part before space)
- Iterates through rows (separated by '/')
- Uppercase letters = white pieces, lowercase = black pieces
- Numbers indicate consecutive empty squares

##add an example of matrix##

2. SAM GRID EXTRACTOR (sam_grid_extractor_with_centroids.py)
------------------------------------------------------------
Purpose: Extract chess piece positions from an image using SAM (Segment Anything
Model) with per-detection centroid-based grid assignment.

Key Innovation: Instead of creating merged masks for all white/black pieces
(which loses individual piece identity), we process EACH SAM detection 
separately.

Pipeline:
   a) Run SAM with prompt "chess piece" to get all detections
   b) For EACH detection:
      - Compute centroid (center of mass)
      - Classify as white/black based on pixel brightness (median > 110)
      - Determine which grid cell contains the centroid
   c) Filter small detections to remove false positives

Filtering Strategy (Mean-Based):
   - Problem: SAM sometimes splits one piece into multiple small fragments
   - Solution: Filter detections smaller than 50% of the MEAN detection area
   - Example: If average piece is 100 pixels, remove detections < 50 pixels
   - This adaptively filters based on the actual detections in each image

Output: (2, 8, 8) grid + list of PieceDetection objects with:
   - mask: Boolean array of the detection
   - centroid: (cx, cy) coordinates
   - is_white: Boolean classification
   - area: Pixel count
   - grid_row, grid_col: Assigned cell position
   - filtered: Whether it was filtered out


##add an example of sam detection##

3. BOARD METRICS (board_metrics.py)
-----------------------------------
Purpose: Compute accuracy metrics between predicted and ground truth grids.

Metrics Provided:

a) Overall Accuracy (compute_board_accuracy):
   - Element-wise match rate across all 128 values (2 channels x 8x8)
   - Simple but can be misleading (empty squares dominate)

b) Cell-Level Accuracy (compute_cell_level_accuracy):
   - A cell is "correct" only if BOTH channels match
   - More meaningful: measures per-square correctness

c) Per-Channel Accuracy (compute_per_channel_accuracy):
   - Separate accuracy for white and black channels
   - Helps diagnose color-specific issues

d) F1 Metrics (compute_f1_metrics):
   - Precision: Of all predicted pieces, how many are correct?
   - Recall: Of all actual pieces, how many were found?
   - F1 Score: Harmonic mean of precision and recall
   - Computed separately for white and black pieces

   Definitions:
   - True Positive (TP):  Predicted 1, Actual 1 (correct detection)
   - False Positive (FP): Predicted 1, Actual 0 (hallucination)
   - False Negative (FN): Predicted 0, Actual 1 (missed piece)


4. DATA LOADER (fen_data_loader.py)
-----------------------------------
Purpose: Load evaluation data from CSV files with FEN labels.

Input:
- CSV with columns: 'IMG_NAME', 'FEN'
- Directory containing the images

Output: Iterator yielding (image_path, fen_string, image_array)

Features:
- Validates that images exist
- Reports missing files
- Supports batch iteration


5. DATA SAVER (data_saver.py)
-----------------------------
Purpose: Organize and save all evaluation artifacts.

Directory Structure Created:
   output_dir/
   ├── debug/
   │   ├── sample_001/
   │   │   ├── input.png          # The generated image
   │   │   ├── gt_original.png    # Ground truth image (if available)
   │   │   ├── grid_gt.txt        # Ground truth grid as text
   │   │   ├── grid_gt.png        # Ground truth grid visualization
   │   │   ├── grid_pred.txt      # Predicted grid as text
   │   │   ├── grid_pred.png      # Predicted grid visualization
   │   │   ├── detections.png     # Debug image with centroids/masks
   │   │   ├── mask_white.png     # Combined white piece mask
   │   │   └── mask_black.png     # Combined black piece mask
   │   ├── sample_002/
   │   │   └── ...
   └── summary.json               # Aggregated metrics

Debug Visualization (detections.png):
- Green grid overlay showing 8x8 cell boundaries
- Red circles with yellow outline: White piece centroids
- Blue circles with yellow outline: Black piece centroids
- Gray X markers: Filtered detections (too small)
- Mask outlines showing SAM segmentation
- Labels: "W(row,col) 85%" or "B(row,col) 120%" (area as % of mean)


6. MAIN EVALUATOR (evaluate_model.py)
-------------------------------------
Purpose: Orchestrate the full evaluation pipeline.

Three Evaluation Modes:

a) evaluate_single(image, fen, file_id, extractor, saver):
   - Evaluate a single pre-loaded image
   - Returns metrics dict with grids and detections

b) evaluate_batch(images, fens, file_ids, extractor, saver):
   - Evaluate multiple pre-loaded images
   - Returns list of metrics dicts

c) evaluate_folder(generated_dir, csv_path, output_dir, ...):
   - Load images from directory using FenDataLoader
   - Process all images
   - Compute aggregated summary statistics
   - Save all debug outputs

Command Line Usage:
   python evaluate_model.py \
       --generated_dir path/to/outputs \
       --csv_path path/to/data.csv \
       --output_dir path/to/results \
       --gt_images_dir path/to/ground_truth_images


================================================================================
                         WHY THIS APPROACH?
================================================================================

Problem with Previous Approaches:
---------------------------------
1. MERGED MASKS: Earlier we created one white mask + one black mask by OR-ing
   all SAM detections. This caused adjacent pieces to merge into single blobs,
   making centroid detection unreliable.

2. CELL-AREA THRESHOLD: Using a fixed fraction of cell area as threshold 
   doesn't adapt to image content. Different images may have pieces of 
   different apparent sizes.

Current Solution:
-----------------
1. PER-DETECTION PROCESSING: Each SAM detection is processed individually 
   before any merging. Centroids are computed per-piece, not per-merged-blob.

2. MEAN-BASED FILTERING: The threshold adapts to each image. If most pieces
   are ~100 pixels, we filter < 50 pixels. If pieces are larger (200 pixels),
   we filter < 100 pixels. This handles varying piece sizes across images.

3. CENTROID-BASED ASSIGNMENT: A piece belongs to exactly ONE cell - the cell
   containing its centroid. This prevents pieces that span cell boundaries
   from being counted in multiple cells.


================================================================================
                              LEGACY FILES
================================================================================

The legacy/ folder contains older implementations:
- mask_to_grid.py: Area-coverage based grid assignment (deprecated)
- mask_to_grid_centroid.py: Centroid-based but used merged masks (deprecated)
- sam_mask_extractor.py: SAM extraction with mask merging (deprecated)
- calculate_accuracy.py: Early accuracy metrics (superseded by board_metrics.py)

These are kept for reference but should not be used for new evaluations.


================================================================================
                              EXAMPLE OUTPUT
================================================================================

EVALUATION SUMMARY
==================================================
Images Evaluated: 50
Average Cell Accuracy: 87.34%
Average White - F1: 82.15%, Precision: 85.20%, Recall: 79.30%
Average Black - F1: 84.67%, Precision: 88.10%, Recall: 81.50%
Error Rate: 12.66%

Per-image output:
  sample_001: Cell Acc=91.40%, W-F1=88.00% (R=85.00%), B-F1=90.00% (R=87.00%)
  sample_002: Cell Acc=85.90%, W-F1=78.00% (R=72.00%), B-F1=82.00% (R=78.00%)
  ...


================================================================================
                         TUNABLE PARAMETERS
================================================================================

SAMGridExtractor:
- conf (default: 0.25): SAM confidence threshold for detections
- brightness_threshold (default: 110): Grayscale value to classify white/black
- min_area_fraction (default: 0.5): Filter detections < 50% of mean area

These can be adjusted based on your specific image characteristics.


================================================================================
                              SUMMARY
================================================================================

This evaluation pipeline provides a robust way to measure chess board generation
quality by:

1. Converting FEN ground truth to comparable grid format
2. Using SAM to detect pieces in generated images
3. Processing each detection individually to avoid merging artifacts
4. Adaptive filtering based on mean detection area
5. Computing multiple accuracy metrics (cell accuracy, F1, precision, recall)
6. Saving comprehensive debug outputs for analysis

The key insight is treating piece detection as a per-instance problem rather
than a semantic segmentation problem, which better matches the discrete nature
of chess positions.
