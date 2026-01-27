"""
Centralized artifact saver for evaluation outputs.
Handles all file organization and saving logic.
"""

import os
import json
import numpy as np
import cv2
from typing import Optional, Dict, List

from src.evaluation.sam_grid_extractor import PieceDetection, BOARD_SIZE


class DataSaver:
    """Saves evaluation artifacts in an organized directory structure."""
    
    def __init__(self, output_dir: str, gt_images_dir: Optional[str] = None):
        self.output_dir = output_dir
        self.debug_dir = os.path.join(output_dir, "debug")
        os.makedirs(self.debug_dir, exist_ok=True)
        
        self.gt_file_map = {}
        if gt_images_dir and os.path.isdir(gt_images_dir):
            for f in os.listdir(gt_images_dir):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.gt_file_map[os.path.splitext(f)[0]] = os.path.join(gt_images_dir, f)
            print(f"DataSaver: Found {len(self.gt_file_map)} GT images")
        
        print(f"DataSaver: Output dir = {output_dir}")
    
    def _get_sample_dir(self, file_id: str) -> str:
        sample_dir = os.path.join(self.debug_dir, file_id)
        os.makedirs(sample_dir, exist_ok=True)
        return sample_dir
    
    def save_input_image(self, file_id: str, image: np.ndarray):
        sample_dir = self._get_sample_dir(file_id)
        cv2.imwrite(os.path.join(sample_dir, "input.png"), image)
    
    def save_gt_image(self, file_id: str):
        if file_id not in self.gt_file_map:
            return
        sample_dir = self._get_sample_dir(file_id)
        img = cv2.imread(self.gt_file_map[file_id])
        if img is not None:
            cv2.imwrite(os.path.join(sample_dir, "gt_original.png"), img)
    
    def save_grid(self, file_id: str, grid: np.ndarray, name: str):
        sample_dir = self._get_sample_dir(file_id)
        combined = grid[0] + grid[1] * 2
        np.savetxt(os.path.join(sample_dir, f"{name}.txt"), combined, fmt='%d')
        self._save_grid_visual(grid, os.path.join(sample_dir, f"{name}.png"))
    
    def save_detections_debug(
        self, 
        file_id: str, 
        image: np.ndarray, 
        detections: List[PieceDetection],
        grid: np.ndarray
    ):
        """Save detection visualization with centroids, masks, and grid overlay."""
        sample_dir = self._get_sample_dir(file_id)
        h, w = image.shape[:2]
        board_size = BOARD_SIZE
        cell_h, cell_w = h / board_size, w / board_size
        
        # Calculate reference area for percentage display
        all_areas = [d.area for d in detections]
        mean_area = np.mean(all_areas) if all_areas else 1.0
        
        # Debug image with detections
        debug_img = image.copy()
        mask_white = np.zeros((h, w), dtype=bool)
        mask_black = np.zeros((h, w), dtype=bool)
        
        for det in detections:
            cx, cy = det.centroid
            area_pct = (det.area / mean_area) * 100
            
            if det.filtered:
                color = (128, 128, 128)
                cv2.circle(debug_img, (int(cx), int(cy)), 4, color, -1)
                size = 6
                cv2.line(debug_img, (int(cx)-size, int(cy)-size), (int(cx)+size, int(cy)+size), color, 1)
                cv2.line(debug_img, (int(cx)-size, int(cy)+size), (int(cx)+size, int(cy)-size), color, 1)
                mask_uint8 = det.mask.astype(np.uint8) * 255
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(debug_img, contours, -1, color, 1)
                cv2.putText(debug_img, f"X {area_pct:.0f}%", (int(cx) + 8, int(cy)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            else:
                color = (0, 0, 255) if det.is_white else (255, 0, 0)
                cv2.circle(debug_img, (int(cx), int(cy)), 6, color, -1)
                cv2.circle(debug_img, (int(cx), int(cy)), 8, (0, 255, 255), 2)
                
                mask_uint8 = det.mask.astype(np.uint8) * 255
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(debug_img, contours, -1, color, 2)
                
                if det.grid_row >= 0:
                    piece_type = 'W' if det.is_white else 'B'
                    cv2.putText(debug_img, f"{piece_type}({det.grid_row},{det.grid_col}) {area_pct:.0f}%", 
                               (int(cx) + 10, int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                if det.is_white:
                    mask_white = np.logical_or(mask_white, det.mask)
                else:
                    mask_black = np.logical_or(mask_black, det.mask)
        
        # Draw grid lines
        for r in range(board_size + 1):
            cv2.line(debug_img, (0, int(r * cell_h)), (w, int(r * cell_h)), (0, 255, 0), 1)
        for c in range(board_size + 1):
            cv2.line(debug_img, (int(c * cell_w), 0), (int(c * cell_w), h), (0, 255, 0), 1)
        
        cv2.imwrite(os.path.join(sample_dir, "detections.png"), debug_img)
        cv2.imwrite(os.path.join(sample_dir, "mask_white.png"), mask_white.astype(np.uint8) * 255)
        cv2.imwrite(os.path.join(sample_dir, "mask_black.png"), mask_black.astype(np.uint8) * 255)
    
    def save_summary(self, summary: Dict):
        summary_path = os.path.join(self.output_dir, "summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved to {summary_path}")
    
    def _save_grid_visual(self, grid: np.ndarray, save_path: str, cell_size: int = 50):
        board_size = grid.shape[1]
        h, w = board_size * cell_size, board_size * cell_size
        img = np.zeros((h, w, 3), dtype=np.uint8)
        
        for r in range(board_size):
            for c in range(board_size):
                y1, y2 = r * cell_size, (r + 1) * cell_size
                x1, x2 = c * cell_size, (c + 1) * cell_size
                
                is_white = grid[0, r, c] > 0.5
                is_black = grid[1, r, c] > 0.5
                
                if is_white and is_black:
                    color = (0, 255, 255)
                elif is_white:
                    color = (0, 255, 0)
                elif is_black:
                    color = (0, 0, 255)
                else:
                    color = (0, 0, 0)
                
                cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
                cv2.rectangle(img, (x1, y1), (x2, y2), (50, 50, 50), 1)
        
        cv2.imwrite(save_path, img)
