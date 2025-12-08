"""
Interactive VLM Models Testing Demo

This script allows you to test different VLM models interactively:
- Grounding DINO: Open-vocabulary object detection
- YOLOv7: COCO object detection
- Mobile SAM: Segmentation with bounding box
- BLIP2: Visual Question Answering
- BLIP2-ITM: Image-Text Matching

Requirements:
    1. VLM servers must be running on cloud server
    2. Configure server host and ports below
    3. Prepare test images in a directory

Usage:
    python test_vlm_models.py
"""

import os
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml
import threading

# Add vlfm to path
sys.path.insert(0, str(Path(__file__).parent))

from vlfm.vlm.blip2 import BLIP2Client
from vlfm.vlm.blip2itm import BLIP2ITMClient
from vlfm.vlm.grounding_dino import GroundingDINOClient
from vlfm.vlm.sam import MobileSAMClient
from vlfm.vlm.yolov7 import YOLOv7Client


class VLMTester:
    """Interactive tester for VLM models."""

    def __init__(self, config_path: str = "behavior.yaml"):
        """
        Initialize VLM clients from config.

        Args:
            config_path: Path to configuration YAML file
        """
        # Load configuration
        config_file = Path(__file__).parent / config_path
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            vlm_config = config["vlm_servers"]
        else:
            print(f"Config file {config_file} not found. Using default localhost settings.")
            vlm_config = {
                "grounding_dino": {"host": "localhost", "port": 12181},
                "yolov7": {"host": "localhost", "port": 12184},
                "mobile_sam": {"host": "localhost", "port": 12183},
                "blip2": {"host": "localhost", "port": 12185},
                "blip2itm": {"host": "localhost", "port": 12182},
            }

        # Initialize clients
        print("Initializing VLM clients...")
        print(f"  Grounding DINO: {vlm_config['grounding_dino']['host']}:{vlm_config['grounding_dino']['port']}")
        self.grounding_dino = GroundingDINOClient(
            host=vlm_config["grounding_dino"]["host"],
            port=vlm_config["grounding_dino"]["port"]
        )

        print(f"  YOLOv7:         {vlm_config['yolov7']['host']}:{vlm_config['yolov7']['port']}")
        self.yolov7 = YOLOv7Client(
            host=vlm_config["yolov7"]["host"],
            port=vlm_config["yolov7"]["port"]
        )

        print(f"  Mobile SAM:     {vlm_config['mobile_sam']['host']}:{vlm_config['mobile_sam']['port']}")
        self.mobile_sam = MobileSAMClient(
            host=vlm_config["mobile_sam"]["host"],
            port=vlm_config["mobile_sam"]["port"]
        )

        print(f"  BLIP2 (VQA):    {vlm_config['blip2']['host']}:{vlm_config['blip2']['port']}")
        self.blip2 = BLIP2Client(
            host=vlm_config["blip2"]["host"],
            port=vlm_config["blip2"]["port"]
        )

        print(f"  BLIP2-ITM:      {vlm_config['blip2itm']['host']}:{vlm_config['blip2itm']['port']}")
        self.blip2itm = BLIP2ITMClient(
            host=vlm_config["blip2itm"]["host"],
            port=vlm_config["blip2itm"]["port"]
        )

        print("✓ All clients initialized\n")

        # Output directory
        self.output_dir = Path("vlm_test_results")
        self.output_dir.mkdir(exist_ok=True)
        print(f"Results will be saved to: {self.output_dir.absolute()}\n")

    def load_image(self, image_path: str) -> Optional[np.ndarray]:
        """Load image from file."""
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            return None

        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Failed to load image: {image_path}")
            return None

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(f"✓ Loaded image: {image_path} (shape: {image.shape})")
        return image

    def draw_text_top(self, image, text, color=(255,255,255), font_scale=0.7, thickness=2):
        """在图片最上方居中绘制文本"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
        w = image.shape[1]
        x = (w - text_size[0]) // 2
        y = 30
        cv2.putText(image, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
        return image

    def run_with_timeout(self, func, *args, timeout: float = 5.0, **kwargs):
        """Run `func(*args, **kwargs)` in a daemon thread and return the result.

        If the call does not complete within `timeout` seconds, return None
        (no retries) and print a timeout message. The worker thread is marked
        as daemon so it won't block the main program if it continues running.
        """
        result_container = {}

        def _target():
            try:
                result_container['result'] = func(*args, **kwargs)
            except Exception as e:
                result_container['exc'] = e

        t = threading.Thread(target=_target, daemon=True)
        t.start()
        t.join(timeout)
        if t.is_alive():
            print(f"❌ Timeout: operation exceeded {timeout}s — skipping this model.")
            return None
        if 'exc' in result_container:
            raise result_container['exc']
        return result_container.get('result')

    def test_grounding_dino(self, image: np.ndarray, caption: str):
        """Test Grounding DINO object detection."""
        print(f"\n{'='*80}")
        print("Testing Grounding DINO (Open-vocabulary Object Detection)")
        print(f"{'='*80}")
        print(f"Caption: {caption}")

        try:
            detections = self.run_with_timeout(self.grounding_dino.predict, image, caption=caption, timeout=5)
            if detections is None:
                return None
            print(f"✓ Detected {detections.num_detections} objects")

            # Visualize
            annotated = detections.annotated_frame.copy()
            # 在图片最上方绘制提示词
            annotated = self.draw_text_top(annotated, f"Prompt: {caption}", color=(255,255,0))
            for i in range(detections.num_detections):
                label = f"{detections.phrases[i]}: {detections.logits[i]:.2f}"
                print(f"  [{i+1}] {label}")

            # Save result
            output_path = self.output_dir / "grounding_dino_result.jpg"
            cv2.imwrite(str(output_path), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
            print(f"✓ Saved result to: {output_path}")

            return detections

        except Exception as e:
            print(f"❌ Error: {e}")
            return None

    def test_yolov7(self, image: np.ndarray):
        """Test YOLOv7 COCO object detection."""
        print(f"\n{'='*80}")
        print("Testing YOLOv7 (COCO Object Detection)")
        print(f"{'='*80}")

        try:
            detections = self.run_with_timeout(self.yolov7.predict, image, timeout=5)
            if detections is None:
                return None
            print(f"✓ Detected {detections.num_detections} objects")

            # Visualize
            annotated = detections.annotated_frame.copy()
            annotated = self.draw_text_top(annotated, "YOLOv7 COCO Detection", color=(0,255,255))
            for i in range(detections.num_detections):
                label = f"{detections.phrases[i]}: {detections.logits[i]:.2f}"
                print(f"  [{i+1}] {label}")

            # Save result
            output_path = self.output_dir / "yolov7_result.jpg"
            cv2.imwrite(str(output_path), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
            print(f"✓ Saved result to: {output_path}")

            return detections

        except Exception as e:
            print(f"❌ Error: {e}")
            return None

    def test_mobile_sam(self, image: np.ndarray, bbox: list):
        """Test Mobile SAM segmentation."""
        print(f"\n{'='*80}")
        print("Testing Mobile SAM (Segmentation)")
        print(f"{'='*80}")
        print(f"Bounding box: {bbox} (format: [x1, y1, x2, y2])")

        try:
            mask = self.run_with_timeout(self.mobile_sam.segment_bbox, image, bbox, timeout=5)
            if mask is None:
                return None
            print(f"✓ Generated segmentation mask (shape: {mask.shape})")
            print(f"  Segmented pixels: {np.sum(mask > 0)} / {mask.size} ({np.sum(mask > 0) / mask.size * 100:.1f}%)")

            # Visualize
            annotated = image.copy()
            # Draw bounding box
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Overlay mask
            mask_overlay = np.zeros_like(annotated)
            mask_overlay[mask > 0] = [255, 0, 0]  # Red for mask
            annotated = cv2.addWeighted(annotated, 0.7, mask_overlay, 0.3, 0)
            # 在图片最上方绘制 box 信息
            annotated = self.draw_text_top(annotated, f"Box: {bbox}", color=(0,255,0))

            # Save result
            output_path = self.output_dir / "mobile_sam_result.jpg"
            cv2.imwrite(str(output_path), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
            print(f"✓ Saved result to: {output_path}")

            # Save mask separately
            mask_path = self.output_dir / "mobile_sam_mask.png"
            cv2.imwrite(str(mask_path), mask.astype(np.uint8) * 255)
            print(f"✓ Saved mask to: {mask_path}")

            return mask

        except Exception as e:
            print(f"❌ Error: {e}")
            return None

    def test_blip2(self, image: np.ndarray, question: str):
        return ""
        """Test BLIP2 Visual Question Answering."""
        print(f"\n{'='*80}")
        print("Testing BLIP2 (Visual Question Answering)")
        print(f"{'='*80}")
        print(f"Question: {question}")

        try:
            answer = self.run_with_timeout(self.blip2.ask, image, prompt=question, timeout=5)
            if answer is None:
                return None
            print(f"✓ Answer: {answer}")

            # Save result with text
            annotated = image.copy()
            # 在图片最上方绘制问题
            annotated = self.draw_text_top(annotated, f"Q: {question}", color=(255,255,255))
            # Add text overlay
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(annotated, f"A: {answer}", (10, 60), font, 0.6, (0, 255, 0), 2)

            output_path = self.output_dir / "blip2_result.jpg"
            cv2.imwrite(str(output_path), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
            print(f"✓ Saved result to: {output_path}")

            return answer

        except Exception as e:
            print(f"❌ Error: {e}")
            return None

    def test_blip2itm(self, image: np.ndarray, text: str):
        """Test BLIP2-ITM Image-Text Matching."""
        print(f"\n{'='*80}")
        print("Testing BLIP2-ITM (Image-Text Matching)")
        print(f"{'='*80}")
        print(f"Text: {text}")

        try:
            score = self.run_with_timeout(self.blip2itm.cosine, image, text, timeout=5)
            if score is None:
                return None
            print(f"✓ Matching score: {score:.4f} ({score*100:.2f}%)")

            # Save result with text
            annotated = image.copy()
            # 在图片最上方绘制文本
            annotated = self.draw_text_top(annotated, f"Text: {text}", color=(255,255,255))
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(annotated, f"Score: {score:.4f}", (10, 60), font, 0.6, (0, 255, 0), 2)

            output_path = self.output_dir / "blip2itm_result.jpg"
            cv2.imwrite(str(output_path), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
            print(f"✓ Saved result to: {output_path}")

            return score

        except Exception as e:
            print(f"❌ Error: {e}")
            return None

    def interactive_loop(self):
        """Main interactive loop."""
        print(f"\n{'='*80}")
        print("VLM Models Interactive Tester")
        print(f"{'='*80}\n")

        while True:
            print("\n" + "="*80)
            print("Select a model to test:")
            print("  [1] Grounding DINO - Open-vocabulary object detection")
            print("  [2] YOLOv7 - COCO object detection")
            print("  [3] Mobile SAM - Segmentation (requires bbox)")
            print("  [4] BLIP2 - Visual Question Answering")
            print("  [5] BLIP2-ITM - Image-Text Matching")
            print("  [6] Run all models (auto demo)")
            print("  [q] Quit")
            print("="*80)

            choice = input("\nYour choice: ").strip().lower()

            if choice == 'q':
                print("Goodbye!")
                break

            # Get image path
            image_path = input("Enter image path (or press Enter for default 'vlm_test_results/test.jpg'): ").strip()
            if not image_path:
                image_path = "vlm_test_results/test.jpg"

            image = self.load_image(image_path)
            if image is None:
                continue

            # Process based on choice
            if choice == '1':
                caption = input("Enter detection caption (e.g., 'chair . bottle . person'): ").strip()
                if not caption:
                    caption = gdino_caption
                    
                self.test_grounding_dino(image, caption)

            elif choice == '2':
                self.test_yolov7(image)

            elif choice == '3':
                bbox_str = input("Enter bbox as 'x1,y1,x2,y2' (or press Enter for image center): ").strip()
                if bbox_str:
                    bbox = [float(x) for x in bbox_str.split(',')]
                else:
                    # Default: center 50% of image
                    h, w = image.shape[:2]
                    bbox = [w*0.25, h*0.25, w*0.75, h*0.75]
                self.test_mobile_sam(image, bbox)

            elif choice == '4':
                question = input("Enter question (or press Enter for 'What is in this image?'): ").strip()
                if not question:
                    question = "What is in this image?"
                self.test_blip2(image, question)

            elif choice == '5':
                text = input("Enter text to match (e.g., 'a photo of a chair'): ").strip()
                if not text:
                    text = blip2itm_text
                    
                self.test_blip2itm(image, text)

            elif choice == '6':
                print("\n🚀 Running all models...")
                
                # 1. Grounding DINO
                detections_dino = self.test_grounding_dino(image, gdino_caption)
                
                # 2. YOLOv7
                detections_yolo = self.test_yolov7(image)
                
                # 3. Mobile SAM (use first detection from DINO if available)
                if detections_dino and detections_dino.num_detections > 0:
                    h, w = image.shape[:2]
                    bbox = detections_dino.boxes[0] * np.array([w, h, w, h])
                    self.test_mobile_sam(image, bbox.tolist())
                else:
                    h, w = image.shape[:2]
                    bbox = [w*0.25, h*0.25, w*0.75, h*0.75]
                    self.test_mobile_sam(image, bbox)
                
                # 4. BLIP2 VQA
                self.test_blip2(image, "What is in this image?")
                
                # 5. BLIP2-ITM
                self.test_blip2itm(image, blip2itm_text)
                
                print(f"\n✓ All tests completed! Check results in {self.output_dir}")
                break
            else:
                print("❌ Invalid choice. Please try again.")


gdino_caption = "chair . bottle . person . window . cup . fish . door . "
blip2itm_text = "a photo with a woman"

def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description="Interactive VLM Models Tester")
    parser.add_argument(
        "--config",
        type=str,
        default="behavior.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Test image path (optional, for quick test)"
    )

    args = parser.parse_args()

    # Initialize tester
    tester = VLMTester(config_path=args.config)


    # Quick test mode
    if args.image:
        image = tester.load_image(args.image)
        if image is not None:
            print("\n🚀 Running quick test with all models...")
            tester.test_grounding_dino(image, "chair . bottle . person")
            tester.test_yolov7(image)
            h, w = image.shape[:2]
            tester.test_mobile_sam(image, [w*0.25, h*0.25, w*0.75, h*0.75])
            tester.test_blip2(image, "What is in this image?")
            tester.test_blip2itm(image, "a photo with objects")
            print(f"\n✓ Quick test completed! Check results in {tester.output_dir}")
    else:
        # Interactive mode
        tester.interactive_loop()


if __name__ == "__main__":
    main()
