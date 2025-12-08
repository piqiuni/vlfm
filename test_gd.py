#!/usr/bin/env python3
"""
GroundingDINO Client-Server Demo

This script demonstrates how to use GroundingDINO for object detection through
a client-server architecture.

Usage:
    # Step 1: Start the server (in one terminal)
    python -m vlfm.vlm.grounding_dino --port 12181
    
    # Step 2: Run this demo (in another terminal)
    python test_grounding_dino.py --image path/to/image.jpg
    
    # Or use webcam
    python test_grounding_dino.py --webcam
    
    # Custom detection classes
    python test_grounding_dino.py --image img.jpg --classes "chair . table . bottle ."
"""

import argparse
import cv2
import numpy as np
import time
from pathlib import Path

from vlfm.vlm.grounding_dino import GroundingDINOClient
from vlfm.vlm.detections import ObjectDetections

def draw_detections(image: np.ndarray, detections, title: str = "Detections") -> np.ndarray:
    """
    Draw bounding boxes and labels on the image.
    
    Args:
        image: Input image (H, W, 3) RGB
        detections: ObjectDetections instance
        title: Window title
        
    Returns:
        Image with drawn boxes
    """
    vis_image = image.copy()
    h, w = image.shape[:2]
    
    # Convert RGB to BGR for OpenCV
    vis_image = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
    
    # Draw each detection
    for i in range(len((detections.phrases))):
        box = detections.boxes[i]  # [x_center, y_center, width, height] normalized
        logit = detections.logits[i]
        phrase = detections.phrases[i]
        
        # Convert from normalized [cx, cy, w, h] to pixel [x1, y1, x2, y2]
        cx, cy, bw, bh = box
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        
        # Draw rectangle
        color = (0, 255, 0)  # Green
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
        
        # Draw label with confidence
        label = f"{phrase}: {logit:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_y = max(y1 - 10, label_size[1])
        
        # Background for text
        cv2.rectangle(
            vis_image,
            (x1, label_y - label_size[1] - 5),
            (x1 + label_size[0], label_y + 5),
            color,
            -1
        )
        
        # Draw text
        cv2.putText(
            vis_image,
            label,
            (x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1
        )
    
    # Add detection count
    info_text = f"Detected {len((detections.phrases))} objects"
    cv2.putText(
        vis_image,
        info_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2
    )
    
    return vis_image


def test_with_image(client: GroundingDINOClient, image_path: str, classes: str):
    """Test with a single image file."""
    print(f"Loading image: {image_path}")
    
    # Load image
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    print(f"Image shape: {image_rgb.shape}")
    print(f"Detection classes: {classes}")
    print("Sending request to GroundingDINO server...")
    
    # Run detection
    start_time = time.time()
    detections: ObjectDetections = client.predict(image_rgb, caption=classes)
    elapsed = time.time() - start_time
    
    print(f"Detection completed in {elapsed:.2f}s")
    # print(f"Found {len(detections)} objects:")
    
    # Print detection details
    for i in range(len(detections.phrases)):
        print(f"  [{i+1}] {detections.phrases[i]}: {detections.logits[i]:.3f}")
    
    # Visualize
    vis_image = draw_detections(image_rgb, detections)
    
    # Show result
    # cv2.imshow("GroundingDINO Detections", vis_image)
    print("\nPress any key to close the window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Save result
    output_path = Path(image_path).stem + "_detections.jpg"
    cv2.imwrite(output_path, vis_image)
    print(f"Result saved to: {output_path}")


def test_with_webcam(client: GroundingDINOClient, classes: str):
    """Test with webcam in real-time."""
    print("Opening webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print(f"Detection classes: {classes}")
    print("Press 'q' to quit, 's' to save current frame")
    
    frame_count = 0
    fps_time = time.time()
    
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        # Run detection every N frames to maintain FPS
        if frame_count % 5 == 0:  # Process every 5th frame
            try:
                detections = client.predict(frame_rgb, caption=classes)
                vis_frame = draw_detections(frame_rgb, detections)
            except Exception as e:
                print(f"Detection error: {e}")
                vis_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        
        # Calculate FPS
        frame_count += 1
        if frame_count % 30 == 0:
            current_time = time.time()
            fps = 30 / (current_time - fps_time)
            fps_time = current_time
            print(f"FPS: {fps:.1f}")
        
        # Show frame
        cv2.imshow("GroundingDINO Webcam", vis_frame)
        
        # Handle keyboard
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            output_path = f"webcam_detection_{int(time.time())}.jpg"
            cv2.imwrite(output_path, vis_frame)
            print(f"Saved frame to: {output_path}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam closed")


def main():
    parser = argparse.ArgumentParser(
        description="Test GroundingDINO object detection with client-server architecture"
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Path to input image file"
    )
    parser.add_argument(
        "--webcam",
        action="store_true",
        help="Use webcam for real-time detection"
    )
    parser.add_argument(
        "--classes",
        type=str,
        default="table . person",
        # default="chair . table . person . bottle . cup . laptop . phone . book .",
        help="Detection classes separated by ' . ' (default: common objects)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=12181,
        help="GroundingDINO server port (default: 12181)"
    )
    
    args = parser.parse_args()
    
    # Check arguments
    if not args.image and not args.webcam:
        parser.error("Must provide either --image or --webcam")
    
    # Initialize client
    print("=" * 60)
    print("GroundingDINO Client-Server Demo")
    print("=" * 60)
    print(f"Connecting to server at http://10.106.11.248:{args.port}/gdino")
    
    client = GroundingDINOClient(port=args.port)
    
    try:
        # Test connection with a dummy request
        print("Testing connection...")
        test_image = np.zeros((224, 224, 3), dtype=np.uint8)
        _ = client.predict(test_image, caption="test .")
        print("✓ Connection successful!")
        print("=" * 60)
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        print("\nMake sure the server is running:")
        print(f"  python -m vlfm.vlm.grounding_dino --port {args.port}")
        return
    
    # Run demo
    if args.image:
        test_with_image(client, args.image, args.classes)
    elif args.webcam:
        test_with_webcam(client, args.classes)
    
    print("\nDemo completed!")


if __name__ == "__main__":
    main()
