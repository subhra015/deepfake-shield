import cv2
import numpy as np
from pathlib import Path

def create_test_video(output_path="test_video.mp4", duration=5, fps=30, size=(640, 480)):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, float(fps), size)
    if not out.isOpened():
        raise RuntimeError("Failed to open VideoWriter. Check codec support.")
    total_frames = int(duration * fps)
    for i in range(total_frames):
        frame = np.full((*size[::-1], 3), 128, dtype=np.uint8)
        center = (size[0] // 2, size[1] // 2)
        cv2.ellipse(frame, center, (180, 220), 0, 0, 360, (180, 160, 140), -1)
        cv2.circle(frame, (center[0] - 70, center[1] - 40), 25, (255, 255, 255), -1)
        cv2.circle(frame, (center[0] + 70, center[1] - 40), 25, (255, 255, 255), -1)
        cv2.circle(frame, (center[0] - 70, center[1] - 40), 10, (50, 50, 50), -1)
        cv2.circle(frame, (center[0] + 70, center[1] - 40), 10, (50, 50, 50), -1)
        smile_offset = int(10 * np.sin(i * 0.2))
        cv2.ellipse(frame, (center[0], center[1] + 80 + smile_offset), (80, 40), 0, 0, 180, (100, 50, 50), 3)
        cv2.ellipse(frame, (center[0], center[1] + 20), (20, 30), 0, 0, 360, (160, 140, 120), -1)
        cv2.putText(frame, f"Frame {i}/{total_frames}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "DeepFake Shield Test", (10, size[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        out.write(frame)
    out.release()
    print(f"✅ Created {output_path}: {total_frames} frames, {fps} FPS, {size}")
    cap = cv2.VideoCapture(output_path)
    actual_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    print(f"✅ Verified: {actual_frames} frames, {actual_fps:.2f} FPS")

if __name__ == "__main__":
    create_test_video()
