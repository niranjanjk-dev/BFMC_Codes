"""
parking_slot_detector.py — Bird's-Eye-View Corner-Based Parking Slot Detector
==============================================================================

Detects parking slots by finding corners (using Shi-Tomasi / Harris) on the
BEV (warped) image and grouping them into rectangles.

Algorithm:
  1. Convert BEV frame to grayscale and detect edges
  2. Detect corners using Shi-Tomasi (goodFeaturesToTrack)
  3. Cluster nearby corners to eliminate duplicates
  4. Find groups of 4 corners that form valid rectangles
  5. Validate rectangle aspect ratio and size against parking slot dimensions
  6. Track confirmed slots across frames with EMA smoothing
  7. Check occupancy against YOLO detections

A valid parking slot is a rectangle where:
  - Width:  60–250 px (in BEV space)
  - Height: 100–400 px (in BEV space)
  - Aspect ratio (height/width): 1.2–5.0 (taller than wide)
  - Corners form approximately right angles (within tolerance)
"""

import cv2
import numpy as np
from itertools import combinations


class SlotDetector:
    def __init__(self):
        # ── Tracking state ───────────────────────────────────────────
        self.tracked_slots = []            # List of tracked slot rectangles
        self.ALPHA = 0.80                  # EMA smoothing for slot tracking
        self.MATCH_DIST_THRESHOLD = 60     # Max distance to match a new slot to tracked

        # ── Corner detection parameters ──────────────────────────────
        self.MAX_CORNERS = 80
        self.QUALITY_LEVEL = 0.08
        self.MIN_CORNER_DIST = 15
        self.BLOCK_SIZE = 5

        # ── Slot geometry constraints (BEV pixel space) ──────────────
        self.MIN_SLOT_WIDTH = 60
        self.MAX_SLOT_WIDTH = 250
        self.MIN_SLOT_HEIGHT = 100
        self.MAX_SLOT_HEIGHT = 400
        self.MIN_ASPECT_RATIO = 1.2        # height / width
        self.MAX_ASPECT_RATIO = 5.0
        self.CORNER_ANGLE_TOL = 25.0       # degrees tolerance from 90°

        # ── Cluster merge distance ───────────────────────────────────
        self.CLUSTER_DIST = 20             # px — merge corners closer than this

        # ── BEV perspective transform (same as lane detector) ────────
        self.SRC_PTS = np.float32([[200, 260], [440, 260], [40, 450], [600, 450]])
        self.DST_PTS = np.float32([[150, 0], [490, 0], [150, 480], [490, 480]])
        self.M_forward = cv2.getPerspectiveTransform(self.SRC_PTS, self.DST_PTS)
        self.M_inv = cv2.getPerspectiveTransform(self.DST_PTS, self.SRC_PTS)

    def detect_slot(self, frame, dets):
        """
        Detect parking slots using Bird's-Eye-View corner detection.

        Parameters
        ----------
        frame : np.ndarray  — Raw camera frame (BGR, 480x640)
        dets  : list[dict]  — YOLO detections with 'label' and 'bbox' keys

        Returns
        -------
        (slot_found: bool, annotated_frame: np.ndarray)
        """
        if frame is None:
            return False, frame

        h, w = frame.shape[:2]

        # ── Step 1: Create BEV ───────────────────────────────────────
        if frame.shape[:2] != (480, 640):
            process_frame = cv2.resize(frame, (640, 480))
        else:
            process_frame = frame

        bev = cv2.warpPerspective(process_frame, self.M_forward, (640, 480))
        bev_gray = cv2.cvtColor(bev, cv2.COLOR_BGR2GRAY)

        # ── Step 2: Enhance edges for corner detection ───────────────
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(bev_gray)

        # Light blur to reduce noise while preserving edges
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)

        # Edge detection for visualization
        edges = cv2.Canny(blurred, 50, 150)

        # ── Step 3: Detect corners (Shi-Tomasi) ──────────────────────
        corners = cv2.goodFeaturesToTrack(
            blurred,
            maxCorners=self.MAX_CORNERS,
            qualityLevel=self.QUALITY_LEVEL,
            minDistance=self.MIN_CORNER_DIST,
            blockSize=self.BLOCK_SIZE
        )

        if corners is None or len(corners) < 4:
            # Draw edges on BEV for debug
            self._draw_bev_overlay(frame, bev, edges, [], [], "NO CORNERS")
            return False, frame

        # Flatten corners to (N, 2)
        pts = corners.reshape(-1, 2)

        # ── Step 4: Cluster nearby corners ───────────────────────────
        clustered = self._cluster_corners(pts)

        if len(clustered) < 4:
            self._draw_bev_overlay(frame, bev, edges, clustered, [], "< 4 CLUSTERS")
            return False, frame

        # ── Step 5: Find rectangles from corner groups ───────────────
        rectangles = self._find_rectangles(clustered)

        # ── Step 6: Validate and filter rectangles ───────────────────
        valid_slots = []
        for rect in rectangles:
            if self._validate_slot_geometry(rect):
                # Check occupancy via YOLO
                occupied = self._check_occupancy(rect, dets, h, w)
                valid_slots.append((rect, occupied))

        # ── Step 7: Track slots and draw results ────────────────────
        slot_found = False
        free_slots = []

        for rect, occupied in valid_slots:
            if not occupied:
                slot_found = True
                free_slots.append(rect)

        # Update tracked slots with EMA
        self._update_tracking(free_slots)

        # ── Step 8: Draw visualization on frame ─────────────────────
        self._draw_bev_overlay(frame, bev, edges, clustered, valid_slots, 
                                "SLOT FOUND" if slot_found else "SCANNING")

        return slot_found, frame

    def _cluster_corners(self, pts):
        """Merge corners that are very close together into single representative points."""
        if len(pts) == 0:
            return np.array([])

        used = [False] * len(pts)
        clusters = []

        for i in range(len(pts)):
            if used[i]:
                continue
            cluster = [pts[i]]
            used[i] = True
            for j in range(i + 1, len(pts)):
                if not used[j] and np.linalg.norm(pts[i] - pts[j]) < self.CLUSTER_DIST:
                    cluster.append(pts[j])
                    used[j] = True
            clusters.append(np.mean(cluster, axis=0))

        return np.array(clusters)

    def _find_rectangles(self, pts):
        """
        Find groups of 4 corners that form valid rectangles.
        
        Uses the approach: for each combination of 4 points, check if they
        form a valid rectangle by verifying:
          1. Order the 4 points into a proper rectangle (TL, TR, BR, BL)
          2. All 4 angles are approximately 90 degrees
          3. Opposite sides are approximately equal length
        """
        rectangles = []

        # Limit combinations to avoid explosive computation with many corners
        max_pts = min(len(pts), 25)
        search_pts = pts[:max_pts]

        for combo in combinations(range(len(search_pts)), 4):
            quad = search_pts[list(combo)]
            ordered = self._order_points(quad)
            if ordered is None:
                continue

            if self._is_valid_rectangle(ordered):
                rectangles.append(ordered)

        # Remove duplicate/overlapping rectangles
        rectangles = self._remove_duplicates(rectangles)

        return rectangles

    def _order_points(self, pts):
        """
        Order 4 points into [top-left, top-right, bottom-right, bottom-left].
        Returns None if ordering fails.
        """
        if len(pts) != 4:
            return None

        # Sort by y-coordinate (top to bottom)
        sorted_by_y = pts[np.argsort(pts[:, 1])]

        # Top two points
        top_two = sorted_by_y[:2]
        top_two = top_two[np.argsort(top_two[:, 0])]  # Sort left to right

        # Bottom two points
        bot_two = sorted_by_y[2:]
        bot_two = bot_two[np.argsort(bot_two[:, 0])]  # Sort left to right

        # [TL, TR, BR, BL]
        return np.array([top_two[0], top_two[1], bot_two[1], bot_two[0]], dtype=np.float32)

    def _is_valid_rectangle(self, ordered):
        """
        Check if 4 ordered points form a valid rectangle.
        Verifies angles are close to 90° and opposite sides are similar length.
        """
        tl, tr, br, bl = ordered

        # Compute side vectors
        sides = [
            tr - tl,  # top
            br - tr,  # right
            bl - br,  # bottom
            tl - bl,  # left
        ]

        # Check all 4 angles are ~90°
        for i in range(4):
            v1 = sides[i]
            v2 = sides[(i + 1) % 4]

            len1 = np.linalg.norm(v1)
            len2 = np.linalg.norm(v2)
            if len1 < 10 or len2 < 10:  # Degenerate side
                return False

            cos_angle = np.dot(v1, v2) / (len1 * len2 + 1e-8)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.degrees(np.arccos(abs(cos_angle)))

            # Angle should be close to 90° (cos ≈ 0)
            if abs(angle - 90.0) > self.CORNER_ANGLE_TOL:
                return False

        # Check opposite sides are similar length
        top_len = np.linalg.norm(sides[0])
        bot_len = np.linalg.norm(sides[2])
        left_len = np.linalg.norm(sides[3])
        right_len = np.linalg.norm(sides[1])

        if top_len > 0 and bot_len > 0:
            ratio_tb = max(top_len, bot_len) / min(top_len, bot_len)
            if ratio_tb > 1.8:
                return False

        if left_len > 0 and right_len > 0:
            ratio_lr = max(left_len, right_len) / min(left_len, right_len)
            if ratio_lr > 1.8:
                return False

        return True

    def _validate_slot_geometry(self, rect):
        """
        Validate that the rectangle dimensions match a parking slot.
        rect: [TL, TR, BR, BL] as float32 array
        """
        tl, tr, br, bl = rect

        width = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2.0
        height = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2.0

        if width < self.MIN_SLOT_WIDTH or width > self.MAX_SLOT_WIDTH:
            return False
        if height < self.MIN_SLOT_HEIGHT or height > self.MAX_SLOT_HEIGHT:
            return False

        aspect = height / (width + 1e-8)
        if aspect < self.MIN_ASPECT_RATIO or aspect > self.MAX_ASPECT_RATIO:
            return False

        return True

    def _check_occupancy(self, rect, dets, frame_h, frame_w):
        """
        Check if a detected slot is occupied using YOLO detections.
        Transforms the slot rectangle back to camera space for comparison.
        """
        if not dets:
            return False

        # Get bounding box of the slot in BEV space
        tl, tr, br, bl = rect
        slot_cx = np.mean([tl[0], tr[0], br[0], bl[0]])
        slot_cy = np.mean([tl[1], tr[1], br[1], bl[1]])

        # Transform slot center back to camera space
        slot_pt_bev = np.array([[[slot_cx, slot_cy]]], dtype=np.float32)
        slot_pt_cam = cv2.perspectiveTransform(slot_pt_bev, self.M_inv)
        cam_x, cam_y = slot_pt_cam[0][0]

        for d in dets:
            label = d.get("label", "").lower()
            if label in ("car", "obstacle", "roadblock", "closed-road-stand"):
                bbox = d.get("bbox", [0, 0, 0, 0])
                bx1, by1, bx2, by2 = bbox
                # Check if slot center falls within detection bbox (with margin)
                margin = 30
                if (bx1 - margin < cam_x < bx2 + margin and
                    by1 - margin < cam_y < by2 + margin):
                    return True

        return False

    def _remove_duplicates(self, rectangles):
        """Remove overlapping rectangles by comparing their centers."""
        if len(rectangles) <= 1:
            return rectangles

        unique = []
        for rect in rectangles:
            center = np.mean(rect, axis=0)
            is_dup = False
            for existing in unique:
                ex_center = np.mean(existing, axis=0)
                if np.linalg.norm(center - ex_center) < 40:
                    is_dup = True
                    break
            if not is_dup:
                unique.append(rect)

        return unique

    def _update_tracking(self, free_slots):
        """EMA-based tracking: smooth slot positions across frames."""
        new_tracked = []

        for slot in free_slots:
            center = np.mean(slot, axis=0)
            matched = False

            for i, tracked in enumerate(self.tracked_slots):
                tracked_center = np.mean(tracked, axis=0)
                if np.linalg.norm(center - tracked_center) < self.MATCH_DIST_THRESHOLD:
                    # EMA update
                    updated = self.ALPHA * tracked + (1.0 - self.ALPHA) * slot
                    new_tracked.append(updated)
                    matched = True
                    break

            if not matched:
                new_tracked.append(slot.copy())

        self.tracked_slots = new_tracked

    def _draw_bev_overlay(self, frame, bev, edges, corners, valid_slots, status_text):
        """
        Draw slot detection visualization on the BEV portion of the frame.
        Shows: corners as red crosshairs, valid slots as colored rectangles,
        and overlays the BEV debug view on the camera video.
        """
        h, w = frame.shape[:2]
        bev_h, bev_w = bev.shape[:2]

        # Create a BEV debug visualization
        bev_dbg = bev.copy()

        # Draw edges as green overlay
        edge_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        edge_color[edges > 0] = [0, 180, 0]
        bev_dbg = cv2.addWeighted(bev_dbg, 0.7, edge_color, 0.3, 0)

        # Draw all detected corners as red crosshairs (+)
        if len(corners) > 0:
            for pt in corners:
                cx, cy = int(pt[0]), int(pt[1])
                size = 8
                cv2.line(bev_dbg, (cx - size, cy), (cx + size, cy), (0, 0, 255), 2)
                cv2.line(bev_dbg, (cx, cy - size), (cx, cy + size), (0, 0, 255), 2)

        # Draw valid slot rectangles
        for i, (rect, occupied) in enumerate(valid_slots if valid_slots else []):
            pts = rect.astype(np.int32)

            if occupied:
                color = (0, 0, 255)      # Red — occupied
                label = "OCCUPIED"
            else:
                color = (0, 255, 255)    # Yellow — free slot
                label = f"SLOT {i+1}"

            # Draw the 4 edges of rectangle
            for j in range(4):
                p1 = tuple(pts[j])
                p2 = tuple(pts[(j + 1) % 4])
                cv2.line(bev_dbg, p1, p2, color, 2)

            # Draw corner markers (red crosshairs like in user's diagram)
            for pt in pts:
                cx, cy = pt
                size = 10
                cv2.line(bev_dbg, (cx - size, cy), (cx + size, cy), (0, 0, 255), 3)
                cv2.line(bev_dbg, (cx, cy - size), (cx, cy + size), (0, 0, 255), 3)

            # Label
            cv2.putText(bev_dbg, label, (pts[0][0], pts[0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Draw tracked slots (green — confirmed)
        for tracked in self.tracked_slots:
            pts = tracked.astype(np.int32)
            for j in range(4):
                p1 = tuple(pts[j])
                p2 = tuple(pts[(j + 1) % 4])
                cv2.line(bev_dbg, p1, p2, (0, 255, 0), 3)

        # Status text on BEV
        cv2.putText(bev_dbg, f"BEV SLOT: {status_text}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(bev_dbg, f"Corners: {len(corners)}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # ── Overlay BEV mini-view on the camera frame ────────────────
        # Place a scaled-down BEV in the top-right corner of the camera frame
        mini_w, mini_h = 200, 150
        bev_mini = cv2.resize(bev_dbg, (mini_w, mini_h))

        # Position: top-right with margin
        x_off = w - mini_w - 10
        y_off = 10

        # Border
        cv2.rectangle(frame, (x_off - 2, y_off - 2),
                      (x_off + mini_w + 2, y_off + mini_h + 2),
                      (0, 255, 255), 2)

        # Only overlay if frame is large enough
        if x_off > 0 and y_off + mini_h < h:
            frame[y_off:y_off + mini_h, x_off:x_off + mini_w] = bev_mini

        # Also project detected slots back onto the original camera frame
        for tracked in self.tracked_slots:
            cam_pts = cv2.perspectiveTransform(
                tracked.reshape(1, -1, 2).astype(np.float32), self.M_inv
            ).reshape(-1, 2).astype(np.int32)

            for j in range(4):
                p1 = tuple(cam_pts[j])
                p2 = tuple(cam_pts[(j + 1) % 4])
                cv2.line(frame, p1, p2, (0, 255, 0), 2)

            # Corner markers on camera view too
            for pt in cam_pts:
                cx, cy = pt
                size = 6
                cv2.line(frame, (cx - size, cy), (cx + size, cy), (0, 0, 255), 2)
                cv2.line(frame, (cx, cy - size), (cx, cy + size), (0, 0, 255), 2)
