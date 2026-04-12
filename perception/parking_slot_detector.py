"""
parking_slot_detector.py — Bird's-Eye-View Corner-Based Parking Slot Detector
==============================================================================

Detects parking slots by finding corners (using Shi-Tomasi / Harris) on the
BEV (warped) image and grouping them into rectangles.

Algorithm:
  1. Convert BEV frame to grayscale and detect edges
  2. Detect Hough Lines on BEV edges for full line visualization
  3. Detect corners using Shi-Tomasi (goodFeaturesToTrack)
  4. Cluster nearby corners to eliminate duplicates
  5. Find groups of 4 corners that form valid rectangles
  6. Validate rectangle aspect ratio and size against parking slot dimensions
  7. Track confirmed slots across frames with EMA smoothing
  8. Check occupancy against YOLO detections
  9. Determine slot side (LEFT / RIGHT) relative to car's forward direction

A valid parking slot is a rectangle where:
  - Width:  60–250 px (in BEV space)
  - Height: 100–400 px (in BEV space)
  - Aspect ratio (height/width): 1.2–5.0 (taller than wide)
  - Corners form approximately right angles (within tolerance)

Visualization draws:
  - All Hough lines (cyan)
  - All raw corners before clustering (small red dots)
  - Clustered corners (green crosshairs)
  - Candidate rectangles — invalid (dim grey), valid occupied (red), valid free (yellow)
  - Tracked / confirmed slots (bright green with thick borders)
  - Slot side label (LEFT / RIGHT)
  - Text annotations: corner count, line count, rectangle/slot counts
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

        # ── Hough Line parameters ────────────────────────────────────
        self.HOUGH_RHO = 1
        self.HOUGH_THETA = np.pi / 180
        self.HOUGH_THRESHOLD = 40
        self.HOUGH_MIN_LINE_LEN = 30
        self.HOUGH_MAX_LINE_GAP = 15

        # ── BEV perspective transform (same as lane detector) ────────
        self.SRC_PTS = np.float32([[200, 260], [440, 260], [40, 450], [600, 450]])
        self.DST_PTS = np.float32([[150, 0], [490, 0], [150, 480], [490, 480]])
        self.M_forward = cv2.getPerspectiveTransform(self.SRC_PTS, self.DST_PTS)
        self.M_inv = cv2.getPerspectiveTransform(self.DST_PTS, self.SRC_PTS)

        # ── Last detected slot side ──────────────────────────────────
        self._last_slot_side = "NONE"

    # ══════════════════════════════════════════════════════════════════
    #  PUBLIC API
    # ══════════════════════════════════════════════════════════════════

    def detect_slot(self, frame, dets):
        """
        Detect parking slots using Bird's-Eye-View corner detection.

        Parameters
        ----------
        frame : np.ndarray  — Raw camera frame (BGR, 480x640)
        dets  : list[dict]  — YOLO detections with 'label' and 'bbox' keys

        Returns
        -------
        (slot_found: bool, annotated_frame: np.ndarray, slot_side: str)
            slot_side is "LEFT", "RIGHT", or "NONE"
        """
        if frame is None:
            return False, frame, "NONE"

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

        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)

        # ── Step 3: Hough Line detection (for visualization) ─────────
        hough_lines = cv2.HoughLinesP(
            edges,
            rho=self.HOUGH_RHO,
            theta=self.HOUGH_THETA,
            threshold=self.HOUGH_THRESHOLD,
            minLineLength=self.HOUGH_MIN_LINE_LEN,
            maxLineGap=self.HOUGH_MAX_LINE_GAP
        )

        # ── Step 4: Detect corners (Shi-Tomasi) ──────────────────────
        corners = cv2.goodFeaturesToTrack(
            blurred,
            maxCorners=self.MAX_CORNERS,
            qualityLevel=self.QUALITY_LEVEL,
            minDistance=self.MIN_CORNER_DIST,
            blockSize=self.BLOCK_SIZE
        )

        if corners is None or len(corners) < 4:
            self._draw_bev_overlay(frame, bev, edges, hough_lines,
                                   None, [], [], [], "NO CORNERS")
            return False, frame, "NONE"

        # Raw corners (before clustering) for visualization
        raw_pts = corners.reshape(-1, 2)

        # ── Step 5: Cluster nearby corners ───────────────────────────
        clustered = self._cluster_corners(raw_pts)

        if len(clustered) < 4:
            self._draw_bev_overlay(frame, bev, edges, hough_lines,
                                   raw_pts, clustered, [], [], "< 4 CLUSTERS")
            return False, frame, "NONE"

        # ── Step 6: Find rectangles from corner groups ───────────────
        rectangles, invalid_rects = self._find_rectangles_with_rejects(clustered)

        # ── Step 7: Validate and filter rectangles ───────────────────
        valid_slots = []
        for rect in rectangles:
            if self._validate_slot_geometry(rect):
                occupied = self._check_occupancy(rect, dets, h, w)
                valid_slots.append((rect, occupied))

        # ── Step 8: Track slots and determine side ───────────────────
        slot_found = False
        free_slots = []
        slot_side = "NONE"

        for rect, occupied in valid_slots:
            if not occupied:
                slot_found = True
                free_slots.append(rect)

        # Update tracked slots with EMA
        self._update_tracking(free_slots)

        # Determine slot side from tracked slots
        if self.tracked_slots:
            slot_side = self._determine_slot_side(self.tracked_slots)
            self._last_slot_side = slot_side

        # ── Step 9: Draw full visualization on frame ─────────────────
        self._draw_bev_overlay(frame, bev, edges, hough_lines,
                               raw_pts, clustered, valid_slots,
                               invalid_rects,
                               "SLOT FOUND" if slot_found else "SCANNING",
                               slot_side=slot_side)

        return slot_found, frame, slot_side

    def get_last_slot_side(self):
        """Return the last detected slot side: 'LEFT', 'RIGHT', or 'NONE'."""
        return self._last_slot_side

    # ══════════════════════════════════════════════════════════════════
    #  CORNER CLUSTERING
    # ══════════════════════════════════════════════════════════════════

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

    # ══════════════════════════════════════════════════════════════════
    #  RECTANGLE DETECTION
    # ══════════════════════════════════════════════════════════════════

    def _find_rectangles_with_rejects(self, pts):
        """
        Find groups of 4 corners that form valid rectangles.
        Also returns invalid rectangles (for visualization of rejected candidates).

        Returns
        -------
        (valid_rectangles, invalid_rectangles)
        """
        valid_rectangles = []
        invalid_rectangles = []

        max_pts = min(len(pts), 25)
        search_pts = pts[:max_pts]

        for combo in combinations(range(len(search_pts)), 4):
            quad = search_pts[list(combo)]
            ordered = self._order_points(quad)
            if ordered is None:
                continue

            if self._is_valid_rectangle(ordered):
                valid_rectangles.append(ordered)
            else:
                invalid_rectangles.append(ordered)

        # Remove duplicate/overlapping rectangles
        valid_rectangles = self._remove_duplicates(valid_rectangles)

        # Limit invalid rectangles for drawing (don't flood the display)
        if len(invalid_rectangles) > 10:
            invalid_rectangles = invalid_rectangles[:10]

        return valid_rectangles, invalid_rectangles

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

    # ══════════════════════════════════════════════════════════════════
    #  OCCUPANCY CHECK
    # ══════════════════════════════════════════════════════════════════

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

    # ══════════════════════════════════════════════════════════════════
    #  DUPLICATE REMOVAL & TRACKING
    # ══════════════════════════════════════════════════════════════════

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

    # ══════════════════════════════════════════════════════════════════
    #  SLOT SIDE DETERMINATION
    # ══════════════════════════════════════════════════════════════════

    def _determine_slot_side(self, tracked_slots):
        """
        Determine whether the detected slot is to the LEFT or RIGHT
        of the car's forward direction.

        In BEV space the car is driving "up" (from bottom to top).
        The BEV frame center x-axis is at x = 320 (for 640-wide BEV).

        - If the slot center is to the LEFT of the BEV center → LEFT slot
        - If the slot center is to the RIGHT of the BEV center → RIGHT slot

        Returns 'LEFT', 'RIGHT', or 'NONE'.
        """
        if not tracked_slots or len(tracked_slots) == 0:
            return "NONE"

        # Use the first (most confident) tracked slot
        slot = tracked_slots[0]
        slot_cx = np.mean(slot[:, 0])  # Average x of 4 corners

        bev_center_x = 320.0  # Half of 640-wide BEV

        if slot_cx < bev_center_x:
            return "LEFT"
        else:
            return "RIGHT"

    # ══════════════════════════════════════════════════════════════════
    #  FULL VISUALIZATION
    # ══════════════════════════════════════════════════════════════════

    def _draw_bev_overlay(self, frame, bev, edges, hough_lines,
                          raw_corners, clustered_corners,
                          valid_slots, invalid_rects,
                          status_text, slot_side="NONE"):
        """
        Draw comprehensive slot detection visualization.

        Shows on BEV debug view:
          - Canny edges (green tint)
          - ALL Hough lines (cyan, thin)
          - Raw corners before clustering (small red dots)
          - Clustered corners (green crosshairs)
          - Invalid candidate rectangles (dim grey dashed)
          - Valid slots: yellow (free) / red (occupied) borders
          - Tracked confirmed slots (bright green, thick)
          - Slot side label (LEFT / RIGHT)
          - Text annotations with counts

        Also projects confirmed slots onto the original camera frame.
        """
        h, w = frame.shape[:2]
        bev_h, bev_w = bev.shape[:2]

        # Create BEV debug visualization
        bev_dbg = bev.copy()

        # ── 1. Draw Canny edges as green overlay ─────────────────────
        edge_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        edge_color[edges > 0] = [0, 180, 0]
        bev_dbg = cv2.addWeighted(bev_dbg, 0.65, edge_color, 0.35, 0)

        # ── 2. Draw ALL Hough lines (cyan, thin) ─────────────────────
        num_hough = 0
        if hough_lines is not None:
            num_hough = len(hough_lines)
            for line in hough_lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(bev_dbg, (x1, y1), (x2, y2), (255, 255, 0), 1)  # Cyan

        # ── 3. Draw raw corners (small red filled circles) ───────────
        num_raw = 0
        if raw_corners is not None and len(raw_corners) > 0:
            num_raw = len(raw_corners)
            for pt in raw_corners:
                cx, cy = int(pt[0]), int(pt[1])
                cv2.circle(bev_dbg, (cx, cy), 4, (0, 0, 255), -1)  # Red filled

        # ── 4. Draw clustered corners (green crosshairs) ─────────────
        num_clustered = 0
        if clustered_corners is not None and len(clustered_corners) > 0:
            num_clustered = len(clustered_corners)
            for pt in clustered_corners:
                cx, cy = int(pt[0]), int(pt[1])
                size = 10
                cv2.line(bev_dbg, (cx - size, cy), (cx + size, cy), (0, 255, 0), 2)
                cv2.line(bev_dbg, (cx, cy - size), (cx, cy + size), (0, 255, 0), 2)
                # Small label with index
                cv2.circle(bev_dbg, (cx, cy), 3, (0, 255, 0), -1)

        # ── 5. Draw INVALID candidate rectangles (dim grey) ──────────
        num_invalid = 0
        if invalid_rects and len(invalid_rects) > 0:
            num_invalid = len(invalid_rects)
            for rect in invalid_rects:
                pts = rect.astype(np.int32)
                for j in range(4):
                    p1 = tuple(pts[j])
                    p2 = tuple(pts[(j + 1) % 4])
                    cv2.line(bev_dbg, p1, p2, (100, 100, 100), 1)  # Dim grey

        # ── 6. Draw VALID slot rectangles ────────────────────────────
        num_valid = 0
        num_free = 0
        for i, (rect, occupied) in enumerate(valid_slots if valid_slots else []):
            num_valid += 1
            pts = rect.astype(np.int32)

            if occupied:
                color = (0, 0, 255)      # Red — occupied
                label = "OCCUPIED"
            else:
                num_free += 1
                color = (0, 255, 255)    # Yellow — free slot
                label = f"SLOT {i+1}"

            # Draw the 4 edges of rectangle (thick)
            for j in range(4):
                p1 = tuple(pts[j])
                p2 = tuple(pts[(j + 1) % 4])
                cv2.line(bev_dbg, p1, p2, color, 2)

            # Draw corner markers (bright red crosshairs)
            for pt in pts:
                cx, cy = pt
                size = 12
                cv2.line(bev_dbg, (cx - size, cy), (cx + size, cy), (0, 0, 255), 3)
                cv2.line(bev_dbg, (cx, cy - size), (cx, cy + size), (0, 0, 255), 3)

            # Slot label
            cv2.putText(bev_dbg, label, (pts[0][0], pts[0][1] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Draw slot dimensions text
            tl, tr, br, bl = rect
            slot_w = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2.0
            slot_h = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2.0
            dim_text = f"{slot_w:.0f}x{slot_h:.0f}px"
            center = np.mean(pts, axis=0).astype(int)
            cv2.putText(bev_dbg, dim_text, (center[0] - 30, center[1] + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # ── 7. Draw tracked / confirmed slots (bright green, thick) ──
        for tracked in self.tracked_slots:
            pts = tracked.astype(np.int32)
            for j in range(4):
                p1 = tuple(pts[j])
                p2 = tuple(pts[(j + 1) % 4])
                cv2.line(bev_dbg, p1, p2, (0, 255, 0), 3)

            # CONFIRMED label
            center = np.mean(pts, axis=0).astype(int)
            cv2.putText(bev_dbg, "CONFIRMED", (center[0] - 40, center[1] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # ── 8. Draw slot side indicator ──────────────────────────────
        if slot_side != "NONE":
            side_color = (255, 200, 0) if slot_side == "LEFT" else (0, 200, 255)
            cv2.putText(bev_dbg, f"SIDE: {slot_side}", (bev_w - 160, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, side_color, 2)

            # Draw arrow indicating side
            arrow_y = 80
            if slot_side == "LEFT":
                cv2.arrowedLine(bev_dbg, (bev_w - 80, arrow_y),
                                (bev_w - 150, arrow_y), side_color, 3, tipLength=0.3)
            else:
                cv2.arrowedLine(bev_dbg, (bev_w - 150, arrow_y),
                                (bev_w - 80, arrow_y), side_color, 3, tipLength=0.3)

        # ── 9. Draw BEV center line (reference for LEFT/RIGHT) ───────
        cv2.line(bev_dbg, (bev_w // 2, 0), (bev_w // 2, bev_h),
                 (255, 255, 255), 1)  # White dashed center line
        cv2.putText(bev_dbg, "CENTER", (bev_w // 2 - 25, bev_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

        # ── 10. Text annotations (top-left panel) ────────────────────
        info_lines = [
            f"BEV SLOT: {status_text}",
            f"Hough Lines: {num_hough}",
            f"Raw Corners: {num_raw}",
            f"Clustered: {num_clustered}",
            f"Rects (invalid): {num_invalid}",
            f"Valid Slots: {num_valid} (Free: {num_free})",
            f"Tracked: {len(self.tracked_slots)}",
        ]

        # Draw semi-transparent background for text panel
        panel_h = 20 + len(info_lines) * 22
        overlay = bev_dbg.copy()
        cv2.rectangle(overlay, (0, 0), (220, panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, bev_dbg, 0.5, 0, bev_dbg)

        for idx, text in enumerate(info_lines):
            y_pos = 20 + idx * 22
            # First line is status — use yellow
            color = (0, 255, 255) if idx == 0 else (200, 200, 200)
            thickness = 2 if idx == 0 else 1
            font_scale = 0.55 if idx == 0 else 0.45
            cv2.putText(bev_dbg, text, (8, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

        # ── 11. Overlay BEV mini-view on the camera frame ────────────
        # Use a LARGER mini-view so all details are visible
        mini_w, mini_h = 280, 210
        bev_mini = cv2.resize(bev_dbg, (mini_w, mini_h))

        # Position: top-right with margin
        x_off = w - mini_w - 10
        y_off = 10

        # Border with label
        cv2.rectangle(frame, (x_off - 3, y_off - 18),
                      (x_off + mini_w + 3, y_off - 1),
                      (0, 120, 200), -1)
        cv2.putText(frame, "BEV SLOT DETECTOR", (x_off + 5, y_off - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.rectangle(frame, (x_off - 3, y_off - 3),
                      (x_off + mini_w + 3, y_off + mini_h + 3),
                      (0, 255, 255), 2)

        # Only overlay if frame is large enough
        if x_off > 0 and y_off + mini_h < h:
            frame[y_off:y_off + mini_h, x_off:x_off + mini_w] = bev_mini

        # ── 12. Project confirmed slots back onto camera frame ───────
        for tracked in self.tracked_slots:
            cam_pts = cv2.perspectiveTransform(
                tracked.reshape(1, -1, 2).astype(np.float32), self.M_inv
            ).reshape(-1, 2).astype(np.int32)

            # Draw slot border on camera view (bright green)
            for j in range(4):
                p1 = tuple(cam_pts[j])
                p2 = tuple(cam_pts[(j + 1) % 4])
                cv2.line(frame, p1, p2, (0, 255, 0), 2)

            # Corner markers on camera view
            for pt in cam_pts:
                cx, cy = pt
                size = 8
                cv2.line(frame, (cx - size, cy), (cx + size, cy), (0, 0, 255), 2)
                cv2.line(frame, (cx, cy - size), (cx, cy + size), (0, 0, 255), 2)

            # Side label on camera view
            if slot_side != "NONE":
                cam_center = np.mean(cam_pts, axis=0).astype(int)
                side_color = (255, 200, 0) if slot_side == "LEFT" else (0, 200, 255)
                cv2.putText(frame, f"{slot_side} SLOT",
                            (cam_center[0] - 35, cam_center[1] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, side_color, 2)
