import cv2
import numpy as np

class SlotDetector:
    def __init__(self):
        self.tracked_left_slot = None
        self.tracked_right_slot = None
        self.ALPHA = 0.85
        self.CENTER_THRESHOLD = 100
        self.WIDTH_THRESHOLD = 100

    def detect_slot(self, frame, dets):
        if frame is None:
            return False, frame

        # Making a copy to draw on, though we can draw on original
        h, w, _ = frame.shape

        roi_right = frame[int(h*0.55):h, int(w*0.45):w]
        roi_left  = frame[int(h*0.55):h, 0:int(w*0.55)]

        def process_roi(roi, x_offset):
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)

            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50,
                                    minLineLength=50, maxLineGap=25)

            vertical_lines = []
            horizontal_lines = []

            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]

                    x1 += x_offset
                    x2 += x_offset
                    y1 += int(h*0.55)
                    y2 += int(h*0.55)

                    angle = np.degrees(np.arctan2((y2 - y1), (x2 - x1 + 1e-5)))

                    if abs(angle) > 40:
                        vertical_lines.append((x1, y1, x2, y2))
                    elif abs(angle) < 40:
                        horizontal_lines.append((x1, y1, x2, y2))

            return vertical_lines, horizontal_lines

        # process both sides
        v_right, h_right = process_roi(roi_right, int(w*0.45))
        v_left,  h_left  = process_roi(roi_left, 0)

        # draw lines
        for (x1, y1, x2, y2) in v_right + v_left:
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        for (x1, y1, x2, y2) in h_right + h_left:
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        def process_slots(vertical_lines, side):
            if len(vertical_lines) < 2:
                return False

            xs = [x for (x1, _, x2, _) in vertical_lines for x in (x1, x2)]
            xs = sorted(xs)

            if len(xs) >= 4:
                q1 = np.percentile(xs, 25)
                q3 = np.percentile(xs, 75)
                xs = [x for x in xs if q1 <= x <= q3]

            xs = sorted(list(set(xs)))

            best_slot = None

            for i in range(len(xs) - 1):
                left_x = xs[i]
                right_x = xs[i+1]
                width = right_x - left_x

                if 80 < width < 300:
                    best_slot = (left_x, right_x)
                    break

            if best_slot is None:
                return False

            new_l, new_r = best_slot
            
            # --- Check if occupied by a car/obstacle from YOLO dets ---
            occupied = False
            for d in dets:
                if d["label"].lower() in ("car", "obstacle", "roadblock", "closed-road-stand"):
                    bx1, by1, bx2, by2 = d["bbox"]
                    # Check horizontal overlap and vertical proximity
                    if bx2 > new_l and bx1 < new_r and by2 > h * 0.40:
                        occupied = True
                        break

            new_center = (new_l + new_r) // 2
            new_width = new_r - new_l

            if side == "L":
                tracked = self.tracked_left_slot
            else:
                tracked = self.tracked_right_slot

            if occupied:
                # Reset tracking if slot is occupied so it doesn't lock
                tracked = None
                if side == "L": self.tracked_left_slot = None
                else: self.tracked_right_slot = None
                # Draw red box to indicate occupied slot being skipped
                cv2.rectangle(frame, (new_l, int(h * 0.6)), (new_r, h), (0, 0, 255), 3)
                cv2.putText(frame, f"{side} OCCUPIED", (new_l + 5, int(h * 0.6) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                return False

            if tracked is not None:
                old_l, old_r = tracked
                old_center = (old_l + old_r) // 2
                old_width = old_r - old_l

                if (abs(old_center - new_center) < self.CENTER_THRESHOLD and
                    abs(old_width - new_width) < self.WIDTH_THRESHOLD):
                    updated_l = int(self.ALPHA * old_l + (1 - self.ALPHA) * new_l)
                    updated_r = int(self.ALPHA * old_r + (1 - self.ALPHA) * new_r)
                    tracked = (updated_l, updated_r)
                else:
                    tracked = (new_l, new_r)
            else:
                tracked = (new_l, new_r)

            if side == "L":
                self.tracked_left_slot = tracked
            else:
                self.tracked_right_slot = tracked

            left_x, right_x = tracked
            y_top = int(h * 0.6)
            y_bottom = h

            cv2.rectangle(frame, (left_x, y_top), (right_x, y_bottom), (0, 255, 255), 3)
            cv2.putText(frame, f"{side} Slot 1", (left_x + 5, y_top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            return True

        slot_right = process_slots(v_right, "R")
        slot_left  = process_slots(v_left, "L")

        return (slot_right or slot_left), frame
