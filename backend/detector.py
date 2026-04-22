import cv2
import numpy as np
import math
from ultralytics import YOLO


class AdvancedViolationDetector:
    def __init__(
        self,
        model_path="yolov8n.pt",
        helmet_model_path="runs/detect/helmet_yolov8n/weights/best.pt",
    ):
        self.base_model = YOLO(model_path)

        try:
            self.helmet_model = YOLO(helmet_model_path)
            self.has_helmet_model = True
            print("[Detector] Specialized helmet model loaded.")
        except Exception:
            self.has_helmet_model = False
            print("[Detector] No helmet model found – using color/shape heuristic fallback.")

    # ──────────────────────────────────────────────────────────────────────────
    # Geometry helpers
    # ──────────────────────────────────────────────────────────────────────────
    def get_center(self, box):
        return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2

    def get_distance(self, p1, p2):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def calculate_iou(self, box1, box2):
        xi1 = max(box1[0], box2[0])
        yi1 = max(box1[1], box2[1])
        xi2 = min(box1[2], box2[2])
        yi2 = min(box1[3], box2[3])
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        if inter_area == 0:
            return 0
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area

    # ──────────────────────────────────────────────────────────────────────────
    # Helmet heuristic (fallback when no trained model)
    # ──────────────────────────────────────────────────────────────────────────
    def _heuristic_helmet(self, image: np.ndarray, person_box) -> bool:
        """
        Analyse the upper ~30% of a person bounding box (head region).
        Detects helmets via:
          1) Explicit colored-helmet HSV ranges (yellow, blue, red, black, white)
          2) Compact non-skin blob analysis as fallback
        """
        x1 = max(0, int(person_box[0]))
        y1 = max(0, int(person_box[1]))
        x2 = min(image.shape[1], int(person_box[2]))
        y2 = min(image.shape[0], int(person_box[3]))

        person_h = y2 - y1
        person_w = x2 - x1

        if person_h < 20 or person_w < 10:
            return False

        # Head region: top 30% of person box
        head_y2 = y1 + int(person_h * 0.30)
        # Slight horizontal inset
        hx1 = x1 + int(person_w * 0.08)
        hx2 = x2 - int(person_w * 0.08)

        head_roi = image[y1:head_y2, hx1:hx2]
        if head_roi.size == 0:
            return False

        hsv = cv2.cvtColor(head_roi, cv2.COLOR_BGR2HSV)
        total_px = head_roi.shape[0] * head_roi.shape[1]
        if total_px == 0:
            return False

        # ── 1. Explicit helmet color detection ─────────────────────────────
        # Yellow helmets (hue 20-35, high saturation)
        yellow_mask = cv2.inRange(hsv,
                                  np.array([18, 100, 100], dtype=np.uint8),
                                  np.array([38, 255, 255], dtype=np.uint8))
        # Blue helmets (hue 95-135)
        blue_mask = cv2.inRange(hsv,
                                np.array([90, 80, 60], dtype=np.uint8),
                                np.array([140, 255, 255], dtype=np.uint8))
        # Red helmets (hue 0-10 and 165-180, high saturation)
        red_mask1 = cv2.inRange(hsv,
                                np.array([0, 120, 80], dtype=np.uint8),
                                np.array([10, 255, 255], dtype=np.uint8))
        red_mask2 = cv2.inRange(hsv,
                                np.array([160, 120, 80], dtype=np.uint8),
                                np.array([180, 255, 255], dtype=np.uint8))
        # White helmets (low saturation, high value)
        white_mask = cv2.inRange(hsv,
                                 np.array([0, 0, 180], dtype=np.uint8),
                                 np.array([180, 40, 255], dtype=np.uint8))
        # Black helmets (low value)
        black_mask = cv2.inRange(hsv,
                                 np.array([0, 0, 0], dtype=np.uint8),
                                 np.array([180, 255, 55], dtype=np.uint8))

        color_helmet_mask = (yellow_mask | blue_mask | red_mask1 |
                             red_mask2 | white_mask | black_mask)
        color_px = int(cv2.countNonZero(color_helmet_mask))
        color_ratio = color_px / total_px

        # If a strong colored region is present in the head area → likely helmet
        if color_ratio > 0.35:
            return True

        # ── 2. Skin-tone mask fallback ──────────────────────────────────────
        skin_lower1 = np.array([0,  30, 50],  dtype=np.uint8)
        skin_upper1 = np.array([22, 255, 255], dtype=np.uint8)
        skin_lower2 = np.array([160, 30, 50],  dtype=np.uint8)
        skin_upper2 = np.array([180, 255, 255], dtype=np.uint8)
        skin_mask = (cv2.inRange(hsv, skin_lower1, skin_upper1) |
                     cv2.inRange(hsv, skin_lower2, skin_upper2))

        skin_px    = int(cv2.countNonZero(skin_mask))
        non_skin_r = (total_px - skin_px) / total_px

        # ── 3. Compact-blob check ───────────────────────────────────────────
        fg_mask = cv2.bitwise_not(skin_mask)
        bright_mask = cv2.inRange(hsv,
                                  np.array([0, 0, 40], dtype=np.uint8),
                                  np.array([180, 255, 255], dtype=np.uint8))
        fg_mask = cv2.bitwise_and(fg_mask, bright_mask)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        has_compact_region = False
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < (total_px * 0.12):   # at least 12% of head roi
                continue
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = (4 * math.pi * area) / (perimeter ** 2)
            if circularity > 0.25:          # reasonably round/oval → likely helmet
                has_compact_region = True
                break

        # Decision: non-skin dominant AND a compact blob present → helmet
        return non_skin_r > 0.40 and has_compact_region

    # ──────────────────────────────────────────────────────────────────────────
    # Main detection
    # ──────────────────────────────────────────────────────────────────────────
    def detect(self, image: np.ndarray):
        img_h, img_w = image.shape[:2]

        # ── Step 1: Base model – detect motorcycles and persons ──────────────
        try:
            base_results = self.base_model(image, verbose=False)[0]
            boxes       = base_results.boxes.xyxy.cpu().numpy()
            classes     = base_results.boxes.cls.cpu().numpy()
            confidences = base_results.boxes.conf.cpu().numpy()
        except Exception:
            return image, []

        motorcycles   = []   # list of (box, conf)
        persons       = []

        for box, cls, conf in zip(boxes, classes, confidences):
            if conf < 0.3:
                continue
            name = self.base_model.names[int(cls)].lower()
            if name == "motorcycle":
                motorcycles.append((box, float(conf)))
            elif name == "person":
                persons.append(box)

        # ── Step 2: Helmet detection ─────────────────────────────────────────
        helmet_boxes   = []   # confirmed WITH helmet (conf >= 0.50)
        no_helmet_boxes = []  # confirmed WITHOUT helmet (conf >= 0.45)
        if self.has_helmet_model:
            try:
                h_results   = self.helmet_model(image, verbose=False)[0]
                h_boxes     = h_results.boxes.xyxy.cpu().numpy()
                h_classes   = h_results.boxes.cls.cpu().numpy()
                h_confs     = h_results.boxes.conf.cpu().numpy()
                for box, cls, conf in zip(h_boxes, h_classes, h_confs):
                    name = self.helmet_model.names[int(cls)]
                    if name in ("With Helmet", "helmet", "Helmet"):
                        if conf >= 0.55:          # higher bar for positive
                            helmet_boxes.append(box)
                    elif name in ("Without Helmet", "no_helmet", "No Helmet"):
                        if conf >= 0.45:          # slightly lower bar for negative
                            no_helmet_boxes.append(box)
            except Exception:
                pass

        # ── Step 3: Assign persons to motorcycles ────────────────────────────
        person_to_mc: dict[int, list] = {}

        for p_box in persons:
            p_cx, p_cy = self.get_center(p_box)
            best_mc  = -1
            best_iou = 0.05   # minimum overlap to be counted as rider

            for m_idx, (m_box, _) in enumerate(motorcycles):
                iou = self.calculate_iou(p_box, m_box)

                # Also consider persons whose centre falls inside/above the bike
                if iou < 0.05:
                    # Expanded motorcycle region (riders sit above the bike)
                    exp_y1 = m_box[1] - (m_box[3] - m_box[1]) * 0.9
                    if (m_box[0] <= p_cx <= m_box[2]) and (exp_y1 <= p_cy <= m_box[3]):
                        iou = max(iou, 0.06)   # small artificial boost

                if iou > best_iou:
                    best_iou = iou
                    best_mc  = m_idx

            if best_mc != -1:
                person_to_mc.setdefault(best_mc, []).append(p_box)

        # ── Step 4: Annotate and build response ──────────────────────────────
        annotated_img = image.copy()
        response_data = []

        # Draw scale-adaptive font/line thickness
        scale      = max(img_w, img_h) / 800
        font_scale = max(0.5, min(0.9, scale))
        thickness  = max(2, int(scale * 2))

        for m_idx, (m_box, mc_conf) in enumerate(motorcycles):
            riders = person_to_mc.get(m_idx, [])
            if len(riders) == 0:
                continue          # skip bikes with no assigned riders

            # Draw motorcycle box (bright green)
            cv2.rectangle(
                annotated_img,
                (int(m_box[0]), int(m_box[1])),
                (int(m_box[2]), int(m_box[3])),
                (0, 230, 0), thickness,
            )

            helmet_status = []

            for p_box in riders:
                # ── Helmet check ──────────────────────────────────────────
                if self.has_helmet_model:
                    def _box_overlaps(h_box, p_box, thresh=0.40):
                        """True if h_box overlaps p_box by > thresh of h_box area."""
                        h_area = (h_box[2]-h_box[0]) * (h_box[3]-h_box[1])
                        if h_area <= 0:
                            return False
                        iy1 = max(p_box[1], h_box[1]); iy2 = min(p_box[3], h_box[3])
                        ix1 = max(p_box[0], h_box[0]); ix2 = min(p_box[2], h_box[2])
                        inter = max(0, ix2-ix1) * max(0, iy2-iy1)
                        return (inter / h_area) > thresh

                    # Check positive evidence (With Helmet)
                    has_helmet = any(_box_overlaps(h, p_box) for h in helmet_boxes)

                    # Negative evidence overrides: if a 'Without Helmet' box
                    # matches this person, mark as no helmet regardless
                    if has_helmet:
                        if any(_box_overlaps(h, p_box) for h in no_helmet_boxes):
                            has_helmet = False
                else:
                    # Fallback heuristic
                    has_helmet = self._heuristic_helmet(image, p_box)

                helmet_status.append(has_helmet)

                # Cyan = helmet, Red = no helmet
                p_color = (0, 200, 255) if has_helmet else (0, 0, 230)
                cv2.rectangle(
                    annotated_img,
                    (int(p_box[0]), int(p_box[1])),
                    (int(p_box[2]), int(p_box[3])),
                    p_color, thickness,
                )
                label = "Helmet" if has_helmet else "No Helmet"
                cv2.putText(
                    annotated_img, label,
                    (int(p_box[0]), max(10, int(p_box[1]) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.65, p_color, thickness,
                )

            # ── Violation logic ───────────────────────────────────────────
            rider_count = len(riders)
            violations  = []
            if rider_count > 2:
                violations.append("Triple Riding")
            if False in helmet_status:
                violations.append("No Helmet")

            mc_violation = " + ".join(violations) if violations else "Safe"

            v_color = (0, 220, 0) if mc_violation == "Safe" else (0, 0, 220)

            # MC label above box
            mc_label = f"Bike {m_idx+1}: {mc_violation}"
            label_y  = max(20, int(m_box[1]) - 8)
            cv2.putText(
                annotated_img, mc_label,
                (int(m_box[0]), label_y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.75, v_color, thickness,
            )

            response_data.append({
                "id":        m_idx + 1,
                "riders":    rider_count,
                "helmets":   helmet_status,
                "violation": mc_violation,
                "confidence": round(mc_conf, 2),
            })

        return annotated_img, response_data
