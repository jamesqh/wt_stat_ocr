from collections import Counter, defaultdict
import cv2 as cv
import itertools
import numpy as np
import os
import pytesseract

from config import *

pytesseract.pytesseract.tesseract_cmd = PYTESSERACT_PATH

TOP_LEFT_TEMPLATE = cv.imread("templates/topleft.jpg", cv.IMREAD_GRAYSCALE)
TOP_RIGHT_TEMPLATE = cv.imread("templates/topright.jpg", cv.IMREAD_GRAYSCALE)
BOTTOM_LEFT_TEMPLATE = cv.imread("templates/bottomleft.jpg", cv.IMREAD_GRAYSCALE)
BOTTOM_RIGHT_TEMPLATE = cv.imread("templates/bottomright.jpg", cv.IMREAD_GRAYSCALE)

FLAGS = {
    "USA": cv.imread("templates/american.jpg", cv.IMREAD_UNCHANGED),
    "Britain": cv.imread("templates/british.jpg", cv.IMREAD_UNCHANGED),
    "China": cv.imread("templates/chinese.jpg", cv.IMREAD_UNCHANGED),
    "France": cv.imread("templates/french.jpg", cv.IMREAD_UNCHANGED),
    "Germany": cv.imread("templates/german.jpg", cv.IMREAD_UNCHANGED),
    "Italy": cv.imread("templates/italian.jpg", cv.IMREAD_UNCHANGED),
    "Sweden": cv.imread("templates/swedish.jpg", cv.IMREAD_UNCHANGED),
    "USSR": cv.imread("templates/russian.jpg", cv.IMREAD_UNCHANGED),
    "Japan": cv.imread("templates/japanese.jpg", cv.IMREAD_UNCHANGED),
}

COLOURS = itertools.cycle(
    [
        (255, 34, 0),
        (255, 170, 0),
        (218, 230, 172),
        (116, 217, 0),
        (64, 255, 242),
        (0, 95, 102),
        (61, 133, 242),
        (182, 61, 242),
        (115, 0, 77),
        (255, 191, 208),
    ]
)


def equalize_colour_hist(img):
    for c in range(0, 2):
        img[:, :, c] = cv.equalizeHist(img[:, :, c])
    return img


FLAGS = {k: equalize_colour_hist(v) for k, v in FLAGS.items()}
# FLAGS = {k: cv.resize(v, (0, 0), fx=3, fy=3) for k, v in FLAGS.items()}


def check_repeats(text):
    i = 1
    while i <= len(text) // 3 + 1:
        if text[0:i].strip() == text[i : 2 * i].strip() == text[2 * i : 3 * i].strip():
            return text[0:i].strip()
        i += 1
    return text


class ScreenshotOCRInterface:
    def __init__(self, image_path, queue, debug=False):
        self.queue = queue
        self.debug = debug
        self.filename, self.ext = os.path.splitext(os.path.basename(image_path))
        self.image_colour = cv.imread(image_path, cv.IMREAD_UNCHANGED)
        self.image_bw = cv.cvtColor(self.image_colour, cv.COLOR_RGB2GRAY)
        w, h = self.image_bw.shape[::-1]
        if (w, h) in RESOLUTION_CONFIGS:
            self.config = RESOLUTION_CONFIGS[(w, h)]
        else:
            self.config = DEFAULT_RESOLUTION_CONFIG
        self.crop_bw, self.crop_colour = self.crop_stats_window(
            self.image_bw, self.image_colour
        )
        if self.debug:
            self.save_debug(self.crop_colour, 0)
        self.rows = None

    def ocr_row(self, row):
        row_roi = self.crop_cell(self.join_boxes(row), self.crop_colour)
        import base64

        _, img_enc = cv.imencode(".jpg", row_roi)
        img_enc_b64 = base64.b64encode(img_enc).decode()
        self.queue.put({"tag": "update_row_reference_img", "args": img_enc_b64})
        self.queue.put({"tag": "clear_data_preview", "args": None})

        def update(data):
            self.queue.put({"tag": "update_data_preview", "args": data})

        record = {}
        big_padding = self.config["BIG_PADDING"]
        small_padding = self.config["SMALL_PADDING"]
        record["row_num"] = self.ocr_digits_only(
            ~self.crop_cell(row[0], self.crop_bw, padding=big_padding)
        )
        update(record)
        try:
            record["rank"] = self.ocr_digits_only(self.crop_cell(row[1], self.crop_bw, padding=big_padding), single_digit=True)
        except:
            pass
        update(record)
        record["country"] = self.match_flag(
            self.crop_cell(row[2], self.crop_colour, padding=big_padding)
        )
        update(record)
        record["name"] = self.ocr_name(
            self.crop_cell(
                row[4],
                self.crop_bw,
                padding=small_padding,
                min_width=self.config["NAME_MIN_WIDTH"],
            )
        )
        update(record)
        record["wins"] = self.ocr_digits_only(self.crop_cell(row[5], self.crop_bw, padding=big_padding))
        update(record)
        record["battles"] = self.ocr_digits_only(self.crop_cell(row[6], self.crop_bw, padding=big_padding))
        update(record)
        record["win_percent"] = self.ocr_percentage(self.crop_cell(row[7], self.crop_bw))
        update(record)
        record["respawns"] = self.ocr_digits_only(self.crop_cell(row[8], self.crop_bw, padding=big_padding))
        update(record)
        record["deaths"] = self.ocr_digits_only(self.crop_cell(row[9], self.crop_bw, padding=big_padding))
        update(record)
        record["air_kills"] = self.ocr_digits_only(self.crop_cell(row[10], self.crop_bw, padding=big_padding))
        update(record)
        record["ground_kills"] = self.ocr_digits_only(self.crop_cell(row[11], self.crop_bw, padding=big_padding))
        update(record)
        record["boat_kills"] = self.ocr_digits_only(self.crop_cell(row[12], self.crop_bw, padding=big_padding))
        update(record)
        record["sl_earned"] = self.ocr_abbreviated_num(self.crop_cell(row[13], self.crop_bw))
        update(record)
        record["rp_earned"] = self.ocr_abbreviated_num(self.crop_cell(row[14], self.crop_bw))
        update(record)
        record["b64_img_preview"] = img_enc_b64
        return record

    def find_rows(self):
        table_boxes = []
        pre = self.pre_process_image(self.crop_bw)
        if self.debug:
            self.save_debug(pre, 1)
        boxes = self.find_text_boxes(pre)
        if self.debug:
            self.write_debug_image_with_overlay(self.crop_colour, boxes, 2)
        rows = self.find_table_in_boxes(boxes)
        if self.debug:
            self.write_debug_image_with_overlay(self.crop_colour, rows, 3)
        rows = self.isolate_row_bounding_boxes(rows)
        if self.debug:
            self.write_debug_image_with_overlay(self.crop_colour, rows, 4)
        for row in rows:
            table_row = []
            x, y, w, h = row
            row_crop = self.crop_bw[y : y + h, x : x + w]
            dilate = self.pre_process_image(
                row_crop, morph_size=self.config["SMALL_MORPH_SIZE"]
            )
            boxes = self.find_text_boxes(
                dilate, min_text_height_limit=self.config["SMALL_MIN_TEXT_HEIGHT_LIMIT"]
            )
            for box in boxes:
                X, Y, W, H = box
                table_boxes.append((x + X, y + Y, W, H))
        if self.debug:
            self.write_debug_image_with_overlay(self.crop_colour, table_boxes, 5)
        cell_mask = np.zeros(self.crop_bw.shape, dtype=np.uint8)
        for box in table_boxes:
            x, y, w, h = box
            cv.rectangle(cell_mask, (x, y), (x + w, y + h), (255, 255, 255), -1)
        if self.debug:
            self.save_debug(cell_mask, 6)
        confirmed_cols = np.apply_along_axis(
            lambda col: sum(col) / len(col) > 0.5, 0, cell_mask
        )
        confirmed_cols = np.nonzero(confirmed_cols)[0]
        cols = []
        col_start = confirmed_cols[0]
        prev_col = col_start
        for col in confirmed_cols[1:]:
            if col != prev_col + 1:
                cols.append([col_start, prev_col])
                col_start = col
            prev_col = col
        cols.append([col_start, prev_col])
        name_col_pad = self.config["NAME_COL_PADDING"]
        if cols[4][1] + name_col_pad > cols[5][0]:
            cols[4][1] = max(cols[4][1] + name_col_pad, cols[5][1])
            del cols[5]
        table_rows = []
        for row in rows:
            table_row = []
            _, y, _, h = row
            for col in cols:
                x, X = col
                box = (x, y, X - x, h)
                table_row.append(box)
            table_rows.append(table_row)
        self.write_debug_image_with_overlay(self.crop_colour, table_rows, 7)
        self.rows = table_rows

    def write_debug_image_with_overlay(self, image, overlay, debug_num):
        debug_image = image.copy()
        from collections.abc import Iterable

        if isinstance(overlay[0][0], Iterable):
            for row in overlay:
                colour = next(COLOURS)
                for cell in row:
                    x, y, w, h = cell
                    cv.rectangle(debug_image, (x, y), (x + w, y + h), colour)
        else:
            for box in overlay:
                x, y, w, h = box
                cv.rectangle(debug_image, (x, y), (x + w, y + h), (255, 255, 255))
        self.save_debug(debug_image, debug_num)

    def ocr(self, image, psm=1, allowed_chars=None):
        image = cv.threshold(image, 127, 255, cv.THRESH_BINARY)[1]
        image = cv.resize(
            image, (0, 0), fx=IMAGE_UPSCALE_FACTOR, fy=IMAGE_UPSCALE_FACTOR
        )
        if allowed_chars is not None:
            cmd_args = (
                f"-l eng --oem 0 --psm {psm} -c tessedit_char_whitelist={allowed_chars}"
            )
        else:
            cmd_args = f"-l eng --oem 1 --psm {psm}"
        text = pytesseract.image_to_string(image, config=cmd_args)
        return text

    def crop_stats_window(self, image, image_colour):
        templates = {
            (0, 0): TOP_LEFT_TEMPLATE,
            (0, 1): TOP_RIGHT_TEMPLATE,
            (1, 0): BOTTOM_LEFT_TEMPLATE,
            (1, 1): BOTTOM_RIGHT_TEMPLATE,
        }
        corners = []
        for corner, template in templates.items():
            w, h = template.shape[::-1]
            method = cv.TM_SQDIFF_NORMED
            res = cv.matchTemplate(image, template, method)
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
            if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
                top_left = min_loc
            else:
                top_left = max_loc
            bottom_right = (top_left[0] + w, top_left[1] + h)
            corners.append((top_left[0] + corner[1] * w, top_left[1] + corner[0] * h))
        top_left = (min(c[0] for c in corners), min(c[1] for c in corners))
        bottom_right = (max(c[0] for c in corners), max(c[1] for c in corners))
        y, x = top_left
        Y, X = bottom_right
        crop = image[x:X, y:Y].copy()
        crop_colour = image_colour[x:X, y:Y].copy()
        return crop, crop_colour

    def pre_process_image(self, image, morph_size=None):
        if morph_size is None:
            morph_size = self.config["NORMAL_MORPH_SIZE"]
        blur = cv.GaussianBlur(image, self.config["BLUR_SIZE"], 0)
        pre = cv.threshold(blur, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
        struct = cv.getStructuringElement(cv.MORPH_RECT, morph_size)
        return cv.dilate(pre, struct, anchor=(-1, -1), iterations=1)

    def find_text_boxes(
        self, pre, min_text_height_limit=None, max_text_height_limit=None
    ):
        if min_text_height_limit is None:
            min_text_height_limit = self.config["NORMAL_MIN_TEXT_HEIGHT_LIMIT"]
        if max_text_height_limit is None:
            max_text_height_limit = self.config["MAX_TEXT_HEIGHT_LIMIT"]
        contours, hierarchy = cv.findContours(pre, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        boxes = []
        for contour in contours:
            box = cv.boundingRect(contour)
            h = box[3]
            if min_text_height_limit < h < max_text_height_limit:
                boxes.append(box)
        return boxes

    def join_boxes(self, boxes):
        left, top, w, h = boxes[0]
        right = left + w
        bottom = top + h
        for box in boxes[1:]:
            x, y, w, h = box
            left = min(left, x)
            top = min(top, y)
            right = max(right, x + w)
            bottom = max(bottom, y + h)
        return left, top, right - left, bottom - top

    def find_table_in_boxes(
        self, boxes, row_threshold=None, col_threshold=None, min_columns=13, min_rows=5
    ):
        if row_threshold is None:
            row_threshold = self.config["ROW_THRESHOLD"]
        if col_threshold is None:
            col_threshold = self.config["COL_THRESHOLD"]
        rows = defaultdict(lambda: [])
        cols = defaultdict(lambda: [])
        for box in boxes:
            (x, y, w, h) = box
            col_key = int(round((x + w / 4) / col_threshold))
            row_key = int(round((y + h / 2) / row_threshold))
            cols[col_key].append(box)
            rows[row_key].append(box)
        new_cols = {
            col_key: col for col_key, col in cols.items() if len(col) > min_rows
        }
        for col_key, col in cols.items():
            if len(col) < min_rows:
                prev_col = max(c for c in new_cols.keys() if c < col_key)
                new_cols[prev_col] += col
        cols = new_cols
        new_rows = defaultdict(lambda: [])
        for col_key, row_key in itertools.product(cols.keys(), rows.keys()):
            boxes = [box for box in cols[col_key] if box in rows[row_key]]
            if len(boxes) > 0:
                box = self.join_boxes(boxes)
                new_rows[row_key].append(box)
        rows = new_rows
        table_rows = list(filter(lambda r: len(r) >= min_columns, rows.values()))
        table_rows = [list(sorted(tb)) for tb in table_rows]
        table_rows = list(sorted(table_rows, key=lambda r: r[0][1]))
        return table_rows

    def isolate_row_bounding_boxes(self, rows):
        return [self.join_boxes(row) for row in rows]

    def crop_cell(self, cell, image, min_width=0, min_height=0, padding=0):
        x, y, w, h = cell
        w = max(w + padding, min_width)
        h = max(h + padding, min_height)
        crop = image[y - padding : y + h, x - padding : x + w].copy()
        return crop

    def tile_image(self, image, tile=3):
        return np.tile(image, (tile, tile))

    def save_debug(self, image, debug_num):
        filename = f"{self.filename}-{debug_num}.{self.ext}"
        filepath = os.path.join("debug", filename)
        if not os.path.exists("debug"):
            os.mkdir("debug")
        cv.imwrite(filepath, image)

    def ocr_digits_only(self, roi, single_digit=False):
        border = 20
        roi = ~roi
        border_roi = cv.copyMakeBorder(
            roi, border, border, border, border, cv.BORDER_CONSTANT, None, None
        )
        tile_roi = self.tile_image(border_roi)
        text = self.ocr(tile_roi, allowed_chars="0123456789", psm=6)
        text = max(Counter(text.replace("\n", " ").split(" ")))
        if text == "":
            text = self.ocr(self.tile_image(roi), allowed_chars="0123456789", psm=6)
            text = max(Counter(text.replace("\n", " ").split(" ")))
            text = check_repeats(text)
        if single_digit:
            text = text[0]
        return text

    def match_flag(self, roi):
        roi = equalize_colour_hist(roi)
        country_scores = {
            x: cv.minMaxLoc(cv.matchTemplate(roi, FLAGS[x], cv.TM_CCOEFF_NORMED))[1]
            for x in FLAGS.keys()
        }
        return max(country_scores.keys(), key=lambda x: country_scores[x])

    def ocr_name(self, roi):
        roi = self.tile_image(~roi)
        texts = self.ocr(roi, psm=3).strip().split("\n")
        return [
            name
            for name in set(check_repeats(text) for text in texts)
            if name.strip() != ""
        ]
    
    def ocr_percentage(self, roi):
        border = self.config["BIG_BORDER"]
        roi = cv.copyMakeBorder(roi, border, border, border, border, cv.BORDER_REPLICATE, None, None)
        roi = self.tile_image(roi)
        text = self.ocr(roi, allowed_chars="0123456789%", psm=6)
        answer = max(Counter(text.replace("\n", " ").split(" ")))
        if answer.endswith("96") or answer.endswith("41") or answer.endswith("95"):
            answer = answer[:-2] + "%"
        return answer
    
    def ocr_abbreviated_num(self, roi):
        border = self.config["BIG_BORDER"]
        roi = cv.copyMakeBorder(roi, border, border, border, border, cv.BORDER_REPLICATE, None, None)
        roi = self.tile_image(roi)
        text = self.ocr(roi, allowed_chars="0123456789.KM", psm=6)
        text = text.replace(" .", ".").replace("\n", " ")
        return max(Counter(text.split(" ")))


if __name__ == "__main__":
    filename = os.path.join("screenshots", os.listdir("screenshots")[0])
    reader = ScreenshotOCRInterface(filename, debug=True)
    reader.find_rows()
