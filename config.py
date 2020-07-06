PYTESSERACT_PATH = r"D:\Program Files\Tesseract-OCR\tesseract.exe"
IMAGE_UPSCALE_FACTOR = 3

RESOLUTION_CONFIGS = {
    (1920, 1080): {
        "NORMAL_MORPH_SIZE": (int(1920 * 6 / 1920), int(1080 * 2 / 1080)),
        "SMALL_MORPH_SIZE": (int(1920 * 2 / 1920), int(1080 * 2 / 1080)),
        "BLUR_SIZE": (int(1920 * 5 / 1920), int(1920 * 5 / 1920)),
        "NORMAL_MIN_TEXT_HEIGHT_LIMIT": int(1080 * 8 / 1080),
        "SMALL_MIN_TEXT_HEIGHT_LIMIT": int(1080 * 4 / 1080),
        "MAX_TEXT_HEIGHT_LIMIT": int(1080 * 50 / 1080),
        "ROW_THRESHOLD": int(1080 * 20 / 1080),
        "COL_THRESHOLD": int(1920 * 60 / 1920),
        "BIG_BORDER": 20,
        "BIG_PADDING": 5,
        "SMALL_PADDING": 3,
        "NAME_MIN_WIDTH": int(1920 * 200 / 1920),
        "NAME_COL_PADDING": int(1920 * 50 / 1920),
    }
}

DEFAULT_RESOLUTION_CONFIG = RESOLUTION_CONFIGS[(1920, 1080)]