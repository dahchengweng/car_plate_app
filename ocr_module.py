# ocr_module.py
import easyocr
import torch
import re

class LicensePlateOCR:
    _instance = None

    # 實作 Singleton (單例模式) 核心邏輯
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(LicensePlateOCR, cls).__new__(cls, *args, **kwargs)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        # 確保模型只會被載入記憶體一次
        if self._initialized:
            return
            
        print("⏳ [OCR 系統] 正在初始化單例 EasyOCR 模型並載入權重...")
        
        # 檢查是否有 NVIDIA GPU 可用
        self.use_gpu = torch.cuda.is_available()
        
        # 載入英文與數字辨識器 (車牌主要為英數字，只載入 'en' 可以大幅加快速度與準確率)
        self.reader = easyocr.Reader(['en'], gpu=self.use_gpu)
        
        self._initialized = True
        print(f"✅ [OCR 系統] 單例模型載入成功！硬體加速狀態：{'GPU (CUDA)' if self.use_gpu else 'CPU'}")

    def recognize_plate(self, cropped_img):
        """
        輸入: 已經被 YOLO 裁剪下來的車牌區域圖片 (OpenCV 格式)
        輸出: 辨識出的車牌號碼字串
        """
        if cropped_img is None or cropped_img.size == 0:
            return "UNKNOWN"

        # 執行文字辨識 (detail=0 代表只回傳文字字串，不回傳座標與置信度)
        results = self.reader.readtext(cropped_img, detail=0)
        
        if not results:
            return "UNKNOWN"
            
        # 將辨識到的文字組合，並轉為大寫
        raw_text = "".join(results).upper()
        
        # 清洗字串：利用正規表示式，只保留標準車牌的英文字母、數字與連字號
        clean_text = re.sub(r'[^A-Z0-9-]', '', raw_text)
        
        return clean_text if clean_text else "UNKNOWN"
