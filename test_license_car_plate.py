# test_license_car_plate.py
import time
import os
import cv2
import subprocess
from ultralytics import YOLO

# 引入你的單例 OCR 模組 (請確保同目錄下有 ocr_module.py)
from ocr_module import LicensePlateOCR

# ==========================================
# 功能一：圖片 - 單純定位車牌外框
# ==========================================
def run_image_plate_localization(model_path, image_path):
    print("🚀 [系統資訊] 正在初始化 YOLOv8 定位模型...")
    try:
        model = YOLO(model_path)
        print("✅ [系統資訊] 定位模型載入成功！")
    except Exception as e:
        print(f"❌ [錯誤] 無法載入模型: {str(e)}")
        return None

    print(f"📸 [系統資訊] 開始定位圖片中的車牌: {image_path}")
    results = model.predict(source=image_path, conf=0.25, verbose=False)
    
    for result in results:
        boxes = result.boxes
        print(f"📊 [分析結果] 畫面上共定位到 {len(boxes)} 個車牌位置。")
        for i, box in enumerate(boxes):
            conf = box.conf.item() if hasattr(box.conf, 'item') else box.conf[0]
            print(f"   🔹 車牌位置 {i+1}: 置信度={conf:.2f}")
            
    print("🎉 [系統資訊] 車牌定位流程執行完畢！")
    return results[0].plot()


# ==========================================
# 功能二：圖片 - 定位車牌 + 英數字辨識 (OCR)
# ==========================================
def run_image_ocr_pipeline(model_path, image_path):
    print("🚀 [系統資訊] 正在啟動『定位 + 文字辨識』雙階段流水線...")
    
    # 取得或初始化 OCR 單例實例 (確保只會 init 一次)
    ocr_helper = LicensePlateOCR()
    
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"❌ [錯誤] 無法載入 YOLO 模型: {str(e)}")
        return None

    # 用 OpenCV 讀取原始圖片，方便切片裁剪
    img = cv2.imread(image_path)
    if img is None:
        print("❌ [錯誤] 無法讀取圖片，請檢查路徑。")
        return None

    results = model.predict(source=img, conf=0.25, verbose=False)
    
    plate_count = 0
    for result in results:
        for box in result.boxes:
            plate_count += 1
            # 取得車牌座標
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # 安全防護：防止邊界溢出
            h, w, _ = img.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            # 利用 NumPy 矩陣切片直接裁剪出車牌局部小圖
            cropped_plate = img[y1:y2, x1:x2]
            
            # 送入單例 OCR 辨識文字
            print(f"⏳ [OCR 進行中] 正在辨識第 {plate_count} 片車牌內的英數字...")
            plate_number = ocr_helper.recognize_plate(cropped_plate)
            print(f"🚗 [辨識結果] 第 {plate_count} 片車牌號碼為: 【{plate_number}】")
            
            # 在圖片上繪製紅色邊框與辨識文字
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(img, plate_number, (x1, y1 - 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            
    if plate_count == 0:
        print("📊 [分析結果] 畫面上未偵測到任何車牌，無法進行 OCR。")
        
    print("🎉 [系統資訊] 雙階段車牌文字辨識執行完畢！")
    return img


# ==========================================
# 功能三：影片 - 使用 GPU 進行批量加速辨識
# ==========================================
def run_video_detection_with_gpu(model_path, video_path, output_path, compressed_path):
    print("🚀 [系統資訊] 正在初始化影片處理模型...")
    import torch
    
    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ [系統資訊] 目前使用的運算裝置: {'NVIDIA GPU (CUDA)' if device == '0' else 'CPU'}")
    
    model = YOLO(model_path)
    
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # BATCH 推理效能優化
    BATCH_SIZE = 16  
    frames_buffer = []
    frame_count = 0
    
    print(f"🎬 開始讀取影片，啟用 GPU 批量推理 (Batch Size = {BATCH_SIZE})...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frames_buffer.append(frame)
        frame_count += 1
        
        if len(frames_buffer) == BATCH_SIZE:
            results = model(frames_buffer, device=device, verbose=False)
            
            for idx, result in enumerate(results):
                f = frames_buffer[idx]
                for box in result.boxes:
                    # 修正：使用 squeeze 確保一維座標抽取不報錯
                    x1, y1, x2, y2 = map(int, box.xyxy.squeeze())
                    conf = float(box.conf.squeeze())
                    
                    cv2.rectangle(f, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(f, f"Plate {conf:.2f}", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                out.write(f)
                
            frames_buffer = [] 
            
            if frame_count % (BATCH_SIZE * 2) == 0:
                print(f"⏳ 已完成 {frame_count} 影格之車牌辨識...")

    # 剩餘影格處理
    if len(frames_buffer) > 0:
        results = model(frames_buffer, device=device, verbose=False)
        for idx, result in enumerate(results):
            f = frames_buffer[idx]
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy.squeeze())
                conf = float(box.conf.squeeze())
                cv2.rectangle(f, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(f, f"Plate {conf:.2f}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            out.write(f)
            
    cap.release()
    out.release()
    print(f"✅ 影像偵測處理完成！總計：{frame_count} 幀。")
    
    print("⏳ 正在啟動 FFmpeg 壓縮影片格式以供網頁播放...")
    ffmpeg_command = ["ffmpeg", "-y", "-i", output_path, "-vcodec", "libx264", "-crf", "28", "-preset", "fast", compressed_path]
    res = subprocess.run(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return res.returncode == 0


# ==========================================
# 功能四：影片 - 標準逐幀辨識 (CPU 或舊版相容用)
# ==========================================
def run_video_detection(model_path, video_path, output_path, compressed_path):
    print("🚀 [系統資訊] 正在初始化影片處理模型...")
    model = YOLO(model_path)
    
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    print("🎬 開始讀取影片並逐幀進行車牌偵測...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        results = model(frame, verbose=False)
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy.squeeze())
                conf = float(box.conf.squeeze())
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Plate {conf:.2f}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        out.write(frame)
        frame_count += 1
        
        if frame_count % 10 == 0:
            print(f"⏳ 已處理 {frame_count} 影格...")
            
    cap.release()
    out.release()
    print(f"✅ 影像偵測處理完成！總計：{frame_count} 幀。")
    
    print("⏳ 正在啟動 FFmpeg 壓縮影片格式以供網頁播放...")
    ffmpeg_command = [
        "ffmpeg", "-y", "-i", output_path, "-vcodec", "libx264", 
        "-crf", "28", "-preset", "fast", compressed_path
    ]
    res = subprocess.run(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    if res.returncode == 0:
        print("🎉 [系統資訊] 影片壓縮成功，網頁準備就緒！")
        return True
    else:
        print(f"⚠️ [警告] FFmpeg 壓縮失敗。將僅提供原始影片下載。")
        return False
