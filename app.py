# app.py
import streamlit as st
import os
import contextlib
import io
import cv2

# 引入你的車牌辨識模組
import test_license_car_plate

st.set_page_config(page_title="AI 影像辨識平台", layout="wide")
st.title("🚗 AI 多功能車牌影像辨識系統")

with st.sidebar:
    st.header("功能設定")
    app_mode = st.selectbox(
        "請選擇辨識功能：",
        ["1. 車牌辨識 (License Plate)", "2. 人臉辨識 (未來擴充)"]
    )

# ==========================================
# 自訂的 stdout 攔截器 (已修正參數問題)
# ==========================================
class StreamlitStdout:
    def __init__(self, container, log_key):
        self.container = container
        self.log_key = log_key
        self.string_io = io.StringIO()
        
    def write(self, text):
        self.string_io.write(text)
        # 即時寫入並指定獨一無二的 key
        self.container.text_area("系統日誌 (Log)", value=self.string_io.getvalue(), height=250)
        
    def flush(self):
        pass

# ==========================================
# 主要功能實作：車牌辨識
# ==========================================
if "1. 車牌辨識" in app_mode:
    # 使用 Tabs 分開「圖片」與「影片」
    tab_img, tab_vid = st.tabs(["📸 圖片辨識", "🎬 影片辨識"])
    
    model_file = "best.pt"
    
        # ------------------------------------------
    # 區塊一：圖片辨識與下載 (已修改函數名稱並新增功能按鈕)
    # ------------------------------------------
    with tab_img:
        st.subheader("圖片車牌偵測與文字辨識")
        uploaded_img = st.file_uploader("請上傳汽車照片", type=["jpg", "jpeg", "png"], key="img_uploader")
        
        if uploaded_img is not None:
            temp_img_path = "temp_upload.jpg"
            with open(temp_img_path, "wb") as f:
                f.write(uploaded_img.getbuffer())
                
            c1, c2 = st.columns(2)
            with c1:
                st.image(uploaded_img, caption="原始圖片", width="stretch")
            with c2:
                log_img_box = st.empty()
                log_img_box.text_area("系統日誌 (Log)", value="等待執行...", height=250, key="img_log_box")
                
                # 按鈕一：單純定位車牌
                btn_locate = st.button("🔍 步驟一：僅定位車牌外框", width="stretch", key="btn_img_locate")
                
                # 按鈕二：新增的車牌英數字辨識
                btn_ocr = st.button("🔤 步驟二：辨識車牌英數字 (OCR)", width="stretch", key="btn_img_ocr")
                
                # 初始化用來存放最終產出影像的變數
                res_img = None
                
                if not os.path.exists(model_file):
                    st.error(f"找不到 '{model_file}'！請將檔案放置專案目錄下。")
                else:
                    # 當點擊「僅定位車牌外框」
                    if btn_locate:
                        custom_stdout = StreamlitStdout(log_img_box, log_key="img_log_box")
                        with contextlib.redirect_stdout(custom_stdout):
                            res_img = test_license_car_plate.run_image_plate_localization(model_file, temp_img_path)
                    
                    # 當點擊「辨識車牌英數字 (OCR)」
                    elif btn_ocr:
                        custom_stdout = StreamlitStdout(log_img_box, log_key="img_log_box")
                        with contextlib.redirect_stdout(custom_stdout):
                            res_img = test_license_car_plate.run_image_ocr_pipeline(model_file, temp_img_path)
                    
                    # 如果有成功產出處理後的影像，將它渲染在左側並提供下載
                    if res_img is not None:
                        with c1:
                            res_rgb = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)
                            st.image(res_rgb, caption="🎉 處理結果", width="stretch")
                        
                        is_success, buffer = cv2.imencode(".jpg", res_img)
                        if is_success:
                            st.download_button(
                                label="💾 下載處理後圖片",
                                data=buffer.tobytes(),
                                file_name="result_plate_processed.jpg",
                                mime="image/jpeg",
                                width="stretch",
                                key="btn_img_download"
                            )

    # ------------------------------------------
    # 區塊二：影片辨識、播放與下載
    # ------------------------------------------
    with tab_vid:
        st.subheader("影片車牌偵測")
        uploaded_vid = st.file_uploader("請上傳汽車行駛影片", type=["mp4", "avi", "mov"], key="vid_uploader")
        
        if uploaded_vid is not None:
            temp_vid_path = "temp_upload.mp4"
            out_vid_path = "output_raw.mp4"
            comp_vid_path = "output_compressed.mp4"
            
            with open(temp_vid_path, "wb") as f:
                f.write(uploaded_vid.getbuffer())
                
            c3, c4 = st.columns(2)
            with c3:
                st.video(temp_vid_path)
                st.caption("原始影片")
            with c4:
                log_vid_box = st.empty()
                log_vid_box.text_area("系統日誌 (Log)", value="等待執行...", height=250, key="vid_log_box")
                
                if st.button("🚀 開始影片辨識 (GPU 批量加速)", width="stretch", key="btn_vid_start"):
                    if not os.path.exists(model_file):
                        st.error(f"找不到 '{model_file}' 權重檔案！")
                    else:
                        custom_stdout = StreamlitStdout(log_vid_box, log_key="vid_log_box")
                        with contextlib.redirect_stdout(custom_stdout):
                            # 執行影片偵測與轉碼
                            success = test_license_car_plate.run_video_detection_with_gpu(
                                model_file, temp_vid_path, out_vid_path, comp_vid_path
                            )
                        
                        if success and os.path.exists(comp_vid_path):
                            with c3:
                                st.video(comp_vid_path)
                                st.caption("🎉 辨識後影片（已轉碼壓縮）")
                            
                            with open(comp_vid_path, "rb") as file:
                                st.download_button(
                                    label="💾 下載辨識後影片 (MP4)",
                                    data=file,
                                    file_name="result_plate_video.mp4",
                                    mime="video/mp4",
                                    width="stretch",
                                    key="btn_vid_download"
                                )
                        elif os.path.exists(out_vid_path):
                            with open(out_vid_path, "rb") as file:
                                st.download_button(
                                    label="💾 下載辨識後影片 (原始碼檔)",
                                    data=file,
                                    file_name="result_plate_raw.mp4",
                                    mime="video/mp4",
                                    width="stretch",
                                    key="btn_vid_download_raw"
                                )
else:
    st.info("💡 請從左側下拉式選單選擇『1. 車牌辨識』。")
