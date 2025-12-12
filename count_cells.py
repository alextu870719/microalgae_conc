import cv2
import numpy as np
import os

def count_cells(image_path, output_path=None):
    # 1. 讀取圖片
    img = cv2.imread(image_path)
    if img is None:
        print(f"無法讀取圖片: {image_path}")
        return

    output_img = img.copy()
    
    # 2. 轉換為灰階
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 3. 模糊處理 (減少雜訊)
    # 根據圖片清晰度，可能需要調整 kernel size (9, 9)
    gray_blurred = cv2.medianBlur(gray, 3)
    
    # 4. 霍夫圓變換 (Hough Circle Transform)
    # 這些參數非常關鍵，通常需要根據實際圖片進行微調：
    # dp: 解析度倒數比，1 代表與原圖一樣，2 代表一半解析度
    # minDist: 圓心之間的最小距離 (太小會導致誤判多個圓，太大會漏算)
    # param1: Canny 邊緣檢測的高閾值
    # param2: 圓心檢測閾值 (越小檢測越多圓，但也越多誤判)
    # minRadius: 最小半徑
    # maxRadius: 最大半徑
    
    # 注意：這裡的參數是預設值，您可能需要根據照片中球藻的大小進行調整
    circles = cv2.HoughCircles(gray_blurred, 
                               cv2.HOUGH_GRADIENT, 
                               dp=1, 
                               minDist=12,  # 兩個球藻圓心最小距離 (像素)
                               param1=50, 
                               param2=23, 
                               minRadius=4, # 球藻最小半徑
                               maxRadius=40) # 球藻最大半徑

    count = 0
    if circles is not None:
        circles = np.uint16(np.around(circles))
        count = circles.shape[1]
        
        for i in circles[0, :]:
            # 畫出圓的外框 (綠色)
            cv2.circle(output_img, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # 畫出圓心 (紅色)
            cv2.circle(output_img, (i[0], i[1]), 2, (0, 0, 255), 3)

    print(f"圖片 {os.path.basename(image_path)} 中檢測到的球藻數量: {count}")

    # 5. 儲存結果圖片
    if output_path:
        cv2.imwrite(output_path, output_img)
        print(f"結果已儲存至: {output_path}")

if __name__ == "__main__":
    # 測試第一張圖片
    # 您可以修改這裡來測試其他圖片，例如 '2.jpeg'
    input_image = "1.jpeg" 
    output_image = "result_1.jpeg"
    
    if os.path.exists(input_image):
        count_cells(input_image, output_image)
    else:
        # 如果找不到 1.jpeg，嘗試搜尋目錄下的所有 jpeg
        files = [f for f in os.listdir('.') if f.endswith('.jpeg') or f.endswith('.jpg')]
        if files:
            print(f"找到圖片: {files}")
            for f in files:
                count_cells(f, f"result_{f}")
        else:
            print("當前目錄下找不到 .jpeg 圖片")
