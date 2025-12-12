import cv2
import numpy as np
import os

def count_cells_contours(image_path, output_path=None):
    img = cv2.imread(image_path)
    if img is None:
        print(f"無法讀取圖片: {image_path}")
        return

    output_img = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. 高斯模糊 (比中值模糊更能保留整體形狀)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 2. 自適應閾值 (Adaptive Thresholding)
    # 這比固定閾值好，因為它能處理照片中光線不均勻的問題
    # block_size: 鄰域大小，必須是奇數 (例如 11, 15, 19)
    # C: 常數，從平均值中減去的值 (調整這個可以控制對雜訊的敏感度)
    thresh = cv2.adaptiveThreshold(blurred, 255, 
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, # 背景變黑，細胞變白
                                   15, # Block Size
                                   2)  # C: 降到 2 讓線條更粗，更容易連接
                                   
    # 3. 形態學操作 (Morphological Operations)
    # 使用 Convex Hull (凸包) 技術來解決半圓和中空問題
    # 這裡只需要輕微的閉運算來連接非常細微的斷裂
    kernel = np.ones((3,3), np.uint8)
    processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    sure_bg = processed # 用於除錯圖片輸出
    
    # 4. 尋找輪廓
    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    count = 0
    
    # 5. 過濾輪廓 (使用 Convex Hull)
    for cnt in contours:
        # 計算凸包 (Convex Hull)
        # 這是關鍵：它會自動把 "C" 字形、半圓、或空心環變成一個實心的多邊形
        hull = cv2.convexHull(cnt)
        
        # 使用凸包的面積來過濾
        area = cv2.contourArea(hull)
        
        # 條件 A: 面積過濾
        # 因為是凸包，面積會比原本的輪廓大，所以 min_area 可以稍微調高一點點以過濾雜訊
        min_area = 15 
        max_area = 2500
        
        if min_area < area < max_area:
            # 條件 B: 圓度過濾 (使用凸包的周長和面積)
            perimeter = cv2.arcLength(hull, True)
            if perimeter == 0: continue
            
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            
            # 過濾掉網格線 (通常圓度很低)
            if circularity > 0.4: 
                count += 1
                # 畫出凸包 (綠色) - 這樣您可以看到程式把半圓補成了什麼樣子
                cv2.drawContours(output_img, [hull], -1, (0, 255, 0), 2)
                # 畫出中心點
                M = cv2.moments(hull)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    cv2.circle(output_img, (cX, cY), 2, (0, 0, 255), -1)

    print(f"[方法二] 圖片 {os.path.basename(image_path)} 中檢測到的球藻數量: {count}")

    if output_path:
        cv2.imwrite(output_path, output_img)
        # 也可以存一下二值化的中間過程，方便除錯
        cv2.imwrite(output_path.replace('.jpeg', '_thresh.jpeg'), sure_bg)
        print(f"結果已儲存至: {output_path}")

if __name__ == "__main__":
    input_image = "1.jpeg" 
    output_image = "result_v2_1.jpeg"
    
    if os.path.exists(input_image):
        count_cells_contours(input_image, output_image)
    else:
        files = [f for f in os.listdir('.') if f.endswith('.jpeg') or f.endswith('.jpg')]
        for f in files:
            if "result" not in f:
                count_cells_contours(f, f"result_v2_{f}")
