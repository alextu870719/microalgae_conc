import cv2
import os
import glob

def split_images(source_dir, output_dir, crop_size=640):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 搜尋所有 jpeg/jpg 圖片
    image_paths = glob.glob(os.path.join(source_dir, "*.jpeg")) + \
                  glob.glob(os.path.join(source_dir, "*.jpg"))

    print(f"找到 {len(image_paths)} 張原始圖片")

    for img_path in image_paths:
        filename = os.path.basename(img_path)
        name_no_ext = os.path.splitext(filename)[0]
        
        # 略過已經是結果圖的檔案
        if "result" in filename or "thresh" in filename:
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w, _ = img.shape
        print(f"正在處理 {filename} (大小: {w}x{h})...")

        # 計算可以切成幾塊
        # 這裡我們不重疊切圖，方便您標註
        count = 0
        for y in range(0, h, crop_size):
            for x in range(0, w, crop_size):
                # 確保切出來的圖不會超過邊界
                # 如果剩下的部分太小 (小於 1/3)，就忽略，避免邊緣不完整的細胞
                if (h - y) < (crop_size / 3) or (w - x) < (crop_size / 3):
                    continue
                
                # 裁切
                crop_img = img[y:y+crop_size, x:x+crop_size]
                
                # 如果裁切出來的大小不足 640x640 (在邊緣)，用黑色填滿
                ch, cw, _ = crop_img.shape
                if ch < crop_size or cw < crop_size:
                    padded = cv2.copyMakeBorder(crop_img, 0, crop_size-ch, 0, crop_size-cw, cv2.BORDER_CONSTANT, value=(0,0,0))
                    crop_img = padded

                save_name = f"{name_no_ext}_y{y}_x{x}.jpg"
                cv2.imwrite(os.path.join(output_dir, save_name), crop_img)
                count += 1
        
        print(f"  -> 已切成 {count} 張小圖")

if __name__ == "__main__":
    # 當前目錄
    source = "." 
    # 輸出目錄
    output = "yolo_workspace/images_to_label"
    
    split_images(source, output)
    print("\n完成！請前往 yolo_workspace/images_to_label 資料夾挑選圖片進行標註。")
