import cv2
import numpy as np
import time

# 赤色の範囲を定義
lower_red = np.array([0, 50, 50])
upper_red = np.array([10, 255, 255])

# カメラのキャプチャを開始
cap = cv2.VideoCapture(0)

while True:
    # フレームをキャプチャ
    ret, frame = cap.read()

    # フレームをHSV色空間に変換
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 赤色の範囲内のピクセルを抽出
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # 輪郭を検出
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 最大の輪郭を見つける
    if len(contours) > 0:
        max_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(max_contour)

        if M["m00"] > 0:
            # 輪郭の重心を計算
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            # 画面中央と重心の位置を比較して、方向を決定
            frame_height, frame_width = frame.shape[:2]
            if cX < frame_width // 2:
                direction = "Left"
            else:
                direction = "Right"

            # 重心を示す円を描画
            cv2.circle(frame, (cX, cY), 5, (0, 255, 0), -1)

            # 中心座標に線を描画
            cv2.line(
                frame,
                (frame_width // 2, frame_height // 2 - 50),
                (frame_width // 2, frame_height // 2 + 50),
                (0, 255, 0),
                2,
            )
            cv2.line(
                frame,
                (frame_width // 2 - 50, frame_height // 2),
                (frame_width // 2 + 50, frame_height // 2),
                (0, 255, 0),
                2,
            )

            # 横:縦 = 16:9の四角形を描画
            rect_width = frame_height * 16 // 9
            rect_height = frame_height
            rect_x = frame_width // 2 - rect_width // 2
            rect_y = 0
            cv2.rectangle(
                frame,
                (rect_x, rect_y),
                (rect_x + rect_width, rect_y + rect_height),
                (0, 255, 0),
                2,
            )

            # 方向を表示
            cv2.putText(
                frame,
                direction,
                (cX - 50, cY - 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            # 結果
            print("進行方向："+direction)
            time.sleep(0.25)

    # 結果を表示
    cv2.imshow("Frame", frame)

    # 'q'キーを押して終了
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# キャプチャを解放してウィンドウを閉じる
cap.release()
cv2.destroyAllWindows()
