import cv2
import numpy as np

# カメラのキャプチャを開始
cap = cv2.VideoCapture(0)

while True:
    # フレームをキャプチャ
    ret, frame = cap.read()

    # 画像をグレースケールに変換
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Cannyエッジ検出を適用
    edges = cv2.Canny(gray, 50, 150)

    # 直線を検出
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10
    )

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 赤色の直線を描画

    # 結果を表示
    cv2.imshow("Frame", frame)

    # 'q'キーを押して終了
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# キャプチャを解放してウィンドウを閉じる
cap.release()
cv2.destroyAllWindows()
