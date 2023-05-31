import cv2
import numpy as np
import time


class ObjectTracker:
    def __init__(self) -> None:

        # 赤色の範囲を定義
        self.lower_red = np.array([0, 50, 50])
        self.upper_red = np.array([10, 255, 255])

        # 中心座標の枠色
        self.center_color = (0, 255, 255)

        # track座標の色
        self.tracking_color = (0, 255, 0)

        # 赤色面積の閾値割合
        self.red_ratio_threshold = 0.9

        # カメラのキャプチャを開始
        self.cap = cv2.VideoCapture(0)

    def frame_draw(self, frame_height, frame_width):
        # 四角形を描画
        self.rect_width = frame_height * 16 // 9
        self.rect_height = frame_height
        self.rect_x = frame_width // 2 - self.rect_width // 2
        self.rect_y = 0
        self.rect_thickness = 5

    def tracking(self):
        while True:
            # フレームをキャプチャ
            _, frame = self.cap.read()

            # 画面中央と重心の位置を比較して、方向を決定
            frame_height, frame_width = frame.shape[:2]
            self.frame_draw(frame_height, frame_width)

            # フレームをHSV色空間に変換
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # 赤色の範囲内のピクセルを抽出
            mask = cv2.inRange(hsv, self.lower_red, self.upper_red)

            # 輪郭を検出
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            # 最大の輪郭を見つける
            if len(contours) > 0:
                max_contour = max(contours, key=cv2.contourArea)
                M = cv2.moments(max_contour)
                print("max_contour:", max_contour, "M:", M["m00"])

                if M["m00"] > 0:
                    # 輪郭の重心を計算
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])

                    # 中心座標に線を描画
                    # 縦線
                    cv2.line(
                        frame,
                        (frame_width // 2, frame_height // 2 - 50),
                        (frame_width // 2, frame_height // 2 + 50),
                        self.center_color,
                        self.rect_thickness,
                    )
                    # 横線
                    cv2.line(
                        frame,
                        (frame_width // 2 - 50, frame_height // 2),
                        (frame_width // 2 + 50, frame_height // 2),
                        self.center_color,
                        self.rect_thickness,
                    )

                    # 中心座標を囲む正方形を描画
                    square_size = 100
                    square_x = frame_width // 2 - square_size // 2
                    square_y = frame_height // 2 - square_size // 2
                    cv2.rectangle(
                        frame,
                        (square_x, square_y),
                        (square_x + square_size, square_y + square_size),
                        self.center_color,
                        self.rect_thickness,
                    )
                    # 正方形の領域を切り出す
                    square_region = hsv[
                        square_y : square_y + square_size,
                        square_x : square_x + square_size,
                        :,
                    ]
                    # 赤色の範囲に絞り込むマスクを作成
                    mask = cv2.inRange(square_region, self.lower_red, self.upper_red)

                    # 赤色のピクセル数を計算
                    red_pixels = np.sum(mask == 255)

                    # 正方形の総ピクセル数を計算
                    total_pixels = square_size * square_size

                    # 赤色の割合を計算
                    red_ratio = red_pixels / total_pixels

                    print("赤色の割合:", red_ratio)

                    if M["m00"] < 100:
                        direction = None
                    else:
                        if cX < frame_width // 2:
                            direction = "Left"
                        else:
                            direction = "Right"
                        # 重心が四角形の中に収まっていれば direction を "Straight" に設定
                        if red_ratio > self.red_ratio_threshold:
                            direction = "stop"
                        elif (
                            square_x < cX < square_x + square_size
                            and square_y < cY < square_y + square_size
                        ):
                            direction = "Straight"

                    if direction is not None:
                        # 重心を示す円を描画
                        cv2.circle(frame, (cX, cY), 5, self.tracking_color, -1)
                        # 方向を表示
                        cv2.putText(
                            frame,
                            direction,
                            (cX - 50, cY - 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            self.tracking_color,
                            2,
                        )

                    # 結果
                    print("進行方向：" + str(direction))
                    time.sleep(0.25)

            # 結果を表示
            cv2.imshow("Frame", frame)

            # 'q'キーを押して終了
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # キャプチャを解放してウィンドウを閉じる
        self.cap.release()
        cv2.destroyAllWindows()


camera_trace_route = ObjectTracker()
camera_trace_route.tracking()
