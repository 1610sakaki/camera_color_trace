import cv2
import numpy as np
import time


class RouteNavigatorFromImage:
    def __init__(self) -> None:
        # 判定ブロックエリア座標
        self.resize_width, self.resize_height = 194, 160

        # 赤色の範囲を定義
        self.lower_red = np.array([0, 50, 50])
        self.upper_red = np.array([10, 255, 255])

        # 緑色の範囲を定義
        self.lower_green = np.array([50, 100, 50])
        self.upper_green = np.array([70, 255, 255])

        # 青色の範囲を定義
        self.lower_blue = np.array([100, 50, 50])
        self.upper_blue = np.array([130, 255, 255])

        # 中心座標の枠色
        self.center_color = (0, 255, 255)

        # 中心に描く四角形の大きさ
        self.square_size = 100

        # track座標の色
        self.tracking_color = (0, 255, 255)

        # マーカー面積の閾値割合
        self.ratio_threshold = 0.9

        # 実行間隔
        self.time_sleep = 0.1

        # カメラのキャプチャを開始
        self.cap = cv2.VideoCapture(0)

    # 受け取った画像メッセージをOpenCV形式の画像に変換して、指定色の範囲内のピクセルを抽出
    def __image_converter(self):
        # 受け取った画像メッセージをOpenCV形式の画像に変換
        # self.bridge = CvBridge()

        # self.converted_image = self.bridge.imgmsg_to_cv2(
        #     image_message, desired_encoding="passthrough"
        # )
        _, self.converted_image = self.cap.read()


        # 画面中央と重心の位置を比較して、方向を決定
        self.frame_height, self.frame_width = self.converted_image.shape[:2]

        # 四角形を描画
        self.rect_width = self.frame_height * 16 // 9
        self.rect_height = self.frame_height
        self.rect_x = self.frame_width // 2 - self.rect_width // 2
        self.rect_y = 0
        self.rect_thickness = 5

        # フレームをHSV色空間に変換
        self.hsv = cv2.cvtColor(self.converted_image, cv2.COLOR_BGR2HSV)

        # 指定色の範囲内のピクセルを抽出
        if self.color == "red":
            self.mask = cv2.inRange(self.hsv, self.lower_red, self.upper_red)
        elif self.color == "blue":
            self.mask = cv2.inRange(self.hsv, self.lower_blue, self.upper_blue)
        elif self.color == "green":
            self.mask = cv2.inRange(self.hsv, self.lower_green, self.upper_green)

    # 最大の輪郭を見つけて、画像中に図形を描画
    def __find_max_outline(self):
        contours, _ = cv2.findContours(
            self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if len(contours) > 0:
            max_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(max_contour)

            if M["m00"] > 0:
                # 輪郭の重心を計算
                self.cX, self.cY = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])

                # 重心を示す円を描画
                cv2.circle(
                    self.converted_image, (self.cX, self.cY), 5, self.tracking_color, -1
                )

                # 中心座標に線を描画
                # 縦線
                cv2.line(
                    self.converted_image,
                    (self.frame_width // 2, self.frame_height // 2 - 50),
                    (self.frame_width // 2, self.frame_height // 2 + 50),
                    self.center_color,
                    self.rect_thickness,
                )
                # 横線
                cv2.line(
                    self.converted_image,
                    (self.frame_width // 2 - 50, self.frame_height // 2),
                    (self.frame_width // 2 + 50, self.frame_height // 2),
                    self.center_color,
                    self.rect_thickness,
                )

                # 中心座標を囲む正方形を描画
                self.square_x = self.frame_width // 2 - self.square_size // 2
                self.square_y = self.frame_height // 2 - self.square_size // 2
                cv2.rectangle(
                    self.converted_image,
                    (self.square_x, self.square_y),
                    (
                        self.square_x + self.square_size,
                        self.square_y + self.square_size,
                    ),
                    self.center_color,
                    self.rect_thickness,
                )
                # 正方形の領域を切り出す
                self.square_region = self.hsv[
                    self.square_y : self.square_y + self.square_size,
                    self.square_x : self.square_x + self.square_size,
                    :,
                ]
                # 正方形の総ピクセル数を計算
                self.total_pixels = self.square_size * self.square_size

    # 進路方向を決定する
    def __track_determine(self, color):
        if color == "red":
            lower_color = self.lower_red
            upper_color = self.upper_red
        elif color == "green":
            lower_color = self.lower_green
            upper_color = self.upper_green
        elif color == "blue":
            lower_color = self.lower_blue
            upper_color = self.upper_blue
        else:
            raise ValueError("Invalid color")

        # 色の範囲に絞り込むマスクを作成
        color_mask = cv2.inRange(self.square_region, lower_color, upper_color)

        # 色のピクセル数を計算
        pixels = np.sum(color_mask == 255)

        # 色の割合を計算
        color_ratio = pixels / self.total_pixels

        print(f"{color}の割合:", color_ratio)

        if self.cX < self.frame_width // 2:
            direction = "Left"
        elif self.cX > self.frame_width // 2:
            direction = "Right"
        # 重心が四角形の中に収まっていれば direction を "Straight" に設定
        if color_ratio > self.ratio_threshold:
            if color == "red":
                direction = "stop"
            elif color == "green":
                direction = "left_turn"
            elif color == "blue":
                direction = "right_turn"
        elif (
            self.square_x < self.cX < self.square_x + self.square_size
            and self.square_y < self.cY < self.square_y + self.square_size
        ):
            direction = "Straight"
        return direction

    # 解析結果の画像を出力する
    def __test_image_viewer(self, direction):
        # 方向を表示
        cv2.putText(
            self.converted_image,
            direction,
            (self.cX - 50, self.cY - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            self.tracking_color,
            2,
        )

        # 結果
        # print("進行方向：" + direction)
        time.sleep(self.time_sleep)
        cv2.imshow("Frame", self.converted_image)
        cv2.waitKey(1)

    # 他クラスでパブリックメソッドとして使用する
    def route_planner(self, color):
        # 指定の色を決定
        self.color = color

        # 受け取った画像メッセージをOpenCV形式の画像に変換して、指定色の範囲内のピクセルを抽出
        self.__image_converter()

        # 最大の輪郭を見つけて、画像中に図形を描画
        self.__find_max_outline()

        # 進路方向を決定する
        direction = self.__track_determine(self.color)

        # 解析結果の画像を出力する
        self.__test_image_viewer(direction)

        return direction

ot = RouteNavigatorFromImage()
while True:
    ot.route_planner("red")
