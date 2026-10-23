高清海上目标检测数据集  基于 YOLOv11 海上各类船只检测系统![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/24e571bb483548ce94b460604836ee74.png)

），9891张，yolo，voc，coco三种标注方式![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/5c0e237b793f4fc6bfa30336d9b1bc7f.png)

图像尺寸:1080*1920
类别数量:9类![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/e8fb9254956f4ed98dd1f8efa3d70659.png)

训练集图像数量:4445; 验证集图像数量:1390； 测试集图像数量:635
类别名称: 每一类图像数 ，每一类标注数![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/f8aae5ed59f9431ba576b555679c4b7d.png)

Vessel-ship：轮船 / 货船，6300，35389
Buoy：浮标，567，750
Boat：小船 / 艇，350，382
Speed boat：快艇，1548，2240
Ferry：渡轮，1572，2237
Other：其他（工具），2962，6515
Sail boat：帆船，616，616
Flying bird-plane：飞鸟 - 飞机等飞行目，194，231
Kayak：皮划艇，200，865
image num: 6470
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/a568102376ab4dd182b0ce3e572d2592.png)


模型代码
采用 YOLOv11n 网络训练
训练轮次：80 个 epoch![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/7c7b540ead3448af8a7ba3e6b8dcfb98.png)

提供全部训练 + 测试源代码
训练精度 mAP 效果如图所示
PyQt5 界面功能![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/30692d2475e94292814233c06d89a944.png)

界面使用 PyQt5 开发，提供全部源码（.ui、.qrc、.py 及图标文件）
支持图片检测、视频检测、摄像头实时检测
界面实时显示：目标位置、目标总数、置信度等信息
支持检测结果保存导出
操作简单直观，无需命令行
4、
运行环境：Python=3.8、opencv-python、PyQt5、torch
支持 Windows、Linux 系统

# 基于YOLOv11海上各类船只检测系统
## 1、数据集配置文件 ship.yaml
```yaml
path: "./ship_dataset"
train: images/train
val: images/val
test: images/test

names:
  0: Vessel-ship
  1: Buoy
  2: Boat
  3: Speed boat
  4: Ferry
  5: Other
  6: Sail boat
  7: Flying bird-plane
  8: Kayak
```

## 2、训练脚本 train_ship.py
```python
from ultralytics import YOLO

if __name__ == '__main__':
    # 加载YOLOv11n预训练权重
    model = YOLO("yolo11n.pt")
    results = model.train(
        data="ship.yaml",
        epochs=100,
        imgsz=640,
        batch=8,
        device=0,
        workers=2,
        project="ship_detect_result",
        name="yolo11_ship",
        amp=True,
        mosaic=0.8,
        conf=0.25
    )
```

## 3、推理脚本 predict_ship.py
```python
from ultralytics import YOLO
import cv2

model = YOLO("./ship_detect_result/yolo11_ship/weights/best.pt")

def ship_detect(img_path, save_path="ship_out.jpg"):
    res = model(img_path)[0]
    draw_img = res.plot()
    cv2.imwrite(save_path, draw_img)
    return draw_img, res.boxes

if __name__ == "__main__":
    ship_detect("test_ship.jpg")
```

## 4、PyQt5可视化界面 ship_gui.py
```python
import sys
import cv2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel, QFileDialog,
                             QTableWidget, QTableWidgetItem, QComboBox, QGroupBox, QVBoxLayout, QHBoxLayout, QWidget)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
from ultralytics import YOLO

class ShipDetectUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("基于YOLOv11的海上各类型船只检测系统")
        self.resize(1400, 980)
        self.model = YOLO("./ship_detect_result/yolo11_ship/weights/best.pt")
        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        left_layout = QVBoxLayout()
        self.img_label = QLabel("图像显示区域")
        self.img_label.setFixedSize(840, 660)
        left_layout.addWidget(self.img_label)

        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["序号", "类别", "置信度", "坐标", "文件路径"])
        left_layout.addWidget(self.table)

        right_layout = QVBoxLayout()
        group_file = QGroupBox("文件导入")
        self.btn_img = QPushButton("选择图片")
        self.btn_img.clicked.connect(self.load_img)
        self.btn_video = QPushButton("选择视频")
        self.btn_cam = QPushButton("开启摄像头")
        v1 = QVBoxLayout()
        v1.addWidget(self.btn_img)
        v1.addWidget(self.btn_video)
        v1.addWidget(self.btn_cam)
        group_file.setLayout(v1)

        group_info = QGroupBox("检测结果")
        self.label_time = QLabel("用时：0 s | 目标数目：0")
        self.combo_cls = QComboBox()
        self.combo_cls.addItems(["全部", "Vessel-ship","Buoy","Boat","Speed boat","Ferry","Other","Sail boat","Flying bird-plane","Kayak"])
        self.label_pos = QLabel("xmin:0 ymin:0 xmax:0 ymax:0")
        self.label_conf = QLabel("置信度：0%")
        self.btn_save = QPushButton("保存结果")
        v2 = QVBoxLayout()
        v2.addWidget(self.label_time)
        v2.addWidget(self.combo_cls)
        v2.addWidget(self.label_pos)
        v2.addWidget(self.label_conf)
        v2.addWidget(self.btn_save)
        group_info.setLayout(v2)

        right_layout.addWidget(group_file)
        right_layout.addWidget(group_info)
        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)

    def load_img(self):
        fpath, _ = QFileDialog.getOpenFileName(self, "打开图片", ".", "图片(*.jpg *.png *.jpeg)")
        if not fpath:
            return
        pred = self.model(fpath, conf=0.25)[0]
        draw_img = pred.plot()
        self.show_img(draw_img, self.img_label)
        self.table.setRowCount(0)
        for idx, box in enumerate(pred.boxes):
            cls_name = pred.names[int(box.cls)]
            conf = float(box.conf)
            xyxy = box.xyxy[0].cpu().numpy()
            self.table.insertRow(idx)
            self.table.setItem(idx, 0, QTableWidgetItem(str(idx+1)))
            self.table.setItem(idx,1,QTableWidgetItem(cls_name))
            self.table.setItem(idx,2,QTableWidgetItem(f"{conf:.2f}"))
            self.table.setItem(idx,3,QTableWidgetItem(f"{xyxy.round(0)}"))
            self.table.setItem(idx,4,QTableWidgetItem(fpath))
        self.label_time.setText(f"用时：{pred.speed['inference']/1000:.3f} s | 目标数目：{len(pred.boxes)}")

    def show_img(self,cv_img,label):
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h,w,c = rgb_img.shape
        qimg = QImage(rgb_img.data, w, h, w*c, QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(qimg).scaled(label.size(),Qt.KeepAspectRatio))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = ShipDetectUI()
    win.show()
    sys.exit(app.exec_())
```
## 环境安装
```bash
pip install ultralytics torch opencv-python pyqt5
```
## 关键词标签
海上船只检测，YOLOv11，船舶识别，海事监控，海上目标检测，PyQt5可视化界面
## 应用场景
1. 港口、近海海域监控，识别各类船舶、浮标、皮划艇等目标
2. 海上安防、船舶流量统计，异常船只预警
3. 支持图片、视频、海上摄像头实时检测，输出目标坐标与置信度
4. 海事巡检、海岸雷达辅助视觉监测


