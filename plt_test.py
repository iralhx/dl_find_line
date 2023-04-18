import cv2
import numpy as np

# 生成随机点集
points = np.random.randint(0, 256, (100, 2)).astype(np.int32)

# 拟合直线
vx, vy, x, y = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)

# 计算直线起点和终点
lefty = int((-x * vy / vx) + y)
righty = int(((256 - x) * vy / vx) + y)

# 绘制直线
img = np.zeros((256, 256), dtype=np.uint8)
cv2.polylines(img, [np.array([(0, lefty), (256 - 1, righty)])], False, 255)

