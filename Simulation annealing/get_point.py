import math
import numpy as np

# arguments: Point x y, radius, lower bound, upper bound
def GetPointInCircle(Point, r, lb, ub):
    circle_r, x0, y0 = r, 0, 0
    alpha = 2 * math.pi * np.random.rand()
    r = circle_r * math.sqrt(np.random.rand())
    x = r * math.cos(alpha) + x0
    y = r * math.sin(alpha) + y0
    Coords = [x,y]
    NewPoint = Point + Coords
    while(NewPoint[0] < lb or NewPoint[0] > ub or NewPoint[1] < lb or NewPoint[1] > ub):
        NewPoint = GetPointInCircle(Point,r,lb,ub)
    return NewPoint
