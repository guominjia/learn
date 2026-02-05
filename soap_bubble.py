from typing import List, Tuple, Set
from collections import defaultdict
import random

class SoapBubble:
    def __init__(self, grid_size: Tuple[int, int], boundary_points: dict = None):
        """
        初始化肥皂泡网格
        :param grid_size: (rows, cols) 网格大小
        :param boundary_points: {(x, y): height} 边界点及其高度
        """
        self.rows, self.cols = grid_size
        self.boundary_points = boundary_points or {}
        self.V = defaultdict(float)
        
        # 初始化所有网格点
        self.status = []
        for i in range(self.rows):
            for j in range(self.cols):
                point = (i, j)
                self.status.append(point)
                if point in self.boundary_points:
                    self.V[point] = self.boundary_points[point]
                else:
                    self.V[point] = random.random()
    
    def get_neighbors(self, point: Tuple[int, int]) -> List[Tuple[int, int]]:
        """获取四个相邻网格点"""
        x, y = point
        neighbors = []
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # 右、左、下、上
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.rows and 0 <= ny < self.cols:
                neighbors.append((nx, ny))
        
        return neighbors
    
    def random_walk(self, start_point: Tuple[int, int]) -> float:
        """
        从起始点开始随机游走直到到达边界
        :param start_point: 起始点坐标
        :return: 到达边界点的高度
        """
        current = start_point
        
        # 如果起始点就是边界，直接返回
        if current in self.boundary_points:
            return self.boundary_points[current]
        
        # 随机游走直到到达边界
        while current not in self.boundary_points:
            neighbors = self.get_neighbors(current)
            current = random.choice(neighbors)
        
        return self.boundary_points[current]
    
    def learn(self, num_episodes: int = 1000):
        """
        使用蒙特卡洛方法估计每个网格点的高度
        :param num_episodes: 每个网格点的随机游走次数
        """
        returns = defaultdict(list)
        
        # 对每个非边界点进行多次随机游走
        for point in self.status:
            if point not in self.boundary_points:
                for _ in range(num_episodes):
                    boundary_height = self.random_walk(point)
                    returns[point].append(boundary_height)
        
        # 更新每个点的高度为所有游走的平均值
        for point in returns:
            self.V[point] = sum(returns[point]) / len(returns[point])
    
    def iterative_solve(self, iterations: int = 1000, tolerance: float = 1e-6):
        """
        使用迭代方法求解（用于对比）
        每个点的高度是其四个邻居的平均值
        """
        for _ in range(iterations):
            max_change = 0
            new_V = self.V.copy()
            
            for point in self.status:
                if point not in self.boundary_points:
                    neighbors = self.get_neighbors(point)
                    avg_height = sum(self.V[n] for n in neighbors) / len(neighbors)
                    new_V[point] = avg_height
                    max_change = max(max_change, abs(new_V[point] - self.V[point]))
            
            self.V = new_V
            
            if max_change < tolerance:
                break
    
    def print_surface(self):
        """打印表面高度"""
        print("\n肥皂泡表面高度:")
        for i in range(self.rows):
            row = []
            for j in range(self.cols):
                point = (i, j)
                if point in self.boundary_points:
                    row.append(f"[{self.V[point]:.2f}]")
                else:
                    row.append(f" {self.V[point]:.2f} ")
            print(" ".join(row))


if __name__ == "__main__":
    # 创建一个简单的矩形边界
    # 5x5网格，边界高度为1.0，模拟一个平面边框
    boundary = {}
    grid_size = (5, 5)
    
    # 设置边界点（矩形边框）
    for i in range(grid_size[0]):
        boundary[(i, 0)] = 1.0  # 左边界
        boundary[(i, grid_size[1]-1)] = 1.0  # 右边界
    for j in range(grid_size[1]):
        boundary[(0, j)] = 1.0  # 上边界
        boundary[(grid_size[0]-1, j)] = 1.0  # 下边界
    
    # 测试蒙特卡洛方法
    print("=" * 50)
    print("蒙特卡洛方法:")
    print("=" * 50)
    mc_bubble = SoapBubble(grid_size, boundary)
    mc_bubble.learn(num_episodes=5000)
    mc_bubble.print_surface()
    
    # 测试迭代方法（用于对比）
    print("\n" + "=" * 50)
    print("迭代方法:")
    print("=" * 50)
    iter_bubble = SoapBubble(grid_size, boundary)
    iter_bubble.iterative_solve(iterations=1000)
    iter_bubble.print_surface()
    
    # 计算两种方法的差异
    print("\n" + "=" * 50)
    print("两种方法的差异:")
    print("=" * 50)
    total_diff = 0
    for point in mc_bubble.status:
        if point not in boundary:
            diff = abs(mc_bubble.V[point] - iter_bubble.V[point])
            total_diff += diff
    print(f"平均差异: {total_diff / len([p for p in mc_bubble.status if p not in boundary]):.6f}")