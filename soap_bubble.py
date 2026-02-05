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
                    self.V[point] = 0.0  # 未计算的点初始化为0
    
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
    
    def estimate_point(self, point: Tuple[int, int], num_episodes: int = 1000) -> float:
        """
        仅估计单个点的高度（按需计算）
        :param point: 目标点
        :param num_episodes: 随机游走次数
        :return: 估计的高度值
        """
        if point in self.boundary_points:
            return self.boundary_points[point]
        
        returns = []
        for _ in range(num_episodes):
            boundary_height = self.random_walk(point)
            returns.append(boundary_height)
        
        estimated_height = sum(returns) / len(returns)
        self.V[point] = estimated_height
        return estimated_height
    
    def estimate_points(self, points: List[Tuple[int, int]], num_episodes: int = 1000):
        """
        估计指定点的高度（按需计算）
        :param points: 需要估计的点列表
        :param num_episodes: 每个点的随机游走次数
        """
        for point in points:
            self.estimate_point(point, num_episodes)
    
    def learn(self, num_episodes: int = 1000, sample_points: List[Tuple[int, int]] = None):
        """
        使用蒙特卡洛方法估计网格点的高度
        :param num_episodes: 每个网格点的随机游走次数
        :param sample_points: 如果指定，只计算这些点；否则计算所有点
        """
        if sample_points is None:
            # 计算所有非边界点（会很慢）
            sample_points = [p for p in self.status if p not in self.boundary_points]
        
        returns = defaultdict(list)
        
        # 对指定点进行随机游走
        for point in sample_points:
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
        for iteration in range(iterations):
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
                print(f"迭代方法在第 {iteration+1} 次迭代后收敛")
                break
    
    def print_surface(self, show_all: bool = False):
        """
        打印表面高度
        :param show_all: 是否显示所有点（大网格时建议False）
        """
        if self.rows <= 10 and self.cols <= 10 or show_all:
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
        else:
            print(f"\n网格太大 ({self.rows}x{self.cols})，显示摘要信息:")
            non_boundary = [self.V[p] for p in self.status if p not in self.boundary_points]
            if non_boundary:
                print(f"  非边界点数量: {len(non_boundary)}")
                print(f"  高度范围: [{min(non_boundary):.4f}, {max(non_boundary):.4f}]")
                print(f"  平均高度: {sum(non_boundary)/len(non_boundary):.4f}")


if __name__ == "__main__":
    # 测试小网格（5x5）
    print("=" * 70)
    print("小网格示例 (5x5) - 蒙特卡洛 vs 迭代方法")
    print("=" * 70)
    
    boundary_small = {}
    grid_size_small = (5, 5)
    
    for i in range(grid_size_small[0]):
        boundary_small[(i, 0)] = 1.0
        boundary_small[(i, grid_size_small[1]-1)] = 1.0
    for j in range(grid_size_small[1]):
        boundary_small[(0, j)] = 1.0
        boundary_small[(grid_size_small[0]-1, j)] = 1.0
    
    mc_small = SoapBubble(grid_size_small, boundary_small)
    mc_small.learn(num_episodes=5000)
    print("\n蒙特卡洛方法:")
    mc_small.print_surface()
    
    iter_small = SoapBubble(grid_size_small, boundary_small)
    iter_small.iterative_solve(iterations=1000)
    print("\n迭代方法:")
    iter_small.print_surface()
    
    # 测试大网格（50x50）- 只估计几个关键点
    print("\n" + "=" * 70)
    print("大网格示例 (50x50) - 只估计中心及几个关键点")
    print("=" * 70)
    
    boundary_large = {}
    grid_size_large = (50, 50)
    
    # 设置边界：上下边界为1.0，左右边界为0.5
    for i in range(grid_size_large[0]):
        boundary_large[(i, 0)] = 0.5
        boundary_large[(i, grid_size_large[1]-1)] = 0.5
    for j in range(grid_size_large[1]):
        boundary_large[(0, j)] = 1.0
        boundary_large[(grid_size_large[0]-1, j)] = 1.0
    
    # 蒙特卡洛方法：只估计几个关键点
    print("\n蒙特卡洛方法（只计算5个关键点）:")
    mc_large = SoapBubble(grid_size_large, boundary_large)
    
    # 选择要估计的点：中心点和四个象限中心
    key_points = [
        (25, 25),  # 中心
        (12, 12),  # 左上象限中心
        (12, 37),  # 右上象限中心
        (37, 12),  # 左下象限中心
        (37, 37),  # 右下象限中心
    ]
    
    print(f"估计点: {key_points}")
    mc_large.estimate_points(key_points, num_episodes=5000)
    
    for point in key_points:
        print(f"  点 {point}: 高度 = {mc_large.V[point]:.4f}")
    
    # 迭代方法：计算所有点（但速度快）
    print("\n迭代方法（计算所有点）:")
    iter_large = SoapBubble(grid_size_large, boundary_large)
    import time
    start_time = time.time()
    iter_large.iterative_solve(iterations=1000)
    elapsed_time = time.time() - start_time
    print(f"耗时: {elapsed_time:.2f} 秒")
    iter_large.print_surface()
    
    # 比较关键点的结果
    print("\n两种方法在关键点的对比:")
    print(f"{'点':<15} {'蒙特卡洛':<12} {'迭代方法':<12} {'差异':<12}")
    print("-" * 55)
    for point in key_points:
        mc_val = mc_large.V[point]
        iter_val = iter_large.V[point]
        diff = abs(mc_val - iter_val)
        print(f"{str(point):<15} {mc_val:<12.4f} {iter_val:<12.4f} {diff:<12.6f}")
    
    # 演示：如果你只关心一个点，蒙特卡洛会非常快
    print("\n" + "=" * 70)
    print("演示：单点估计的效率优势")
    print("=" * 70)
    
    single_point = (25, 25)
    mc_single = SoapBubble(grid_size_large, boundary_large)
    
    start_time = time.time()
    estimated_height = mc_single.estimate_point(single_point, num_episodes=5000)
    mc_time = time.time() - start_time
    
    print(f"\n蒙特卡洛估计点 {single_point}:")
    print(f"  估计高度: {estimated_height:.4f}")
    print(f"  耗时: {mc_time:.4f} 秒")
    
    print(f"\n迭代方法计算点 {single_point}:")
    print(f"  高度: {iter_large.V[single_point]:.4f}")
    print(f"  耗时: {elapsed_time:.2f} 秒（需要计算所有点）")
    print(f"\n速度比: 迭代方法耗时是蒙特卡洛的 {elapsed_time/mc_time:.1f} 倍")