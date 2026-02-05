from typing import List, Tuple, Set
from collections import defaultdict
import random
import math

class SoapBubble:
    def __init__(self, grid_size: Tuple[int, int], boundary_points: dict = None, gravity: float = 0.0):
        """
        初始化肥皂泡网格
        :param grid_size: (rows, cols) 网格大小
        :param boundary_points: {(x, y): height} 边界点及其高度
        :param gravity: 重力系数，0表示无重力，>0表示有重力作用
        """
        self.rows, self.cols = grid_size
        self.boundary_points = boundary_points or {}
        self.gravity = gravity
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
                    self.V[point] = 0.0
    
    def get_neighbors(self, point: Tuple[int, int]) -> List[Tuple[int, int]]:
        """获取四个相邻网格点"""
        x, y = point
        neighbors = []
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.rows and 0 <= ny < self.cols:
                neighbors.append((nx, ny))
        
        return neighbors
    
    def random_walk(self, start_point: Tuple[int, int]) -> float:
        """
        从起始点开始随机游走直到到达边界
        考虑重力时，累积路径上的重力势能
        :param start_point: 起始点坐标
        :return: 到达边界点的高度（考虑重力修正）
        """
        current = start_point
        
        if current in self.boundary_points:
            return self.boundary_points[current]
        
        # 累积重力影响
        gravity_accumulation = 0.0
        step_count = 0
        
        # 随机游走直到到达边界
        while current not in self.boundary_points:
            neighbors = self.get_neighbors(current)
            current = random.choice(neighbors)
            
            # 累积重力效应（每走一步，受到向下的拉力）
            # 重力会使表面向下弯曲
            if self.gravity > 0:
                step_count += 1
                gravity_accumulation += self.gravity
        
        # 边界高度减去重力累积效应
        boundary_height = self.boundary_points[current]
        
        # 重力修正：路径越长，重力影响越大
        # 这模拟了表面在重力作用下向下弯曲
        return boundary_height - gravity_accumulation
    
    def random_walk_weighted(self, start_point: Tuple[int, int]) -> float:
        """
        考虑重力的加权随机游走
        重力会影响游走的方向概率（更倾向于向下）
        """
        current = start_point
        
        if current in self.boundary_points:
            return self.boundary_points[current]
        
        gravity_correction = 0.0
        
        while current not in self.boundary_points:
            neighbors = self.get_neighbors(current)
            
            if self.gravity > 0:
                # 计算每个邻居的权重
                # 向下的邻居有更高的概率被选中
                weights = []
                for nx, ny in neighbors:
                    # 向下方向（行数增加）有更高权重
                    dx = nx - current[0]
                    if dx > 0:  # 向下
                        weight = 1.0 + self.gravity
                    elif dx < 0:  # 向上
                        weight = 1.0 / (1.0 + self.gravity)
                    else:  # 水平
                        weight = 1.0
                    weights.append(weight)
                
                # 加权随机选择
                total_weight = sum(weights)
                probabilities = [w / total_weight for w in weights]
                current = random.choices(neighbors, weights=probabilities)[0]
                
                # 累积重力下拉效应
                gravity_correction += self.gravity * 0.01
            else:
                current = random.choice(neighbors)
        
        return self.boundary_points[current] - gravity_correction
    
    def estimate_point(self, point: Tuple[int, int], num_episodes: int = 1000, 
                      weighted: bool = False) -> float:
        """
        仅估计单个点的高度（按需计算）
        :param point: 目标点
        :param num_episodes: 随机游走次数
        :param weighted: 是否使用加权随机游走
        :return: 估计的高度值
        """
        if point in self.boundary_points:
            return self.boundary_points[point]
        
        returns = []
        for _ in range(num_episodes):
            if weighted:
                boundary_height = self.random_walk_weighted(point)
            else:
                boundary_height = self.random_walk(point)
            returns.append(boundary_height)
        
        estimated_height = sum(returns) / len(returns)
        self.V[point] = estimated_height
        return estimated_height
    
    def estimate_points(self, points: List[Tuple[int, int]], num_episodes: int = 1000,
                       weighted: bool = False):
        """估计指定点的高度"""
        for point in points:
            self.estimate_point(point, num_episodes, weighted)
    
    def learn(self, num_episodes: int = 1000, sample_points: List[Tuple[int, int]] = None,
             weighted: bool = False):
        """使用蒙特卡洛方法估计网格点的高度"""
        if sample_points is None:
            sample_points = [p for p in self.status if p not in self.boundary_points]
        
        returns = defaultdict(list)
        
        for point in sample_points:
            if point not in self.boundary_points:
                for _ in range(num_episodes):
                    if weighted:
                        boundary_height = self.random_walk_weighted(point)
                    else:
                        boundary_height = self.random_walk(point)
                    returns[point].append(boundary_height)
        
        for point in returns:
            self.V[point] = sum(returns[point]) / len(returns[point])
    
    def iterative_solve(self, iterations: int = 1000, tolerance: float = 1e-6):
        """
        使用迭代方法求解，考虑重力
        重力会使每个点相对于邻居平均值向下偏移
        """
        for iteration in range(iterations):
            max_change = 0
            new_V = self.V.copy()
            
            for point in self.status:
                if point not in self.boundary_points:
                    neighbors = self.get_neighbors(point)
                    avg_height = sum(self.V[n] for n in neighbors) / len(neighbors)
                    
                    # 考虑重力：表面会向下弯曲
                    # 重力修正项：距离边界越远，下垂越多
                    x, y = point
                    # 简化模型：中心区域下垂更多
                    center_x, center_y = self.rows / 2, self.cols / 2
                    distance_from_center = math.sqrt((x - center_x)**2 + (y - center_y)**2)
                    gravity_correction = self.gravity * distance_from_center * 0.001
                    
                    new_V[point] = avg_height - gravity_correction
                    max_change = max(max_change, abs(new_V[point] - self.V[point]))
            
            self.V = new_V
            
            if max_change < tolerance:
                print(f"迭代方法在第 {iteration+1} 次迭代后收敛")
                break
    
    def iterative_solve_pde(self, iterations: int = 1000, tolerance: float = 1e-6):
        """
        使用泊松方程求解（考虑重力的更精确模型）
        ∇²h = -g （拉普拉斯算子 = 重力源项）
        """
        for iteration in range(iterations):
            max_change = 0
            new_V = self.V.copy()
            
            for point in self.status:
                if point not in self.boundary_points:
                    neighbors = self.get_neighbors(point)
                    avg_height = sum(self.V[n] for n in neighbors) / len(neighbors)
                    
                    # 泊松方程的源项（重力）
                    # h_new = avg(neighbors) + (dx²/4) * g
                    dx = 1.0  # 网格间距
                    gravity_source = (dx * dx / 4.0) * self.gravity
                    
                    new_V[point] = avg_height - gravity_source
                    max_change = max(max_change, abs(new_V[point] - self.V[point]))
            
            self.V = new_V
            
            if max_change < tolerance:
                print(f"泊松方程求解在第 {iteration+1} 次迭代后收敛")
                break
    
    def print_surface(self, show_all: bool = False):
        """打印表面高度"""
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
            print(f"\n网格大小: {self.rows}x{self.cols}")
            non_boundary = [self.V[p] for p in self.status if p not in self.boundary_points]
            if non_boundary:
                print(f"  非边界点数量: {len(non_boundary)}")
                print(f"  高度范围: [{min(non_boundary):.4f}, {max(non_boundary):.4f}]")
                print(f"  平均高度: {sum(non_boundary)/len(non_boundary):.4f}")
                
                # 显示中心点高度（最能体现重力影响）
                center = (self.rows // 2, self.cols // 2)
                if center not in self.boundary_points:
                    print(f"  中心点 {center} 高度: {self.V[center]:.4f}")


if __name__ == "__main__":
    # 测试：无重力 vs 有重力
    print("=" * 70)
    print("对比实验：无重力 vs 有重力的肥皂泡形状")
    print("=" * 70)
    
    boundary = {}
    grid_size = (11, 11)
    
    # 设置边界：所有边界点高度为1.0
    for i in range(grid_size[0]):
        boundary[(i, 0)] = 1.0
        boundary[(i, grid_size[1]-1)] = 1.0
    for j in range(grid_size[1]):
        boundary[(0, j)] = 1.0
        boundary[(grid_size[0]-1, j)] = 1.0
    
    # 无重力情况
    print("\n" + "=" * 70)
    print("情况1：无重力 (gravity = 0)")
    print("=" * 70)
    
    no_gravity = SoapBubble(grid_size, boundary, gravity=0.0)
    no_gravity.iterative_solve_pde(iterations=1000)
    print("\n迭代方法（无重力）:")
    no_gravity.print_surface(show_all=True)
    
    # 有重力情况（小重力）
    print("\n" + "=" * 70)
    print("情况2：小重力 (gravity = 0.5)")
    print("=" * 70)
    
    small_gravity = SoapBubble(grid_size, boundary, gravity=0.5)
    small_gravity.iterative_solve_pde(iterations=1000)
    print("\n迭代方法（小重力）:")
    small_gravity.print_surface(show_all=True)
    
    # 有重力情况（大重力）
    print("\n" + "=" * 70)
    print("情况3：大重力 (gravity = 2.0)")
    print("=" * 70)
    
    large_gravity = SoapBubble(grid_size, boundary, gravity=2.0)
    large_gravity.iterative_solve_pde(iterations=1000)
    print("\n迭代方法（大重力）:")
    large_gravity.print_surface(show_all=True)
    
    # 蒙特卡洛方法对比
    print("\n" + "=" * 70)
    print("蒙特卡洛方法：估计中心点在不同重力下的高度")
    print("=" * 70)
    
    center_point = (5, 5)
    
    print("\n无重力:")
    mc_no_g = SoapBubble(grid_size, boundary, gravity=0.0)
    h_no_g = mc_no_g.estimate_point(center_point, num_episodes=5000, weighted=False)
    print(f"  蒙特卡洛估计: {h_no_g:.4f}")
    print(f"  迭代方法结果: {no_gravity.V[center_point]:.4f}")
    
    print("\n小重力 (0.5):")
    mc_small_g = SoapBubble(grid_size, boundary, gravity=0.5)
    h_small_g = mc_small_g.estimate_point(center_point, num_episodes=5000, weighted=False)
    print(f"  蒙特卡洛估计: {h_small_g:.4f}")
    print(f"  迭代方法结果: {small_gravity.V[center_point]:.4f}")
    
    print("\n大重力 (2.0):")
    mc_large_g = SoapBubble(grid_size, boundary, gravity=2.0)
    h_large_g = mc_large_g.estimate_point(center_point, num_episodes=5000, weighted=False)
    print(f"  蒙特卡洛估计: {h_large_g:.4f}")
    print(f"  迭代方法结果: {large_gravity.V[center_point]:.4f}")
    
    # 加权随机游走对比
    print("\n" + "=" * 70)
    print("加权随机游走 vs 普通随机游走（有重力情况）")
    print("=" * 70)
    
    print("\n普通随机游走:")
    mc_normal = SoapBubble(grid_size, boundary, gravity=1.0)
    h_normal = mc_normal.estimate_point(center_point, num_episodes=5000, weighted=False)
    print(f"  中心点高度: {h_normal:.4f}")
    
    print("\n加权随机游走（考虑重力方向）:")
    mc_weighted = SoapBubble(grid_size, boundary, gravity=1.0)
    h_weighted = mc_weighted.estimate_point(center_point, num_episodes=5000, weighted=True)
    print(f"  中心点高度: {h_weighted:.4f}")
    
    # 可视化重力影响
    print("\n" + "=" * 70)
    print("重力影响分析")
    print("=" * 70)
    print(f"\n中心点高度变化:")
    print(f"  无重力: {no_gravity.V[center_point]:.4f}")
    print(f"  小重力: {small_gravity.V[center_point]:.4f} (下降 {no_gravity.V[center_point] - small_gravity.V[center_point]:.4f})")
    print(f"  大重力: {large_gravity.V[center_point]:.4f} (下降 {no_gravity.V[center_point] - large_gravity.V[center_point]:.4f})")
    print(f"\n观察：重力越大，中心点越低，表面越弯曲（像一个下垂的薄膜）")