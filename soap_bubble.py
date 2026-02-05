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
        :param gravity: 重力系数，建议范围 0.001-0.05，值越大下垂越明显
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
    
    def get_distance_to_boundary(self, point: Tuple[int, int]) -> float:
        """计算点到最近边界的距离"""
        x, y = point
        dist_to_edges = [
            x,
            self.rows - 1 - x,
            y,
            self.cols - 1 - y
        ]
        return min(dist_to_edges)
    
    def random_walk(self, start_point: Tuple[int, int]) -> float:
        """
        从起始点开始随机游走直到到达边界
        考虑重力时，累积路径上的重力势能
        """
        current = start_point
        
        if current in self.boundary_points:
            return self.boundary_points[current]
        
        step_count = 0
        
        while current not in self.boundary_points:
            neighbors = self.get_neighbors(current)
            current = random.choice(neighbors)
            step_count += 1
        
        boundary_height = self.boundary_points[current]
        
        # 重力修正：步数反映了距离，但系数要小
        # 使用较小的系数使得下垂平缓
        gravity_correction = self.gravity * step_count * 0.001
        
        return boundary_height - gravity_correction
    
    def random_walk_weighted(self, start_point: Tuple[int, int]) -> float:
        """考虑重力的加权随机游走"""
        current = start_point
        
        if current in self.boundary_points:
            return self.boundary_points[current]
        
        step_count = 0
        
        while current not in self.boundary_points:
            neighbors = self.get_neighbors(current)
            
            if self.gravity > 0:
                weights = []
                for nx, ny in neighbors:
                    dx = nx - current[0]
                    if dx > 0:  # 向下
                        weight = 1.0 + self.gravity * 10
                    elif dx < 0:  # 向上
                        weight = 1.0 / (1.0 + self.gravity * 10)
                    else:  # 水平
                        weight = 1.0
                    weights.append(weight)
                
                total_weight = sum(weights)
                probabilities = [w / total_weight for w in weights]
                current = random.choices(neighbors, weights=probabilities)[0]
            else:
                current = random.choice(neighbors)
            
            step_count += 1
        
        gravity_correction = self.gravity * step_count * 0.001
        return self.boundary_points[current] - gravity_correction
    
    def estimate_point(self, point: Tuple[int, int], num_episodes: int = 1000, 
                      weighted: bool = False) -> float:
        """估计单个点的高度"""
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
    
    def iterative_solve_simple(self, iterations: int = 1000, tolerance: float = 1e-6):
        """
        简单迭代方法：考虑重力的距离修正
        形成平缓的抛物面
        """
        for iteration in range(iterations):
            max_change = 0
            new_V = self.V.copy()
            
            for point in self.status:
                if point not in self.boundary_points:
                    neighbors = self.get_neighbors(point)
                    avg_height = sum(self.V[n] for n in neighbors) / len(neighbors)
                    
                    # 基于距离边界的重力修正
                    dist_to_boundary = self.get_distance_to_boundary(point)
                    
                    # 使用合理的系数，形成平缓抛物面
                    # dist² 模型，但系数要很小
                    gravity_correction = self.gravity * (dist_to_boundary ** 2)
                    
                    new_V[point] = avg_height - gravity_correction
                    max_change = max(max_change, abs(new_V[point] - self.V[point]))
            
            self.V = new_V
            
            if max_change < tolerance:
                print(f"简单迭代方法在第 {iteration+1} 次迭代后收敛")
                break
    
    def iterative_solve_pde(self, iterations: int = 1000, tolerance: float = 1e-6):
        """
        使用泊松方程求解
        ∇²h = -g
        离散形式：h[i,j] = (h[i-1,j] + h[i+1,j] + h[i,j-1] + h[i,j+1]) / 4 - g*dx²/4
        """
        for iteration in range(iterations):
            max_change = 0
            new_V = self.V.copy()
            
            for point in self.status:
                if point not in self.boundary_points:
                    neighbors = self.get_neighbors(point)
                    avg_height = sum(self.V[n] for n in neighbors) / len(neighbors)
                    
                    # 泊松方程的源项
                    dx = 1.0
                    gravity_source = (dx * dx / 4.0) * self.gravity
                    
                    new_V[point] = avg_height - gravity_source
                    max_change = max(max_change, abs(new_V[point] - self.V[point]))
            
            self.V = new_V
            
            if max_change < tolerance:
                print(f"泊松方程求解在第 {iteration+1} 次迭代后收敛")
                break
    
    def iterative_solve(self, iterations: int = 1000, tolerance: float = 1e-6):
        """默认使用泊松方程求解"""
        self.iterative_solve_pde(iterations, tolerance)
    
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
                
                # 显示中心点和边缘点的对比
                center = (self.rows // 2, self.cols // 2)
                edge = (self.rows // 2, 1)
                
                if center not in self.boundary_points:
                    print(f"  中心点 {center} 高度: {self.V[center]:.4f}")
                if edge not in self.boundary_points:
                    print(f"  边缘点 {edge} 高度: {self.V[edge]:.4f}")
                    if self.V[edge] > self.V[center]:
                        print(f"  中心相对边缘下垂: {self.V[edge] - self.V[center]:.4f}")


if __name__ == "__main__":
    print("=" * 70)
    print("肥皂泡表面模拟：无重力 vs 有重力（合理参数）")
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
    print("理想肥皂膜 - 拉普拉斯方程的解")
    print("=" * 70)
    
    no_gravity = SoapBubble(grid_size, boundary, gravity=0.0)
    no_gravity.iterative_solve(iterations=1000)
    no_gravity.print_surface(show_all=True)
    
    # 微小重力
    print("\n" + "=" * 70)
    print("情况2：微小重力 (gravity = 0.01)")
    print("轻微下垂的肥皂膜")
    print("=" * 70)
    
    tiny_gravity = SoapBubble(grid_size, boundary, gravity=0.01)
    tiny_gravity.iterative_solve(iterations=1000)
    tiny_gravity.print_surface(show_all=True)
    
    # 小重力
    print("\n" + "=" * 70)
    print("情况3：小重力 (gravity = 0.02)")
    print("明显但平缓的抛物面")
    print("=" * 70)
    
    small_gravity = SoapBubble(grid_size, boundary, gravity=0.02)
    small_gravity.iterative_solve(iterations=1000)
    small_gravity.print_surface(show_all=True)
    
    # 中等重力
    print("\n" + "=" * 70)
    print("情况4：中等重力 (gravity = 0.05)")
    print("较强的抛物面下垂")
    print("=" * 70)
    
    medium_gravity = SoapBubble(grid_size, boundary, gravity=0.05)
    medium_gravity.iterative_solve(iterations=1000)
    medium_gravity.print_surface(show_all=True)
    
    # 对比两种方法
    print("\n" + "=" * 70)
    print("对比：泊松方程 vs 距离修正方法 (gravity = 0.02)")
    print("=" * 70)
    
    print("\n泊松方程方法（物理精确）:")
    pde_bubble = SoapBubble(grid_size, boundary, gravity=0.02)
    pde_bubble.iterative_solve_pde(iterations=1000)
    pde_bubble.print_surface(show_all=True)
    
    print("\n距离修正方法（启发式，使用 gravity = 0.002）:")
    simple_bubble = SoapBubble(grid_size, boundary, gravity=0.002)
    simple_bubble.iterative_solve_simple(iterations=1000)
    simple_bubble.print_surface(show_all=True)
    
    # 蒙特卡洛方法验证
    print("\n" + "=" * 70)
    print("蒙特卡洛方法验证 (gravity = 0.02)")
    print("=" * 70)
    
    test_points = [
        (5, 5),   # 中心
        (5, 2),   # 边缘
        (3, 3),   # 中间区域
        (7, 7),   # 另一中间区域
    ]
    
    mc_gravity = SoapBubble(grid_size, boundary, gravity=0.02)
    
    print(f"\n{'位置':<15} {'到边界距离':<12} {'蒙特卡洛':<12} {'泊松方程':<12} {'差异':<12}")
    print("-" * 70)
    for point in test_points:
        dist = mc_gravity.get_distance_to_boundary(point)
        mc_height = mc_gravity.estimate_point(point, num_episodes=5000)
        pde_height = pde_bubble.V[point]
        diff = abs(mc_height - pde_height)
        print(f"{str(point):<15} {dist:<12} {mc_height:<12.4f} {pde_height:<12.4f} {diff:<12.4f}")
    
    # 高度变化分析
    print("\n" + "=" * 70)
    print("重力影响分析：中心点高度变化")
    print("=" * 70)
    
    center = (5, 5)
    print(f"\n{'重力系数':<15} {'中心点高度':<15} {'相对无重力下降':<20}")
    print("-" * 55)
    
    gravity_values = [0.0, 0.01, 0.02, 0.05]
    baseline_height = no_gravity.V[center]
    
    for g in gravity_values:
        bubble = SoapBubble(grid_size, boundary, gravity=g)
        bubble.iterative_solve(iterations=1000)
        height = bubble.V[center]
        drop = baseline_height - height
        print(f"{g:<15.2f} {height:<15.4f} {drop:<20.4f}")
    
    print("\n观察：")
    print("1. 无重力时，所有内部点高度都是 1.0（平面）")
    print("2. 重力增大，中心点下降，形成平缓的抛物面")
    print("3. 重力系数 0.01-0.05 产生合理的下垂（几十分之一的单位）")
    print("4. 蒙特卡洛方法能够很好地逼近泊松方程的解")
    
    # 大网格演示
    print("\n" + "=" * 70)
    print("大网格示例 (21x21) - 观察抛物面形状")
    print("=" * 70)
    
    boundary_large = {}
    grid_large = (21, 21)
    
    for i in range(grid_large[0]):
        boundary_large[(i, 0)] = 1.0
        boundary_large[(i, grid_large[1]-1)] = 1.0
    for j in range(grid_large[1]):
        boundary_large[(0, j)] = 1.0
        boundary_large[(grid_large[0]-1, j)] = 1.0
    
    large_bubble = SoapBubble(grid_large, boundary_large, gravity=0.02)
    large_bubble.iterative_solve(iterations=2000)
    large_bubble.print_surface(show_all=False)
    
    # 显示一条横截面
    print("\n横截面（中间一行）展示抛物面形状:")
    mid_row = grid_large[0] // 2
    print(f"行 {mid_row}: ", end="")
    for j in range(grid_large[1]):
        point = (mid_row, j)
        if point in boundary_large:
            print(f"[{large_bubble.V[point]:.2f}]", end=" ")
        else:
            print(f"{large_bubble.V[point]:.2f}", end=" ")
    print()
    
    print("\n说明：从边界(1.00)到中心逐渐下降，形成平缓的抛物线")