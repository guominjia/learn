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
    
    def get_distance_to_boundary(self, point: Tuple[int, int]) -> float:
        """
        计算点到最近边界的距离
        这个距离代表了该点"支撑"的膜的范围
        距离边界越远，承受的重力拉伸越大
        """
        x, y = point
        # 到四条边的距离
        dist_to_edges = [
            x,  # 到上边界
            self.rows - 1 - x,  # 到下边界
            y,  # 到左边界
            self.cols - 1 - y  # 到右边界
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
        
        # 累积重力影响：步数越多，说明离边界越远
        step_count = 0
        
        # 随机游走直到到达边界
        while current not in self.boundary_points:
            neighbors = self.get_neighbors(current)
            current = random.choice(neighbors)
            step_count += 1
        
        # 边界高度减去重力累积效应
        boundary_height = self.boundary_points[current]
        
        # 重力修正：步数越多（离边界越远），下垂越明显
        # 这符合物理直觉：中心区域路径最长，下垂最多
        gravity_correction = self.gravity * step_count * 0.01
        
        return boundary_height - gravity_correction
    
    def random_walk_weighted(self, start_point: Tuple[int, int]) -> float:
        """
        考虑重力的加权随机游走
        重力会影响游走的方向概率（更倾向于向下）
        """
        current = start_point
        
        if current in self.boundary_points:
            return self.boundary_points[current]
        
        step_count = 0
        
        while current not in self.boundary_points:
            neighbors = self.get_neighbors(current)
            
            if self.gravity > 0:
                # 计算每个邻居的权重
                weights = []
                for nx, ny in neighbors:
                    # 向下方向（行数增加）有更高权重
                    dx = nx - current[0]
                    if dx > 0:  # 向下
                        weight = 1.0 + self.gravity * 0.5
                    elif dx < 0:  # 向上
                        weight = 1.0 / (1.0 + self.gravity * 0.5)
                    else:  # 水平
                        weight = 1.0
                    weights.append(weight)
                
                # 加权随机选择
                total_weight = sum(weights)
                probabilities = [w / total_weight for w in weights]
                current = random.choices(neighbors, weights=probabilities)[0]
            else:
                current = random.choice(neighbors)
            
            step_count += 1
        
        # 使用步数来估计重力影响
        gravity_correction = self.gravity * step_count * 0.01
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
        距离边界越远的点，重力影响越大
        """
        for iteration in range(iterations):
            max_change = 0
            new_V = self.V.copy()
            
            for point in self.status:
                if point not in self.boundary_points:
                    neighbors = self.get_neighbors(point)
                    avg_height = sum(self.V[n] for n in neighbors) / len(neighbors)
                    
                    # 基于距离边界的重力修正
                    # 距离边界越远，下垂越多
                    dist_to_boundary = self.get_distance_to_boundary(point)
                    
                    # 重力修正：使用二次函数，中心区域下垂最多
                    # dist² 符合悬链线的物理模型
                    gravity_correction = self.gravity * (dist_to_boundary ** 2) * 0.01
                    
                    new_V[point] = avg_height - gravity_correction
                    max_change = max(max_change, abs(new_V[point] - self.V[point]))
            
            self.V = new_V
            
            if max_change < tolerance:
                print(f"简单迭代方法在第 {iteration+1} 次迭代后收敛")
                break
    
    def iterative_solve_pde(self, iterations: int = 1000, tolerance: float = 1e-6):
        """
        使用泊松方程求解（考虑重力的精确模型）
        ∇²h = -g （拉普拉斯算子 = 重力源项）
        
        离散形式：h[i,j] = (h[i-1,j] + h[i+1,j] + h[i,j-1] + h[i,j+1]) / 4 - g*dx²/4
        """
        for iteration in range(iterations):
            max_change = 0
            new_V = self.V.copy()
            
            for point in self.status:
                if point not in self.boundary_points:
                    neighbors = self.get_neighbors(point)
                    avg_height = sum(self.V[n] for n in neighbors) / len(neighbors)
                    
                    # 泊松方程的源项（重力）
                    # 这里重力是常数，对所有点影响相同
                    dx = 1.0  # 网格间距
                    gravity_source = (dx * dx / 4.0) * self.gravity
                    
                    # 注意：重力项对所有内部点都一样
                    # 但边界条件会导致中心点下垂更多
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
                edge = (self.rows // 2, 1)  # 靠近边缘的点
                
                if center not in self.boundary_points:
                    print(f"  中心点 {center} 高度: {self.V[center]:.4f}")
                if edge not in self.boundary_points:
                    print(f"  边缘点 {edge} 高度: {self.V[edge]:.4f}")
                    print(f"  中心相对边缘下垂: {self.V[edge] - self.V[center]:.4f}")


if __name__ == "__main__":
    # 测试：展示中心点确实受到最大重力影响
    print("=" * 70)
    print("对比实验：无重力 vs 有重力（修正版）")
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
    no_gravity.iterative_solve(iterations=1000)
    print("\n泊松方程求解（无重力）:")
    no_gravity.print_surface(show_all=True)
    
    # 有重力情况（小重力）
    print("\n" + "=" * 70)
    print("情况2：小重力 (gravity = 1.0)")
    print("=" * 70)
    
    small_gravity = SoapBubble(grid_size, boundary, gravity=1.0)
    small_gravity.iterative_solve(iterations=1000)
    print("\n泊松方程求解（小重力）:")
    small_gravity.print_surface(show_all=True)
    
    # 有重力情况（大重力）
    print("\n" + "=" * 70)
    print("情况3：大重力 (gravity = 5.0)")
    print("=" * 70)
    
    large_gravity = SoapBubble(grid_size, boundary, gravity=5.0)
    large_gravity.iterative_solve(iterations=1000)
    print("\n泊松方程求解（大重力）:")
    large_gravity.print_surface(show_all=True)
    
    # 对比简单方法（基于距离）
    print("\n" + "=" * 70)
    print("对比：泊松方程 vs 距离修正方法")
    print("=" * 70)
    
    print("\n泊松方程方法（物理精确）:")
    pde_bubble = SoapBubble(grid_size, boundary, gravity=2.0)
    pde_bubble.iterative_solve_pde(iterations=1000)
    pde_bubble.print_surface(show_all=True)
    
    print("\n距离修正方法（启发式）:")
    simple_bubble = SoapBubble(grid_size, boundary, gravity=2.0)
    simple_bubble.iterative_solve_simple(iterations=1000)
    simple_bubble.print_surface(show_all=True)
    
    # 蒙特卡洛方法：观察不同位置的点
    print("\n" + "=" * 70)
    print("蒙特卡洛方法：不同位置点的高度对比")
    print("=" * 70)
    
    test_points = [
        (5, 5),   # 中心
        (5, 2),   # 边缘
        (3, 3),   # 中间区域
        (7, 7),   # 另一中间区域
    ]
    
    print("\n有重力 (gravity = 2.0):")
    mc_gravity = SoapBubble(grid_size, boundary, gravity=2.0)
    
    print(f"\n{'位置':<15} {'到边界距离':<12} {'蒙特卡洛':<12} {'泊松方程':<12}")
    print("-" * 60)
    for point in test_points:
        dist = mc_gravity.get_distance_to_boundary(point)
        mc_height = mc_gravity.estimate_point(point, num_episodes=5000)
        pde_height = pde_bubble.V[point]
        print(f"{str(point):<15} {dist:<12} {mc_height:<12.4f} {pde_height:<12.4f}")
    
    print("\n观察：")
    print("1. 中心点 (5,5) 到边界距离最远，下垂最多")
    print("2. 边缘点 (5,2) 到边界距离近，下垂少")
    print("3. 蒙特卡洛的随机游走步数反映了距离，步数越多 -> 下垂越多")
    
    # 分析随机游走的步数分布
    print("\n" + "=" * 70)
    print("随机游走步数分析（体现距离效应）")
    print("=" * 70)
    
    def analyze_walk_steps(bubble, point, num_walks=1000):
        """分析从某点出发的随机游走平均步数"""
        step_counts = []
        for _ in range(num_walks):
            current = point
            steps = 0
            while current not in bubble.boundary_points:
                neighbors = bubble.get_neighbors(current)
                current = random.choice(neighbors)
                steps += 1
            step_counts.append(steps)
        return sum(step_counts) / len(step_counts)
    
    mc_test = SoapBubble(grid_size, boundary, gravity=0.0)
    
    print(f"\n{'位置':<15} {'到边界距离':<15} {'平均游走步数':<15}")
    print("-" * 50)
    for point in test_points:
        dist = mc_test.get_distance_to_boundary(point)
        avg_steps = analyze_walk_steps(mc_test, point, num_walks=1000)
        print(f"{str(point):<15} {dist:<15} {avg_steps:<15.2f}")
    
    print("\n结论：平均游走步数 ≈ 距离的平方关系")
    print("这就是为什么重力修正要用 dist² 或 step_count")