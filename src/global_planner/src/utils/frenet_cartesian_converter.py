# Frenet-Cartesian Converter
import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.spatial.distance import cdist

"""
waypoints: list of (x, y) coordinates of the path
"""
class FrenetCartesianConverter:
    def __init__(self, waypoints):
        self.x_spline = None
        self.y_spline = None
        self._fit_cubic_spline(waypoints)

    def get_frenet(self, cartesian_pose):
        """
        INPUT:
        cartesian_pose: [x(m), y(m), yaw(rad)]
        RETURNS:
        frenet_pose: [s(m), d(m), alpha(rad)]
        """
        x, y, yaw = cartesian_pose
        
        # Find closest point along the path
        num_points = 1000  # Number of points to sample along the path for finding the closest point
        # s_vals = np.linspace(0, self.x_spline.x[-1], num_points)
        s_vals = np.linspace(0, self.x_spline.t[-1], num_points)
        path_points = np.column_stack((self.x_spline(s_vals), self.y_spline(s_vals)))
        distances = cdist(path_points, [[x, y]])
        closest_index = np.argmin(distances)
        closest_point = path_points[closest_index]
        closest_s = s_vals[closest_index]
        
        # Calculate the d coordinate (perpendicular distance to the path)
        d = np.linalg.norm([x - closest_point[0], y - closest_point[1]])
        
        # Determine the sign of d (left or right of the path)
        path_dx = self.x_spline.derivative()(closest_s)
        path_dy = self.y_spline.derivative()(closest_s)
        normal = np.array([-path_dy, path_dx])
        point_vector = np.array([x - closest_point[0], y - closest_point[1]])
        d *= np.sign(np.dot(point_vector, normal))  # Adjust sign based on the side of the path
        
        # Calculate the alpha coordinate (angle difference)
        alpha = self._get_frenet_orientation(closest_s, yaw)
        
        return [closest_s, d, alpha]
    
    def get_cartesian(self, frenet_pose):
        """
        INPUT:
        frenet_pose: [s(m), d(m), alpha(rad)]
        RETURNS:
        cartesian_pose: [x(m), y(m), yaw(rad)]
        """
        s, d, alpha = frenet_pose
        
        # Calculate the Cartesian position of the point on the path corresponding to Frenet s
        x_path = self.x_spline(s)
        y_path = self.y_spline(s)
        
        # Get the path's direction (tangent) at that point
        dx = self.x_spline.derivative()(s)
        dy = self.y_spline.derivative()(s)
        path_yaw = np.arctan2(dy, dx)
        
        # Calculate the perpendicular direction (normal to the path)
        norm_direction = np.array([-dy, dx]) / (np.sqrt(dx**2 + dy**2) + 1e-6)
        
        # Calculate the Cartesian position offset by Frenet d
        x = x_path + d * norm_direction[0]
        y = y_path + d * norm_direction[1]
        
        # Convert Frenet orientation (alpha) to Cartesian yaw
        # Alpha is the angle between the vehicle direction and the path tangent
        # In Cartesian, yaw = path's direction (path_yaw) + alpha
        yaw = path_yaw + alpha
        
        return [x, y, yaw]

    def _get_frenet_orientation(self, s, yaw):
        dx = self.x_spline.derivative()
        dy = self.y_spline.derivative()
        path_angle = np.arctan2(dy(s), dx(s))
        alpha = yaw - path_angle
        if alpha > np.pi:
            alpha -= 2*np.pi
        elif alpha < -np.pi:
            alpha += 2*np.pi
        return alpha
    

    
    def _fit_cubic_spline(self, waypoints):
        # sparsify the waypoints
        waypoints = waypoints[::5]
        x = [point[0] for point in waypoints]
        y = [point[1] for point in waypoints]
        # print("x:", x)
        # print("y:", y)
        dx = np.diff(x)
        dy = np.diff(y)
        s = np.zeros(len(x))
        s[1:] = np.cumsum(np.sqrt(dx**2+dy**2))
        # self.x_spline = CubicSpline(s, x)
        # self.y_spline = CubicSpline(s, y)
        print("length of path s:", s[-1])
        self.x_spline = make_interp_spline(s, x, k=3, bc_type='clamped')
        self.y_spline = make_interp_spline(s, y, k=3, bc_type='clamped')
    # #converting a list of x,y waypoints to s-d cordiantes
    # def _get_s_d_cordinates(waypoints):
    #     x = [point[0] for point in waypoints]
    #     y = [point[1] for point in waypoints]
    #     dx = np.diff(x)
    #     dy = np.diff(y)
    #     s = np.zeros(len(x))
    #     s[1:] = np.cumsum(np.sqrt(dx**2+dy**2))
    #     print(s)
    #     return s, x, y



if __name__ == "__main__":
    x = [0, 0, 0, 0, 0, 0, 19, 29, 38, 9, 1, -39, 3, 0, 50]
    y = [0, 10, 20, 30, 40, 50, 50, 50, 50, 50, 50, 50, 50, 10, 19]
    # random waypoints
    # x = np.random.randint(-50, 50, 20)
    # y = np.random.randint(-50, 50, 20)
    # print("x:", x)
    # print("y:", y)
    # x = [-13, 7, -49, -8, -9, -34, 1, 44, 22, 41, -26, -35, 13, 8, -10, -6, 37, 4,
    #     18, -34, 28, 3, -13, -14, -23, 46, 5, 4, -28, 15, -1, -7, 43, 47, -18, -27,
    #     -35, -50, -34, 18, 17, 30, -26, -34, -24, 36, 43, 12, 6, -29]

    # y = [-4, -5, 44, 3, 0, -45, 4, -45, -19, 18, -42, 33, 39, 0, 0, -32, 7, 27, 49, -42,
    #     -32, 44, 37, -38, 13, -33, 11, -13, -15, 17, -14, -33, -23, 4, 18, 29, 49, 16,
    #     13, 25, 34, -6, 11, 32, -17, 23, -38, 39, -5, 29]


    waypoints = list(zip(x, y))
    f2c = FrenetCartesianConverter(waypoints)
    print("x_spline:", f2c.x_spline)
    print("x_spline coefficients:", f2c.x_spline.c)
    print("x_spline knots:", f2c.x_spline.t)
    # print("x_spline knots", f2c.x_spline.get_knots())
    cartesian_pose = [-46, 50, 0]
    frenet_pose = f2c.get_frenet(cartesian_pose)
    print("Frenet pose:", frenet_pose)
    cartesian_pose = f2c.get_cartesian(frenet_pose)
    print("Cartesian pose:", cartesian_pose)





