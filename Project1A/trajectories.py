#!/usr/bin/env/python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse

"""
Set of classes for defining SE(3) trajectories for the end effector of a robot 
manipulator
"""
# points for the trajectories
goal_points = np.array([[5, 0, 0], [10, 0, 0], [3, 3, -2], [7, -1, 1], [5, 0, 0]])

class Trajectory:

    def __init__(self, total_time):
        """
        Parameters
        ----------
        total_time : float
        	desired duration of the trajectory in seconds 
        """
        self.total_time = total_time

    def target_pose(self, time):
        """
        Returns where the arm end effector should be at time t, in the form of a 
        7D vector [x, y, z, qx, qy, qz, qw]. i.e. the first three entries are 
        the desired end-effector position, and the last four entries are the 
        desired end-effector orientation as a quaternion, all written in the 
        world frame.

        Hint: The end-effector pose with the gripper pointing down corresponds 
        to the quaternion [0, 1, 0, 0]. 

        Parameters
        ----------
        time : float        
    
        Returns
        -------
        7x' :obj:`numpy.ndarray`
            desired configuration in workspace coordinates of the end effector
        """
        pass

    def target_velocity(self, time):
        """
        Returns the end effector's desired body-frame velocity at time t as a 6D
        twist. Note that this needs to be a rigid-body velocity, i.e. a member 
        of se(3) expressed as a 6D vector.

        Parameters
        ----------
        time : float

        Returns
        -------
        6x' :obj:`numpy.ndarray`
            desired body-frame velocity of the end effector
        """
        pass

    def display_trajectory(self, num_waypoints=67, show_animation=False, save_animation=False):
        """
        Displays the evolution of the trajectory's position and body velocity.

        Parameters
        ----------
        num_waypoints : int
            number of waypoints in the trajectory
        show_animation : bool
            if True, displays the animated trajectory
        save_animatioon : bool
            if True, saves a gif of the animated trajectory
        """
        trajectory_name = self.__class__.__name__
        times = np.linspace(0, self.total_time, num=num_waypoints)
        target_positions = np.vstack([self.target_pose(t)[:3] for t in times])
        target_velocities = np.vstack([self.target_velocity(t)[:3] for t in times])
        
        fig = plt.figure(figsize=plt.figaspect(0.5))
        colormap = plt.cm.brg(np.fmod(np.linspace(0, 1, num=num_waypoints), 1))

        # Position plot
        ax0 = fig.add_subplot(1, 2, 1, projection='3d')
        pos_padding = [[-0.1, 0.1],
                        [-0.1, 0.1],
                        [-0.1, 0.1]]
        ax0.set_xlim3d([min(target_positions[:, 0]) + pos_padding[0][0], 
                        max(target_positions[:, 0]) + pos_padding[0][1]])
        ax0.set_xlabel('X')
        ax0.set_ylim3d([min(target_positions[:, 1]) + pos_padding[1][0], 
                        max(target_positions[:, 1]) + pos_padding[1][1]])
        ax0.set_ylabel('Y')
        ax0.set_zlim3d([min(target_positions[:, 2]) + pos_padding[2][0], 
                        max(target_positions[:, 2]) + pos_padding[2][1]])
        ax0.set_zlabel('Z')
        ax0.set_title("%s evolution of\nend-effector's position." % trajectory_name)
        line0 = ax0.scatter(target_positions[:, 0], 
                        target_positions[:, 1], 
                        target_positions[:, 2], 
                        c=colormap,
                        s=2)

        # Velocity plot
        ax1 = fig.add_subplot(1, 2, 2, projection='3d')
        vel_padding = [[-0.1, 0.1],
                        [-0.1, 0.1],
                        [-0.1, 0.1]]
        ax1.set_xlim3d([min(target_velocities[:, 0]) + vel_padding[0][0], 
                        max(target_velocities[:, 0]) + vel_padding[0][1]])
        ax1.set_xlabel('X')
        ax1.set_ylim3d([min(target_velocities[:, 1]) + vel_padding[1][0], 
                        max(target_velocities[:, 1]) + vel_padding[1][1]])
        ax1.set_ylabel('Y')
        ax1.set_zlim3d([min(target_velocities[:, 2]) + vel_padding[2][0], 
                        max(target_velocities[:, 2]) + vel_padding[2][1]])
        ax1.set_zlabel('Z')
        ax1.set_title("%s evolution of\nend-effector's translational body-frame velocity." % trajectory_name)
        line1 = ax1.scatter(target_velocities[:, 0], 
                        target_velocities[:, 1], 
                        target_velocities[:, 2], 
                        c=colormap,
                        s=2)

        if show_animation or save_animation:
            def func(num, line):
                line[0]._offsets3d = target_positions[:num].T
                line[0]._facecolors = colormap[:num]
                line[1]._offsets3d = target_velocities[:num].T
                line[1]._facecolors = colormap[:num]
                return line

            # Creating the Animation object
            line_ani = animation.FuncAnimation(fig, func, frames=num_waypoints, 
                                                          fargs=([line0, line1],), 
                                                          interval=max(1, int(1000 * self.total_time / (num_waypoints - 1))), 
                                                          blit=False)
        plt.show()
        if save_animation:
            line_ani.save('%s.gif' % trajectory_name, writer='pillow', fps=60)
            print("Saved animation to %s.gif" % trajectory_name)

class LinearTrajectory(Trajectory):

    def __init__(self):
        """
        Remember to call the constructor of Trajectory

        Parameters
        ----------
        ????? You're going to have to fill these in how you see fit
        """
        super().__init__(5) # total time
        self.p0 = goal_points[0]
        self.p1 = goal_points[1]
        pass
        # Trajectory.__init__(self, ...)

    def target_pose(self, time):
        """
        Returns where the arm end effector should be at time t, in the form of a 
        7D vector [x, y, z, qx, qy, qz, qw]. i.e. the first three entries are 
        the desired end-effector position, and the last four entries are the 
        desired end-effector orientation as a quaternion, all written in the 
        world frame.

        Hint: The end-effector pose with the gripper pointing down corresponds 
        to the quaternion [0, 1, 0, 0]. 

        Parameters
        ----------
        time : float        
    
        Returns
        -------
        7x' :obj:`numpy.ndarray`
            desired configuration in workspace coordinates of the end effector
        """
        t = time
        tl = self.total_time
        pose = np.array([0, 1, 0, 0])
        
        p0 = self.p0
        p1 = self.p1
        pd = p1-p0
        
        # Method 1: linear velocity profile
        '''
        if t <= tl/2:
            pt = np.array([p0[0] + 2 * pd[0] * (t/tl) ** 2, p0[1] + 2 * pd[1] * (t/tl) ** 2, p0[2] + 2 * pd[2] * (t/tl) ** 2])
        else:
            pt = np.array([p0[0] + pd[0] - 2 * pd[0] * (t/tl-1) ** 2, p0[1] + 2 * pd[1] * (t/tl-1) ** 2, p0[2] + pd[2] - 2 * pd[2] * (t/tl-1) ** 2])
        '''    
        # Method 2: sinusoidal velocity profile
        pt = np.array([p0[0] -0.5 * pd[0] * np.cos(np.pi/tl*t) + 0.5 * pd[0], p0[1] -0.5 * pd[1] * np.cos(np.pi/tl*t) + 0.5 * pd[1], p0[2] -0.5 * pd[2] * np.cos(np.pi/tl*t) + 0.5 * pd[2]])
            
        return np.concatenate((pt, pose))
        pass

    def target_velocity(self, time):
        """
        Returns the end effector's desired body-frame velocity at time t as a 6D
        twist. Note that this needs to be a rigid-body velocity, i.e. a member 
        of se(3) expressed as a 6D vector.

        Parameters
        ----------
        time : float

        Returns
        -------
        6x' :obj:`numpy.ndarray`
            desired body-frame velocity of the end effector
        """
        t = time
        tl = self.total_time
        p0 = self.p0
        p1 = self.p1
        pd = p1-p0
        
        # Method 1: linear velocity profile
        '''
        if t <= tl/2:
            vt = np.array([4 * pd[0] / tl ** 2 * t, 4 * pd[1] / tl ** 2 * t, 4 * pd[2] / tl ** 2 * t])
        else:
            vt = np.array([-4 * pd[0] / tl ** 2 * (t-tl), -4 * pd[1] / tl ** 2 * (t-tl), -4 * pd[2] / tl ** 2 * (t-tl)])
        '''
        # Method 2: sinusoidal velocity profile
        vt = np.array([np.pi * pd[0] / (2*tl) * np.sin(np.pi/tl*t), np.pi * pd[1] / (2*tl) * np.sin(np.pi/tl*t), np.pi * pd[2] / (2*tl) * np.sin(np.pi/tl*t)])

        return np.concatenate((vt, np.array([0, 0, 0])))
        pass

class CircularTrajectory(Trajectory):

    def __init__(self, center_position = np.array([0, 10, 0]), radius = 3, total_time = 5):
        """
        Remember to call the constructor of Trajectory

        Parameters
        ----------
        ????? You're going to have to fill these in how you see fit
        """
        self.center_position = center_position
        self.radius = radius
        self.total_time = total_time
        pass
        # Trajectory.__init__(self, ...)

    def target_pose(self, time):
        """
        Returns where the arm end effector should be at time t, in the form of a 
        7D vector [x, y, z, qx, qy, qz, qw]. i.e. the first three entries are 
        the desired end-effector position, and the last four entries are the 
        desired end-effector orientation as a quaternion, all written in the 
        world frame.

        Hint: The end-effector pose with the gripper pointing down corresponds 
        to the quaternion [0, 1, 0, 0]. 

        Parameters
        ----------
        time : float        
    
        Returns
        -------
        7x' :obj:`numpy.ndarray`
            desired configuration in workspace coordinates of the end effector
        """
        t = time
        tl = self.total_time
        r = self.radius
        pose = np.array([0, 1, 0, 0])
        c = self.center_position
        omega = np.pi/tl
        theta = np.pi - np.pi * np.cos(omega*t)
        
        pt = np.array([c[0] + r*np.cos(theta), c[1] + r*np.sin(theta), c[2]])
        return np.concatenate((pt, pose))
        pass

    def target_velocity(self, time):
        """
        Returns the end effector's desired body-frame velocity at time t as a 6D
        twist. Note that this needs to be a rigid-body velocity, i.e. a member 
        of se(3) expressed as a 6D vector.

        Parameters
        ----------
        time : float

        Returns
        -------
        6x' :obj:`numpy.ndarray`
            desired body-frame velocity of the end effector
        """
        t = time
        tl = self.total_time
        r = self.radius
        omega = np.pi/tl
        theta = np.pi - np.pi * np.cos(omega*t)
        
        vt = np.array([-np.pi * r * omega * np.sin(omega*t) * np.sin(theta),-np.pi * r * omega * np.sin(omega*t) * np.cos(theta), 0])
        
        return np.concatenate((vt, np.array([0, 0, 0])))
        pass

class PolygonalTrajectory(Trajectory):
    def __init__(self, points = goal_points, total_time = 15):
        """
        Remember to call the constructor of Trajectory.
        You may wish to reuse other trajectories previously defined in this file.

        Parameters
        ----------
        ????? You're going to have to fill these in how you see fit

        """
        self.linear = LinearTrajectory()
        self.points = points
        self.total_time = total_time
        self.n = len(points) # number of points
        self.t_chunk = np.zeros(self.n)
        self.l_chunk = np.zeros(self.n-1)
        for i in range(0, self.n - 1):
            self.l_chunk[i] = np.linalg.norm(self.points[i+1] - self.points[i])
        
        for i in range(0, self.n - 1):
            self.t_chunk[i+1] = self.t_chunk[i] + self.l_chunk[i] / np.sum(self.l_chunk) * self.total_time
        self.t_chunk[self.n-1] = self.total_time
        pass
        # Trajectory.__init__(self, total_time)

    def target_pose(self, time):
        """
        Returns where the arm end effector should be at time t, in the form of a 
        7D vector [x, y, z, qx, qy, qz, qw]. i.e. the first three entries are 
        the desired end-effector position, and the last four entries are the 
        desired end-effector orientation as a quaternion, all written in the 
        world frame.

        Hint: The end-effector pose with the gripper pointing down corresponds 
        to the quaternion [0, 1, 0, 0]. 

        Parameters
        ----------
        time : float        
    
        Returns
        -------
        7x' :obj:`numpy.ndarray`
            desired configuration in workspace coordinates of the end effector
        """
        t = time
        for i in range(0, self.n - 1):
            if t >= self.t_chunk[i] and t <= self.t_chunk[i+1]:
                self.linear.total_time = self.t_chunk[i+1] - self.t_chunk[i]
                self.linear.p0 = self.points[i]
                self.linear.p1 = self.points[i+1]
                return self.linear.target_pose(t-self.t_chunk[i])
        pass
        
    def target_velocity(self, time):
        """
        Returns the end effector's desired body-frame velocity at time t as a 6D
        twist. Note that this needs to be a rigid-body velocity, i.e. a member 
        of se(3) expressed as a 6D vector.

        Parameters
        ----------
        time : float

        Returns
        -------
        6x' :obj:`numpy.ndarray`
            desired body-frame velocity of the end effector
        """
        t = time
        
        for i in range(0, self.n - 1):
            if t >= self.t_chunk[i] and t <= self.t_chunk[i+1]:
                self.linear.total_time = self.t_chunk[i+1] - self.t_chunk[i]
                self.linear.p0 = self.points[i]
                self.linear.p1 = self.points[i+1]
                return self.linear.target_velocity(t-self.t_chunk[i])
        pass

def define_trajectories(args):
    """ Define each type of trajectory with the appropriate parameters."""
    trajectory = None
    if args.task == 'line':
        trajectory = LinearTrajectory()
    elif args.task == 'circle':
        trajectory = CircularTrajectory()
    elif args.task == 'polygon':
        trajectory = PolygonalTrajectory()
    return trajectory

if __name__ == '__main__':
    """
    Run this file to visualize plots of your paths. Note: the provided function
    only visualizes the end effector position, not its orientation. Use the 
    animate function to visualize the full trajectory in a 3D plot.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-task', '-t', type=str, default='line', help=
        'Options: line, circle, polygon.  Default: line'
    )
    parser.add_argument('--animate', action='store_true', help=
        'If you set this flag, the animated trajectory will be shown.'
    )
    args = parser.parse_args()

    trajectory = define_trajectories(args)
    
    if trajectory:
        trajectory.display_trajectory(show_animation=args.animate)