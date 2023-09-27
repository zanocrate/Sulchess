# here we define the Chessboard object

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os
import warnings
import copy

from sklearn.base import BaseEstimator
from .utils import *
from skimage.feature import canny
from skimage.color import rgb2gray
from skimage.transform import resize

import itertools




def test_lines(la1,lb1,la2,lb2,edges):

    chessboard = Chessboard()
    best_loss = np.inf

    for i in range(0,7):
        for j in range(i+1,8):
            for k in range(0,7):
                for l in range(k+1,8):

                    chessboard.warp_from_lines(np.array([la1,lb1,la2,lb2]),(i,j,k,l))
                    # loss = chessboard_loss_transform(chessboard,hspace,angles,distances)
                    loss = -(chessboard.draw_filter(*edges.shape,thickness=3)*edges).sum()
                    
                    
                    
                    
                    if loss < best_loss:
                        best_loss = loss
                        best_chessboard = copy.deepcopy(chessboard)
    
    return best_chessboard,best_loss


def arccot(x):
    if x == 0:
        return np.pi / 2  # arccot(0) = pi/2
    elif abs(x) < 1:
        return np.arctan(1 / x)
    elif x > 0:
        return np.arctan(1 / x) + np.pi
    else:
        return np.arctan(1 / x) - np.pi
    
def is_point_inside_quadrilateral(vertices, point): # thanks chatGPT my homie
    """
    vertices is an array-like with shape (4,2)
    """
    
    x, y = point
    a, b, c, d = vertices
    
    ab = (b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0])
    bc = (c[0] - b[0]) * (y - b[1]) - (c[1] - b[1]) * (x - b[0])
    cd = (d[0] - c[0]) * (y - c[1]) - (d[1] - c[1]) * (x - c[0])
    da = (a[0] - d[0]) * (y - d[1]) - (a[1] - d[1]) * (x - d[0])

    if ab >= 0 and bc >= 0 and cd >= 0 and da >= 0:
        return True
    if ab <= 0 and bc <= 0 and cd <= 0 and da <= 0:
        return True
    
    return False


def calculate_intersection(lines):
    rho1, theta1 = lines[0]
    rho2, theta2 = lines[1]

    # Check if the lines are parallel
    if np.abs(theta1 - theta2) < np.finfo(float).eps:
        raise ValueError("Lines are parallel and do not intersect.")

    # Solve the system of equations to find the intersection point
    A = np.array([[np.cos(theta1), np.sin(theta1)],
                  [np.cos(theta2), np.sin(theta2)]])
    b = np.array([rho1, rho2])

    try:
        intersection = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        raise ValueError("Lines do not intersect.")

    return intersection


class Chessboard:

    def __init__(self,n_rows=8,n_cols=8):

        self.unwarped_corners_ = np.array([
                    [0,0],
                    [1,0],
                    [0,1],
                    [1,1]
                ],dtype=np.float32) # important for opencv otherwise assertion fails
        
        # we want it (18,3), i.e. 9 growing x lines and 9 growing y lines, and their vector representation
        
        # get the lines in the ax+by+c=0 form (a,b,c)
        y_parallel_lines=np.stack((
            np.ones(n_rows+1),
            np.zeros(n_rows+1),
            -1/n_rows*np.arange(n_rows+1)
        )).T
        # same for y axis
        x_parallel_lines=np.stack((
            np.zeros(n_cols+1),
            np.ones(n_cols+1),
            -1/n_cols*np.arange(n_cols+1)
        )).T

        self.unwarped_vector_lines_ = np.vstack((y_parallel_lines,x_parallel_lines))
        
        x = np.linspace(0,1,n_rows+1)
        y = np.linspace(0,1,n_cols+1)
        xx,yy=np.meshgrid(x,y)

        # is an array (2,n_rows+1,n_cols+1) where the first index chooses between x and y
        self.unwarped_points_ = np.stack(np.meshgrid(x,y))
        self.warped_points_ = np.copy(self.unwarped_points_)

        # 8x8 means standard chessboard
        self.n_rows = n_rows
        self.n_cols = n_cols

    def _sort_points(self,points):
        """
        Helper function to sort a (4,2) array of points such that the order is always (bottom_left,bottom_right,top_left,top_right)
        """
        



    def warp_homography(self,H):
        """
        Warp directly from a specified homography.
        Args:
        ------
         H : float, (3,3) matrix
        """

        self.H = H
        # some reshaping magic to comply with opencv
        input_points = self.unwarped_points_.reshape(2, (self.n_rows+1)*(self.n_cols+1) ).T.reshape(1, (self.n_rows+1)*(self.n_cols+1) ,2)
        # but still get back the (2,n_rows+1,n_cols+1) format
        self.warped_points_=cv.perspectiveTransform(input_points,self.H)[0].T.reshape(2,self.n_rows+1,self.n_cols+1)

        # line representation
        # initialize array; (n_lines_rows,n_lines_cols,n_features)
        self.warped_lines_ = np.zeros((self.n_rows+1+self.n_cols+1,2))

        # these are shaped like (2,n_rows+1) or (2,n_cols+1)
        self.warped_lower_edge_points = self.warped_points_[:,0,:]
        self.warped_left_edge_points = self.warped_points_[:,:,0]
        self.warped_right_edge_points = self.warped_points_[:,:,-1]
        self.warped_upper_edge_points = self.warped_points_[:,-1,:]

        # first populate the rows, i.e. growing x in unprojected space
        for line,(p1,p2) in enumerate(zip(self.warped_lower_edge_points.T,self.warped_upper_edge_points.T)):
            dx = (p2[0]-p1[0])
            dy = (p2[1]-p1[1])
            if np.isclose(dx,0): theta = 0
            else: theta = arccot(-dy/dx)
            rho = p2[0]*np.cos(theta) + p2[1]*np.sin(theta) 

            # periodic boundary
            if theta > np.pi/2:
                theta = theta - np.pi
                rho = -rho
            elif theta < -np.pi/2:
                theta = theta + np.pi
                rho = -rho

            self.warped_lines_[line,0] = rho
            self.warped_lines_[line,1] = theta

        # then populate columns
        for line,(p1,p2) in enumerate(zip(self.warped_left_edge_points.T,self.warped_right_edge_points.T)):
            dx = (p2[0]-p1[0])
            dy = (p2[1]-p1[1])
            if np.isclose(dx,0): theta = 0
            else: theta = arccot(-dy/dx)
            rho = p2[0]*np.cos(theta) + p2[1]*np.sin(theta) 


            # periodic boundary
            if theta > np.pi/2:
                theta = theta - np.pi
                rho = -rho
            elif theta < -np.pi/2:
                theta = theta + np.pi
                rho = -rho

            self.warped_lines_[line+self.n_rows+1,0] = rho
            self.warped_lines_[line+self.n_rows+1,1] = theta
    
    def warp(self,points,starting_points=None):
        """
        Warp the chessboard using a linear map in the projective space using 4 destination corners.
        The input is a np.array of shape (4,2)
        """
        
        # sort to match the reference grid
        sorted_points = np.copy(points[np.lexsort((points[:,0],points[:,1])) ]).astype(np.float32)


        if starting_points is None: starting_points = self.unwarped_corners_
        # homography
        H = cv.getPerspectiveTransform(starting_points,sorted_points)

        self.warp_homography(H)

    def warp_from_lines(self,lines,indices=None):
        """
        Performs warping but identifying the corners by 2 sets of 2 lines, each couple representing opposing sides of the chessboard.
        Args:
        --------
            lines : array like (4,2); lines[:2] are assumed to be opposing sides, as well as lines[2:]. Features are (r,theta)
            i,j,k,l : ints, from 0 to n_rows and from 0 to n_cols. determines which chessboard line these correspond to.
        """

        opposing_sides_1 = lines[:2]
        opposing_sides_2 = lines[2:]

        intersections = []
        
        for a in range(2):
            for b in range(2):
                x,y = calculate_intersection(np.array([opposing_sides_1[a],opposing_sides_2[b]]))
                intersections.append([x,y])

        # now let's get the points corresponding to the correct line.
        # line_1 and line_2 are parallel to y axis
        # so i will tell the x index of the lower and upper left corner
        #    j           the x index of the lower and upper right corner
        #    k           the y index of the lower right and left corner
        #    l           the y index of the upper right and left corner

        if indices is not None:
            i,j,k,l = indices
            starting_points = np.array([
                self.unwarped_points_[:,i,k],
                self.unwarped_points_[:,j,k],
                self.unwarped_points_[:,i,l],
                self.unwarped_points_[:,j,l]
            ],dtype=np.float32)


            starting_points = np.copy(starting_points[np.lexsort((starting_points[:,0],starting_points[:,1])) ]).astype(np.float32)
            
        else:
            starting_points=None

        self.warp(np.array(intersections),starting_points)




    def return_indices(self,point):
        """
        Accepts a coordinate x,y and returns the indices in the form (i_row,i_col) of the cell it belongs to.
        """

        for i in range(self.n_rows):
            for j in range(self.n_cols):
                
                a = self.warped_points_[:,i,j]
                b = self.warped_points_[:,i+1,j]
                c = self.warped_points_[:,i+1,j+1]
                d = self.warped_points_[:,i,j+1]

                if is_point_inside_quadrilateral((a,b,c,d),point):

                    return i,j
        # returns none if no cell was found
        return None


    def plot_chessboard(
            self,
            ax=None,
            points_kwargs = {'color':'darkgreen'},
            lines_kwargs = {'color':'green','linestyle':'--','alpha':0.7}
    ):
        """
        Helper function to plot the chessboard.
        If the ax is given, will plot there.
        Can also pass kwargs to points and line plotting functions.
        """

        if ax is None:
            fig,ax=plt.subplots()
        
        ax.scatter(self.warped_points_[0].flatten(),self.warped_points_[1].flatten(),color='green')
        for line,(p1,p2) in enumerate(zip(self.warped_lower_edge_points.T,self.warped_upper_edge_points.T)):
            ax.plot([p1[0],p2[0]],[p1[1],p2[1]],color='green',alpha=0.4,linestyle='--')
        for line,(p1,p2) in enumerate(zip(self.warped_left_edge_points.T,self.warped_right_edge_points.T)):
            ax.plot([p1[0],p2[0]],[p1[1],p2[1]],color='green',alpha=0.4,linestyle='--')


    def draw_filter(self,width,heigth,thickness=1,lineType=4):

        filter = np.zeros((width,heigth))
        
        # one bundle
        for start_point,end_point in zip(self.warped_points_[:,0,:].T,self.warped_points_[:,-1,:].T):
            
            filter = cv.line(filter,start_point.astype(int),end_point.astype(int),color=1.,thickness=thickness,lineType=lineType)
        # other one
        for start_point,end_point in zip(self.warped_points_[:,:,0].T,self.warped_points_[:,:,-1].T):
            filter = cv.line(filter,start_point.astype(int),end_point.astype(int),color=1.,thickness=thickness,lineType=lineType)
        return filter

        

def compute_vanishing_point(rho,theta,weights):
    """
    Args:
    ------
        rho : array of (n_samples,)
        theta : array of (theta,)
    Returns:
    ------
        vanishing_point : tuple (2,), best estimate of vanishing point
        loss : sum of squared residuals
    """

    A = (weights*np.cos(theta)**2).sum()
    B = (weights*np.sin(theta)**2).sum()
    C = (weights*np.sin(theta)*np.cos(theta)).sum()
    D = (weights*np.cos(theta)*rho).sum()
    E = (weights*np.sin(theta)*rho).sum()

    coefficient_matrix = np.array([[A,C],[C,B]])
    intercept = np.array([D,E])

    vanishing_point,_,_,_=np.linalg.lstsq(coefficient_matrix,intercept,rcond=None)


    
    return vanishing_point



class VanishingPointRegressor(BaseEstimator):

    def fit(self,X,y,sample_weight=None):

        n_samples = len(X)
        # expects X to be theta,
        # y to be rho
        rho = y
        theta= X[:,0]


        if sample_weight is None:
            sample_weight = np.ones(n_samples)

        self.x0,self.y0 = compute_vanishing_point(rho,theta,sample_weight)
    
    def score(self,X,y,sample_weight=None):

        n_samples = len(X)
        # expects X to be theta,
        # y to be rho
        rho = y
        theta= X[:,0]


        if sample_weight is None:
            sample_weight = np.ones(n_samples)

        rho_pred = self.x0*np.cos(theta) + self.y0*np.sin(theta)
        # squared distance from the cycloid defined by the vanishing point
        residuals = sample_weight*(rho - rho_pred )**2

        return r2_score(rho,rho_pred)
    
    def predict(self,X):

        theta=X[:,0]

        return self.x0*np.cos(theta) + self.y0*np.sin(theta)
    


def bootstrap_sinusoidal(edges,data_valid_threshold=np.pi/8,num_peaks=20,ransac_kwargs={'min_samples':8,'max_trials':5000}):
    """
    Takes a Canny edge detector image and fits two sinusoidal curves to find set of parallel lines.
    These will serve as a bootstrapping estimate for the more accurate subset of samples  in order to find all the peaks
    in a much larger samples set.

    Return:
    ----------
        
    """

    # get peaks
    hspace,angles,distances = hough_line(edges)
    peaks = pd.DataFrame(hough_line_peaks(hspace,angles,distances,num_peaks=num_peaks,threshold=1)).T # low threshold but limited numer of peaks
    peaks.columns = ['count','theta','rho']


    def data_valid(X,y):
        """Check if data is valid by looking at the range in the theta direction."""

        if (X.max() - X.min()) > data_valid_threshold:
            return False
        else:
            return True
    
    estimator = VanishingPointRegressor()
    ransac = RANSACRegressor(estimator,is_data_valid=data_valid,**ransac_kwargs)

    # first regression
    X = peaks.theta.values.reshape(-1,1)
    y = peaks.rho.values

    ransac.fit(X,y)
    peaks['inlier'] = ransac.inlier_mask_

    vp_1 = (ransac.estimator_.x0,ransac.estimator_.y0)

    # second regression
    X = peaks[~peaks.inlier].theta.values.reshape(-1,1)
    y = peaks[~peaks.inlier].rho.values

    ransac.fit(X,y)
    
    peaks.loc[~peaks.inlier,'inlier'] = ransac.inlier_mask_*2

    vp_2 = (ransac.estimator_.x0,ransac.estimator_.y0)

    peaks['inlier']*=1

    return vp_1,vp_2,peaks,hspace,angles,distances


def fit_chessboard(
        img,
        ransac_kwargs={
                            'min_samples' : 4,
                            'max_trials' : 50000
                        },
        canny_kwargs={},
        q1=0.4, # quantiles to pick for chessboard estimation
        q2=0.6


):
    

    edges = canny(img,**canny_kwargs)

    
    # perform double ransac regression
    vp_1,vp_2,peaks,hspace,angles,distances=bootstrap_sinusoidal(edges,ransac_kwargs=ransac_kwargs)


    # grab the q1 q2 quantiles for each bundle
    la1 = peaks[peaks.inlier==1][peaks[peaks.inlier==1].rho == peaks[peaks.inlier==1].rho.quantile(q1,interpolation='lower')][['rho','theta']].values[0]
    lb1 = peaks[peaks.inlier==1][peaks[peaks.inlier==1].rho == peaks[peaks.inlier==1].rho.quantile(q2,interpolation='higher')][['rho','theta']].values[0]

    la2 = peaks[peaks.inlier==2][peaks[peaks.inlier==2].rho == peaks[peaks.inlier==2].rho.quantile(q1,interpolation='lower')][['rho','theta']].values[0]
    lb2 = peaks[peaks.inlier==2][peaks[peaks.inlier==2].rho == peaks[peaks.inlier==2].rho.quantile(q2,interpolation='lower')][['rho','theta']].values[0]


    # now, for the selected lines, we try every possible combinations of indices and see the best one
    best_chessboard,loss=test_lines(la1,lb1,la2,lb2,edges)

    
    
    return best_chessboard
