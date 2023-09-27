import numpy as np
import cv2 as cv

from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize
from sklearn.base import BaseEstimator
from sklearn.metrics import r2_score

import pandas as pd

from sklearn.linear_model import RANSACRegressor
from skimage.transform import hough_line,hough_line_peaks

import torch

from .chessboard import *


def load_skimage(filepath, resize_kwargs={}):
    """
    Load and preprocess image using Sklearn library. Returns a downsampled grayscale image with fixed (width,heigth)
    """
    img = imread(filepath)
    return rgb2gray(downsample(img,**resize_kwargs))

def downsample(img,width=800,height=None):
    
    if len(img.shape)==3:
        W,H,C = img.shape
    elif len(img.shape)==2:
        W,H = img.shape
    else:
        raise ValueError('Incorrect shape of image array')
    
    
    aspect_ratio = H/W
    if height is None: height = int(width*aspect_ratio)
    
    downsample = resize(img,(width,height))

    return downsample


def load_opencv(filepath,width=800,height=None):
    """
    Load and preprocess image using OpenCV library. Returns a downsampled grayscale image with fixed (width,heigth)
    """


    img = cv.imread(filepath)
    W,H,C = img.shape
    aspect_ratio = H/W
    if height is None: height = int(width*aspect_ratio)
    img = cv.resize(img,(height,width))
    # grayscale image
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    return gray





def get_median_row(dataframe,column,distance=0):
    """
    grab the i-th row closest to the specified field median
    distance can also be a list of indices
    """
    
    median = dataframe.median()[column]
    arg = abs(dataframe[column]-median).sort_values().index[list(distance)]
    return dataframe.loc[arg]


def generate_FEN(
    matrix,
    player_to_move : str,
    white_can_short_castle : bool,
    white_can_long_castle : bool,
    black_can_short_castle : bool,
    black_can_long_castle : bool,
    en_passant_target_square = '-',
    halfmove_clock= 0, # number of halfmoves since last capture or pawn advance
    fullmove_number= 0 # number of full moves

):
    assert (player_to_move=='w' or player_to_move=='b')
    fen_string = ''
    num_empty_spaces = 0
    for line in matrix:
        num_empty_spaces=0
        for entry in line:

            if entry =='e':
                num_empty_spaces += 1
            else:
                if num_empty_spaces != 0: fen_string+=str(num_empty_spaces)
                fen_string+=entry
                num_empty_spaces=0
        if num_empty_spaces != 0: fen_string+=str(num_empty_spaces)
        fen_string += '/'
    fen_string=fen_string[:-1]
    fen_string += ' '+player_to_move+' '

    if (white_can_long_castle+white_can_short_castle+black_can_long_castle+black_can_short_castle == 0): fen_string+= '-'
    else:
        if white_can_short_castle:
            fen_string+='K'
        if white_can_long_castle:
            fen_string+='Q'
        if black_can_short_castle:
            fen_string+='k'
        if black_can_long_castle:
            fen_string+='q'
    
    fen_string+=' '+en_passant_target_square+' '
    fen_string+=str(halfmove_clock)+' '+str(fullmove_number)

    return fen_string


def predict_FEN(
    filenames : list,
    model,
    fen_kwargs,
    k = 0, # number of 90 degrees rotation of the chessboard
    flipud = False,
    confidence=0.8,
    fen_classes = ['TEST','b','k','n','p','q','r','B','K','N','P','Q','R'],

    ):

    fen_strings = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # get a list of results, one per images
    results = result = model(filenames,conf=confidence)


    # place pieces for each image
    for fn,result in zip (filenames,results):

        matrix = (np.ones((8,8))).astype(str)
        matrix.fill('e')

        img = downsample(imread(fn))
        gray = rgb2gray(img)
        best_chessboard = fit_chessboard(gray)

        # now we loop through the results of the cnn
        for box in result.boxes:

            class_num = int(box.cls.item())
            class_str = fen_classes[class_num]
            
            # now compute the barycenter of each box
            # that will be our "anchor point"
            # x_piece = (box.xywhn[0,0] - box.xywhn[0,2]).cpu().numpy()
            # y_piece = box.xywhn[0,1].cpu().numpy()

            x_piece = (box.xyxy[0,2].cpu().numpy() + box.xyxy[0,0].cpu().numpy())/2
            y_piece = box.xyxy[0,3].cpu().numpy() 

            # we need to fix the dimensions to match the chessboard processed image
            W_chessboard,H_chessboard,C=img.shape
            W_orig,H_orig=result.orig_shape

            ratio_W = W_chessboard/W_orig
            ratio_H = H_chessboard/H_orig

            x_piece *= ratio_W
            y_piece *= ratio_H

            # which cell is it?
            try:
                i,j = best_chessboard.return_indices((x_piece,y_piece))
                matrix[i,j] = class_str
            except:
                next

        #flip
        if flipud: matrix = np.flipud(matrix)

        # apply rotation
        matrix = np.rot90(matrix,k)

        fen_string = generate_FEN(matrix,**fen_kwargs)

        fen_strings.append(fen_string)

    return fen_strings