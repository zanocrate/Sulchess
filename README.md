# SULChess - yet another chessboard detection algorithm

This is the repository for a Computer Vision project I developed during my studies at University of Padova. The goal of the project was to come up with an original CV algorithm to extrapolate a FEN notation string from a digital image representing a chess game.

As always, the task is conceptually made out of two sub tasks:

1. Detect the chess pieces 
2. Detect the chessboard cells

The first task is a instance of an object detection problem; for this, a ML approach using CNN-based models is used. For this implementation, I trained a YOLOv8n model for object detection using the publicly available [Chess Pieces Dataset](https://public.roboflow.com/object-detection/chess-full/24/download) for experiments.

## Chessboard detection

The second task is the one that this algorithm tackles; the process is essentialy as follows:

1. Preprocess the image by resizing to a predefined resolution, converting to grayscale and computing the Canny edge detector
2. Compute the Hough transform of the image, and detect the most prominent peaks
3. Run twice a RANSAC algorithm to estimate the two vanishing points of the two bundle of parallel lines that define the chessboard
4. From each of the two inliers group, sample 2 lines acting as opposing lines, assuming they belong to the chessboard but not knowing which ones
5. For every combination of possible line indices, compute the homography, mapping the chessboard cells accordingly to the indices sampled
6. Compute a "chessboard" filter, and compute the response of this filter with the Canny edge detector processed image
7. Keep the highest response chessboard as the detected one

See `notebooks/chessboard_detection.ipynb` for an in depth explanation and examples.

