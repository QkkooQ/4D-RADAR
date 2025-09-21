import numpy as np

points = np.fromfile('DRadDataset\\training\\velodyne\\000040.bin', dtype=np.float32).reshape(-1,4)  # N x 4
points_xyz = points[:,:3]  # 取 x,y,z

# 保存前 500 个点为文本方便粘贴
np.savetxt('points_sample.txt', points_xyz[:500], fmt='%.6f')
