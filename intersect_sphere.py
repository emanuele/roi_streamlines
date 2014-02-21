import numpy as np
from numba import jit

@jit('b1(f8[:,:],f8[:],f8)')
def my_inside_sphere2(xyz, center, r2):
    for i in range(xyz.shape[0]):
        tmp = (xyz[i,0] - center[0])
        tot = tmp * tmp
        tmp = (xyz[i,1] - center[1])
        tot = tmp * tmp
        tmp = (xyz[i,2] - center[2])
        tot = tmp * tmp
        if tot <=  r2:
            return True

    return False

def my_inside_sphere(xyz,center,radius2):
    """Faster version of dipy.tracking.metrics.inside_sphere.
    """
    tmp = xyz - center
    return (np.sum((tmp * tmp), axis=1) <= radius2).any()

def compute_intersecting1(voxel, R, coords, s_index, tracks):
    x_idx = np.where((coords[:,0] >= (voxel[0] - R)) & (coords[:,0] <= (voxel[0] + R)))[0]
    y_idx = x_idx[np.where((coords[:,1][x_idx] >=  (voxel[1] - R)) & (coords[:,1][x_idx] <= (voxel[1] + R)))[0]]
    z_idx = y_idx[np.where((coords[:,2][y_idx] >=  (voxel[2] - R)) & (coords[:,2][y_idx] <= (voxel[2] + R)))[0]]
    
    s_idx = np.unique(s_index[z_idx])
    # s_idx = np.array(list(set(s_index[z_idx]))) # expected to be faster but it is slower
    return s_idx[np.array([my_inside_sphere(xyz, voxel, R * R) for xyz in tracks[s_idx]])]

def compute_intersecting2(voxel, R, coords, s_index, tracks):
    idx = ((coords >= (voxel - R)) & (coords <= (voxel + R))).all(axis=1)
    s_idx = np.unique(s_index[idx])
    return s_idx[np.array([my_inside_sphere(xyz, voxel, R * R) for xyz in tracks[s_idx]])]


def compute_intersecting3(voxel, R, coords, s_index, tracks):
    idx = (coords[:,0] >= (voxel[0] - R)) & (coords[:,0] <= (voxel[0] + R))
    idx[idx] = (coords[idx,1] >= (voxel[1] - R)) & (coords[idx,1] <= (voxel[1] + R))
    idx[idx] = (coords[idx,2] >= (voxel[2] - R)) & (coords[idx,2] <= (voxel[2] + R))    
    s_idx = np.unique(s_index[idx])
    return s_idx[np.array([my_inside_sphere(xyz, voxel, R * R) for xyz in tracks[s_idx]])]


def compute_intersecting4(voxel, R, coords, s_index, tracks):
    idx = np.where((coords[:,0] >= (voxel[0] - R)) & (coords[:,0] <= (voxel[0] + R)))[0]
    idx = idx[np.where((coords[:,1][idx] >= (voxel[1] - R)) & (coords[:,1][idx] <= (voxel[1] + R)))[0]]
    idx = idx[np.where((coords[:,2][idx] >= (voxel[2] - R)) & (coords[:,2][idx] <= (voxel[2] + R)))[0]]
    s_idx = np.unique(s_index[idx])
    return s_idx[np.array([my_inside_sphere(xyz, voxel, R * R) for xyz in tracks[s_idx]])]


def compute_intersecting5(voxel, R, coords, s_index, tracks):
    def f(s):
        return ((s >= voxel - R) & (s <= voxel + R)).all(axis=1).any()
    s_idx = np.where([f(s) for s in tracks])[0]
    return s_idx[np.array([my_inside_sphere(xyz, voxel, R * R) for xyz in tracks[s_idx]])]

def compute_intersecting6(voxel, R, coords, s_index, tracks):
    tmp = coords - voxel
    idx = np.where(np.sum((tmp * tmp), axis=1) <= (R * R))[0]
    return np.unique(s_index[idx])


def compute_intersecting7(voxel, R, coords, s_index, tracks):
    idx = np.where((coords[:,0] >= (voxel[0] - R)) & (coords[:,0] <= (voxel[0] + R)))[0]
    idx = idx[np.where((coords[:,1][idx] >= (voxel[1] - R)) & (coords[:,1][idx] <= (voxel[1] + R)))[0]]
    idx = idx[np.where((coords[:,2][idx] >= (voxel[2] - R)) & (coords[:,2][idx] <= (voxel[2] + R)))[0]]
    tmp = coords[idx] - voxel
    idx = idx[np.where(np.sum((tmp * tmp), axis=1) <= (R * R))[0]]
    return np.unique(s_index[idx])


def compute_intersecting8(voxel, R, coords, s_index, tracks):
    """
    """
    idx = np.where(np.abs(coords[:,0] - voxel[0]) <= R)[0]
    idx = idx[np.where(np.abs(coords[:,1][idx] - voxel[1]) <= R)[0]]
    idx = idx[np.where(np.abs(coords[:,2][idx] - voxel[2]) <= R)[0]]
    tmp = coords[idx] - voxel
    idx = idx[np.where(np.sum((tmp * tmp), axis=1) <= (R * R))[0]]
    return np.unique(s_index[idx])

def compute_intersecting9(voxel, R, coords, s_index, tracks):
    tmp = coords[:,0] - voxel[0]
    idx = np.where((tmp <= R) & (tmp >= -R))[0]
    tmp = coords[:,1][idx] - voxel[1]
    idx = idx[np.where((tmp <= R) & (tmp >= -R))[0]]
    tmp = coords[:,2][idx] - voxel[2]
    idx = idx[np.where((tmp <= R) & (tmp >= -R))[0]]
    tmp = coords[idx] - voxel
    idx = idx[np.where(np.sum((tmp * tmp), axis=1) <= (R * R))[0]]
    return np.unique(s_index[idx])


def compute_intersecting10(voxel, R, coords, s_index, tracks):
    tmp = coords[:,0] - voxel[0]
    idx = (tmp <= R) & (tmp >= -R)
    tmp = coords[:,1][idx] - voxel[1]
    idx[idx] = (tmp <= R) & (tmp >= -R)
    tmp = coords[:,2][idx] - voxel[2]
    idx[idx] = (tmp <= R) & (tmp >= -R)
    idx = np.where(idx)[0]
    tmp = coords[idx] - voxel
    idx = idx[np.where(np.sum((tmp * tmp), axis=1) <= (R * R))[0]]
    return np.unique(s_index[idx])


@jit('b1(f8[:,:],f8,f8,f8,f8,f8)')
def compute_intersecting_streamline2(s, R, R2, vx, vy, vz):
    for i in range(s.shape[0]):
        tmpx = s[i,0] - vx
        if tmpx >= -R and tmpx <= R:
            tmpy = s[i,1] - vy
            if tmpy >= -R and tmpy <= R:
                tmpz = s[i,2] - vz
                if tmpz >= -R and tmpz <= R:
                    if (tmpx * tmpx + tmpy * tmpy + tmpz * tmpz) <= R2:
                        return True
    return False


def compute_intersecting11(voxel, R, coords, s_index, tracks):
    result = np.empty(len(tracks))
    R2 = np.double(R * R)
    vx = voxel[0]
    vy = voxel[1]
    vz = voxel[2]
    for i, s in enumerate(tracks):
        result[i] = compute_intersecting_streamline2(s, R, R2, vx, vy, vz)

    return np.where(result)[0]


if __name__ == '__main__':

    np.random.seed(0)
    min_length = 1
    max_length = 150
    size = 300000
    xyz_max = np.array([64.0, 64.0, 34.0])
    # Generating a simple simulated tractography:
    tracks = np.array([np.random.uniform(size=(np.random.randint(min_length, max_length), 3)) * xyz_max for i in range(size)], dtype=np.object)
    # Genrating index
    s_length = np.array([len(s) for s in tracks], dtype=np.int)
    s_index = np.repeat(np.arange(len(tracks)), s_length)
    # s_index = np.concatenate([i*np.ones(len(s)) for i,s in enumerate(tracks)]).astype(np.int)

    coords = np.vstack(tracks)

    implementations = [compute_intersecting1,
                       compute_intersecting2,
                       compute_intersecting3,
                       compute_intersecting4,
                       compute_intersecting5,
                       compute_intersecting6,
                       compute_intersecting7,
                       compute_intersecting8,
                       compute_intersecting9,
                       compute_intersecting10,
                       compute_intersecting11,
                       ]

    Rs = [2.0, 5.0, 10.0, 15.0, 20.0]
    voxel = xyz_max / 2.0
    from time import time

    timings = np.zeros((len(Rs), len(implementations)))
    for i, R in enumerate(Rs):
        print "R =", R
        intersecting = []
        for j, implementation in enumerate(implementations):
            t0 = time()
            result = implementation(voxel, R, coords, s_index, tracks)
            t = time() - t0
            timings[i, j] = t
            print j+1, ')', t, "sec."
            intersecting.append(result)
            # if j > 0:
            #     assert(len(intersecting[-1]) == len(intersecting[-2]))
            #     assert((intersecting[-1] == intersecting[-2]).all())

        print "Ranking (fastest first):", np.argsort(timings[i,:]) + 1
        print
        
    import matplotlib.pyplot as plt
    plt.figure()
    for i in range(len(implementations)):
        plt.plot(Rs, timings[:,i], '-o', label=str(i+1))

    plt.xlabel('R')
    plt.ylabel('sec.')
    plt.legend()
