import numpy as np
from scipy.ndimage import map_coordinates

def diagonal_resample_square(M, frac=0.5, half_size=50, dx=1.0):
    """
    Interpolate matrix M onto a rotated coordinate system.

    Rotated coordinates:
        y' : along the main diagonal (bottom-left to top-right)
        x' : perpendicular to the diagonal

    Parameters
    ----------
    M : 2D ndarray
        Original image/matrix.
    frac : float
        Fraction along the main diagonal for the center.
    half_size : float
        Half-range of y' in pixels. y' runs from -half_size to +half_size.
    dx : float
        Sampling step in rotated coordinates.

    Returns
    -------
    Mrot : 2D ndarray
        Interpolated rotated matrix.
    xprime : 1D ndarray
        Perpendicular coordinate values.
    yprime : 1D ndarray
        Diagonal coordinate values.
    x0, y0 : float
        Center in original pixel coordinates.
    xprime_max : float
        Largest symmetric half-width allowed for x'.
    corners : (4,2) ndarray
        Corner coordinates of the rotated region in original image coordinates.
        Order: lower-left, upper-left, upper-right, lower-right in (x,y).
    boundary : (5,2) ndarray
        Same corners, but closed by repeating the first point at the end.
    """
    ny, nx = M.shape

    x0 = frac * (nx - 1)
    y0 = frac * (ny - 1)

    s2 = np.sqrt(2.0)

    def xy_from_rot(xp, yp):
        x = x0 + (yp - xp) / s2
        y = y0 + (yp + xp) / s2
        return x, y

    def corners_inside(xmax):
        test_corners = [
            (-xmax, -half_size),
            (-xmax, +half_size),
            (+xmax, +half_size),
            (+xmax, -half_size),
        ]
        for xp, yp in test_corners:
            x, y = xy_from_rot(xp, yp)
            if not (0 <= x <= nx - 1 and 0 <= y <= ny - 1):
                return False
        return True

    lo = 0.0
    hi = max(nx, ny)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if corners_inside(mid):
            lo = mid
        else:
            hi = mid
    xprime_max = lo

    xprime = np.arange(-xprime_max, xprime_max + dx, dx)
    yprime = np.arange(-half_size, half_size + dx, dx)

    XP, YP = np.meshgrid(xprime, yprime)

    X = x0 + (YP - XP) / s2
    Y = y0 + (YP + XP) / s2

    coords = np.array([Y, X])
    Mrot = map_coordinates(M, coords, order=1, mode='nearest')

    corners = np.array([
        xy_from_rot(-xprime_max, -half_size),  # lower-left in rotated coords
        xy_from_rot(-xprime_max, +half_size),  # upper-left
        xy_from_rot(+xprime_max, +half_size),  # upper-right
        xy_from_rot(+xprime_max, -half_size),  # lower-right
    ])

    boundary = np.vstack([corners, corners[0]])

    return Mrot, xprime, yprime, x0, y0, xprime_max, corners, boundary