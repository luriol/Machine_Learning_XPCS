import numpy as np
from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt
from lmfit import Model

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




def ridge_model(x, amp, xp0, lam, bg):
    """
    Exponential ridge model:
        y = bg + amp * exp(-abs(x - xp0) / lam)
    """
    return bg + amp * np.exp(-np.abs(x - xp0) / lam)


def fit_ridge_amplitude(
    x,
    y,
    ridge_width=4,
    cen_width=2,
    make_plot=False
):
    """
    Fit a symmetric exponential ridge profile near x = 0 and return the
    fitted peak height at x = 0 (including background).

    Parameters
    ----------
    x : array_like
        x coordinates.
    y : array_like
        Data values corresponding to x.
    ridge_width : int, optional
        Only points with |x| <= ridge_width are considered.
    cen_width : int, optional
        Points with |x| <= cen_width are excluded from the fit.
    make_plot : bool, optional
        If True, generate a diagnostic plot.

    Returns
    -------
    peak_height : float
        Value of the fitted function at x = 0, including background.
    """

    x = np.asarray(x)
    y = np.asarray(y)

    # Select points within the fitting window
    window_mask = np.abs(x) <= ridge_width
    x_window = x[window_mask]
    y_window = y[window_mask]

    if len(x_window) == 0:
        raise ValueError("No data points found within ridge_width.")

    # Exclude the central region
    fit_mask = np.abs(x_window) > cen_width
    xfit = x_window[fit_mask]
    yfit = y_window[fit_mask]

    if len(xfit) < 4:
        raise ValueError(
            "Not enough points remain after excluding the central region."
        )

    # Build model
    model = Model(ridge_model)

    # Initial guesses
    params = model.make_params(
        amp=np.max(yfit) - np.min(yfit),
        xp0=ridge_width / 2,
        lam=max(1.0, ridge_width),
        bg=np.min(yfit)
    )

    # Constraints
    params['lam'].min = 1e-6
    params['xp0'].min = -ridge_width
    params['xp0'].max = ridge_width

    # Perform fit
    result = model.fit(yfit, params, x=xfit)

    # Peak height at x = 0 (includes background)
    peak_height = result.eval(x=np.array([0.0]))[0]

    # Optional diagnostic plot
    if make_plot:
        xplot = np.linspace(x_window.min(), x_window.max(), 400)
        yplot = result.eval(x=xplot)

        plt.figure()

        # All points in the fitting window
        plt.plot(
            x_window, y_window, 'o',
            color='0.75',
            label='all points in window'
        )

        # Points used in the fit
        plt.plot(
            xfit, yfit, 'ob',
            label='points used in fit'
        )

        # Fitted curve
        plt.plot(
            xplot, yplot, '-r',
            linewidth=2,
            label='fit'
        )

        # Reference lines
        plt.axvline(
            0,
            color='k',
            linestyle='--',
            alpha=0.5,
            label='x = 0'
        )

        plt.axvline(
            result.params['xp0'].value,
            color='m',
            linestyle='--',
            alpha=0.7,
            label='fit xp0'
        )

        plt.axhline(
            peak_height,
            color='g',
            linestyle=':',
            alpha=0.7,
            label=f'peak = {peak_height:.4g}'
        )

        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(
            f'Ridge fit (ridge_width={ridge_width}, '
            f'cen_width={cen_width})'
        )
        plt.legend()
        plt.grid(True)

        print(result.fit_report())
        print(f"Peak height at x = 0: {peak_height}")

    return peak_height