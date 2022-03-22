import numpy as np
from scipy.spatial.transform import Rotation as R
import torch


def cine2gridtag(im, label, myo_index: int = 2, mod: float = 0.4, spacing: int = 5, **kwargs):
    """Modifies contrast and adds grid tags (line tags in perpendicular directions) to an image

    Parameters
    ----------
    im : ndarray
        2D image that will have grid tag lines applied
    label : ndarray, or None
        Image labels for ROIs of intrest, can be None to ignore this feature
    myo_index : int
        The index of the LV mask in label, default is 2 per ACDC data
    mod : float
        The contrast modification exponential power
    spacing : float
        The distance between lines, in pixels
    **kwargs : optional
        Passthrogh arguments for sim_gridtag

    Returns
    -------
    ndarray
        The input image with grid tag lines added

    """

    # Try to handle tensor inputs (this is undone at the end of the function)
    is_tensor = False
    if torch.is_tensor(im):
        is_tensor = True
        im = im.numpy()

    if label is not None:
        if torch.is_tensor(label):
            label = label.numpy()

        if label.shape != im.shape:  # This may already be checked somewhere, probably not needed
            print('ERROR: shapes not equal', label.shape, im.shape)
        
        myo_mask = (label == myo_index)

        if myo_mask.sum() > 10:  # Some masks in ACDC are empty? (10 is arbitrary)
            # This selects the image intensity in the 0-50th quantile from the
            # LV.  Randomness per Dan's comment
            transition_val = np.quantile(im[myo_mask], np.random.rand() / 2)
            im = mod_contrast(im, mod, transition_val)

        else:
            im = mod_contrast(im, mod)
    else:
        im = mod_contrast(im, mod)

    im = sim_gridtag(im, spacing, **kwargs)

    im_out = im
    if is_tensor:
        im_out = torch.tensor(im_out)

    return im_out


def mod_contrast(im, mod: float = 0.4, transition_val: float = 0):
    """Modify image contrast
    Parameters
    ----------
    im : ndarray
        2D image
    mod : float, optional
        exponent to apply to the image, by default 0.4
    transition_val : float, optional
        Below this value, contrast will remain linear, and only apply contrast mod after

    Returns
    -------
    ndarray
        The contrast modified image

    Todo
    ----

    More can definitely be done here to get closer to tag contrast, perhaps using
    a more custom contrast curve, or maybe even some form of masking

    """

    if transition_val > 0:
        im /= transition_val  # Make it so our critical point where contrast changes is at 1
        im[im >= 1] = im[im >= 1] ** mod
        im *= transition_val
    else:
        im = im ** mod

    return im


def sim_gridtag(im, spacing=5, total_flip=70, flip_pattern=(1, 3, 3, 1), x_offset=0, y_offset=0):
    """Adds grid tags (line tags in perpendicular directions) to an image

    Parameters
    ----------
    im : ndarray
        2D image that will have grid tag lines applied
    spacing : float
        The distance between lines, in pixels
    total_flip : float, optional
        The total RF flip (degrees) of the tagging pulse, by default 70
        A higher number here, up to 90, will make the dark regions of the tag lines
        darker, with 90 being complete nulling.
    flip_pattern : tuple or ndarray, optional
        The type of tagging pulse to play, by default (1, 3, 3, 1)
        Splitting the tagging pulse into a binomial pulse makes the tag lines
        sharper, other options would be (1, 1) which would give broad lines, and
        (1, 4, 6, 4, 1) for even sharper lines
    x_offset : float, optional
        Offsets the line pattern in the x direction, in pixels, by default 0
    y_offset : int, optional
        Offsets the line pattern in the y direction, in pixels, by default 0

    Returns
    -------
    ndarray
        The input image with grid tag lines added

    Todo
    ----

    Maybe mask out air pixels from getting tag lines?

    """

    singleton_dim0 = False
    if im.ndim == 3 and im.shape[0] == 1:
        singleton_dim0 = True
        im = im.squeeze()

    flip_pattern = np.array(flip_pattern)

    # Flip angles in radians -- reduce to have less
    flip_angles = total_flip * flip_pattern / flip_pattern.sum() * np.pi / 180

    # We now make a map that represents the spatial gradients that get played out
    # between RF pulses, which determines tag line spacing
    xn, yn = tuple(im.shape[:2])
    X, Y = np.meshgrid(np.arange(xn), np.arange(yn), indexing='ij')
    
    # Offset the origin so lines can be shifted
    X += x_offset
    Y += y_offset

    # Scale factor to have line distance = "spacing"
    moment = (2 * np.pi) / (spacing * np.sqrt(2))
    # Note: if this wasnt diagonal it would be 2 pi / spacing

    # Now we apply the line tages in both perpendicular and diagonal  directions
    G_theta = moment * (X + Y)
    im2 = sim_linetag(im, flip_angles, G_theta)
    
    G_theta = moment * (X - Y)
    im3 = sim_linetag(im2, flip_angles, G_theta)

    if singleton_dim0:
        im3 = im3[None, ...]

    return im3


def sim_linetag(im, flip_angles, G_theta):
    """Adds tag lines to an image using a psuedo-Bloch simulation

    Parameters
    ----------
    im : ndarray
        2D image that will have tag lines applied
    flip_angles : ndarray
        1D array of flip angles (radians) for the simulation
    G_theta : ndarray
        2D map of z-axis rotation (radians) for each point in image

    Returns
    -------
    ndarray
        The input image with tag lines added

    Todo
    ----
    There are probably too many vector copies here (Mr, Mr2, Mr3), but the
    wasted memory isnt big...

    Check G_theta is the same size as im
    """
    
    # Magnetization vector for each point in the image
    M = np.zeros((im.size, 3))
    M[:, 2] = im.ravel()
    
    G_theta = G_theta.ravel()

    for i in range(flip_angles.size - 1):
        
        # Tip all spins along the x-axis
        r = R.from_euler('x', flip_angles[i])
        Mr = r.apply(M)

        # Rotate around the z-axis, based on position
        Mr2 = Mr.copy()
        Mr2[:, 0] = np.cos(G_theta) * Mr[:, 0] - np.sin(G_theta) * Mr[:, 1]
        Mr2[:, 1] = np.sin(G_theta) * Mr[:, 0] + np.cos(G_theta) * Mr[:, 1]

        M = Mr2.copy()

    # Apply the final RF tip on the x-axis
    r = R.from_euler('x', flip_angles[i + 1])
    Mr3 = r.apply(Mr2)

    # The final image is whatever magnetization is in the z direction
    im2 = np.abs(Mr3[:, 2])
    im2 = np.reshape(im2, im.shape)

    return im2
