import numpy as np
import healpy as hp

class NeedletTransform:

    def __init__(self) -> None:
        pass

    @staticmethod
    def CosineNeedlet(bandcenters):
        """
        Generate cosine bands based on the provided band centers.

        Parameters:
        - bandcenters: A list or array of band centers.

        Returns:
        - bands: A 2D array of cosine bands.
        """
        lmax = int(np.max(bandcenters))
        nbands = len(bandcenters)
        bands = np.zeros((nbands, lmax + 1), dtype=np.float64)

        for i in range(nbands):
            if i > 0:  # Left part for current band
                a, b = bandcenters[i - 1], bandcenters[i]
                ell_left = np.arange(a, b + 1)
                bands[i, ell_left] = np.cos((b - ell_left) / (b - a) * np.pi / 2)
            if i < nbands - 1:  # Right part for current band
                b, c = bandcenters[i], bandcenters[i + 1]
                ell_right = np.arange(b, c + 1)
                bands[i, ell_right] = np.cos((ell_right - b) / (c - b) * np.pi / 2)

        return bands
    
    @staticmethod
    def GaussianNeedlets(fwhm, lmax):
        """
        Generate Gaussian needlet filters based on the specified Full Width at Half Maximum (FWHM).

        Parameters:
        - fwhm: Array of FWHM values in arcminutes for each scale.
        - lmax: Maximum multipole (lmax).

        Returns:
        - filters: A 2D array of Gaussian needlet filters.
        """
        # FWHM must be in strictly decreasing order
        if any(i <= j for i, j in zip(fwhm, fwhm[1:])):
            raise AssertionError("FWHM values must be in strictly decreasing order.")

        # Convert FWHM from arcminutes to radians
        FWHM = fwhm * np.pi / (180. * 60.)

        # Define Gaussians
        nbands = len(FWHM)
        Gaussians = np.zeros((nbands, lmax + 1))
        for i in range(nbands):
            Gaussians[i] = hp.gauss_beam(FWHM[i], lmax=lmax)

        # Define needlet filters in harmonic space
        filters = np.zeros((nbands + 1, lmax + 1))
        filters[0] = Gaussians[0]
        for i in range(1, nbands):
            filters[i] = np.sqrt(Gaussians[i]**2. - Gaussians[i - 1]**2.)
        filters[nbands] = np.sqrt(1. - Gaussians[nbands - 1]**2.)

        # Simple check to ensure the sum of squared transmission is unity
        sum_squared = np.sum(filters**2, axis=0)
        if not np.allclose(sum_squared, np.ones(lmax + 1)):
            raise ValueError("Sum of squared transmission is not unity, check the filters.")

        return filters
    
    @staticmethod
    def TaperedTopHat(bandcenters, width, taperwidth):
        """
        Generate multiple Tapered Top Hat filters for an array of bandcenters.

        Parameters:
        - bandcenters: Array of central multipoles (ell) of the bands.
        - width: The half-width of the band around each bandcenter.
        - taperwidth: The width of the taper for smoothing near the edges. If 0, it becomes a sharp Top Hat.

        Returns:
        - filters: A 2D array where each row is a filter corresponding to a bandcenter.
        """
        # Determine lmax dynamically based on the highest bandcenter
        lmax = int(np.max(bandcenters) + 2 * width)  # Ensure the filter range is sufficiently covered
        ells = np.arange(lmax + 1)
        filters = np.zeros((len(bandcenters), lmax + 1))

        for i, bandcenter in enumerate(bandcenters):
            # Define the band range for this bandcenter
            lower_bound = max(0, bandcenter - width)
            upper_bound = bandcenter + width

            if taperwidth == 0:
                # Pure Top Hat filter within the band
                filters[i, lower_bound:upper_bound + 1] = 1.0
            else:
                # Tapered edges for smooth transitions
                left_taper = 0.5 * (1.0 + np.tanh((ells - lower_bound) / taperwidth))
                right_taper = 0.5 * (1.0 + np.tanh((upper_bound - ells) / taperwidth))
                filters[i] = left_taper * right_taper
                filters[i, :lower_bound] = 0.0  # Set values below the lower bound to 0
                filters[i, upper_bound + 1:] = 0.0  # Set values above the upper bound to 0

        return filters

    
    @staticmethod
    def TopHat(bandcenter, width):
        """
        Generate a single Top Hat filter.

        Parameters:
        - bandcenter: The central multipole (ell) of the band.
        - width: The half-width of the band around the bandcenter.

        Returns:
        - ells: Array of multipoles.
        - filter: A 1D array representing the filter in harmonic space.
        """
        return NeedletTransform.TaperedTopHat(bandcenter, width, taperwidth=0)
#testing