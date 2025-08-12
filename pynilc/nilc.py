import numpy as np
import healpy as hp
from pynilc.sht import HealpixDUCC
from pynilc.utils import cli, slice_alms
from typing import List, Tuple, Union
from tqdm import tqdm
import gc
import os
from concurrent.futures import ThreadPoolExecutor
import tempfile


class NILC:
    """
    NILC (Needlet Internal Linear Combination) class for component separation and residual calculation.

    Parameters
    ----------
    frequencies : List[float]
        List of frequency channels.
    fwhm : List[float]
        Full width at half maximum (FWHM) values for each frequency channel in arcminutes.
    needlet_filters : List[np.ndarray]
        List of needlet filter arrays for different scales.
    ilc_tolerance : float, optional
        Tolerance for ILC calculations, default is 1e-4.
    """
    def __init__(
        self, 
        frequencies: List[float], 
        fwhm: List[float], 
        needlet_filters: List[np.ndarray], 
        ilc_bias: float = 1e-2,
        deconvolve: bool = True,
        backend: str = "healpy",
        nside: int = None,
        common_fwhm: float = None,
        required_num_modes: int = 2500,
        nthreads: Union[str,int] = 'auto',
        parallel: int = 1,
        bias_method: str = 'nmodes',
        mask = None,
        progress_bar: bool = True
    ):
        self.frequencies = frequencies
        self.input_fwhm = np.array(fwhm)
        self.needlet_filters = needlet_filters
        self.num_needlets = len(needlet_filters)
        self.ilc_bias = ilc_bias
        self.deconvolve = deconvolve
        self.nside = nside
        self.required_num_modes = required_num_modes
        mask = np.ones(hp.nside2npix(nside)) if mask is None else mask
        self.mask = hp.ud_grade(mask, nside, order_out="RING", power=-2)
        self.bias_method = bias_method
        assert bias_method in ['nmodes', 'tol'], "bias_method must be either 'nmodes' or 'tol'"


        assert backend in ["healpy", "ducc"], "Invalid backend. Use 'healpy' or 'ducc'"
        if backend == "ducc":
            assert nside is not None, "NSIDE must be provided for DUCC backend"
            self.hp = HealpixDUCC(nside)
            self.use_healpy = False
            if nthreads == 'auto':
                self.nthreads = os.cpu_count()
            else:
                assert isinstance(nthreads, int), "nthreads must be an integer"
                self.nthreads = nthreads
        else:
            self.use_healpy = True
            self.nthreads = os.cpu_count()
        
        self.parallel = parallel
        if self.parallel == 0:
            print("Parallel processing is disabled")
        elif parallel <= 2:
            print(f"Level {self.parallel} parallel processing is enabled")
        else:
            raise ValueError("Invalid parallel processing level. Use 0, 1, or 2")
        self.common_fwhm = common_fwhm
        self.tmpdir = tempfile.mkdtemp()
        self.progress_bar = progress_bar

    @property
    def lmax_per_needlet(self) -> np.ndarray:
        """Maximum multipole for each needlet filter."""
        return np.array([np.max(np.where(filter_ > 0)) for filter_ in self.needlet_filters])

    @property
    def needlet_nside(self) -> np.ndarray:
        """NSIDE corresponding to the maximum multipole of each needlet filter."""
        return np.array([2**int(np.log2(lmax)) for lmax in self.lmax_per_needlet])
    
    def _apply_common_beam(self, alm: np.ndarray, channel: int, lmax: int):
        """
        If common_fwhm is provided, apply the re-beaming correction:
        multiply alm by (common_beam / original_beam).
        """
        if self.common_fwhm is not None:
            # Compute target common beam transfer function.
            common_beam = hp.gauss_beam(np.radians(self.common_fwhm / 60), lmax=lmax)
            # Compute original beam transfer function for this channel.
            orig_beam = hp.gauss_beam(np.radians(self.input_fwhm[channel] / 60), lmax=lmax)
            # Compute the ratio (avoid division by zero issues by clipping if needed)
            #beam_ratio = common_beam / np.clip(orig_beam, 1e-10, None)
            beam_ratio = orig_beam / np.clip(common_beam, 1e-10, None)

            hp.almxfl(alm, cli(beam_ratio), inplace=True)
        else:
            # If no common beam is provided, use the original beam for deconvolution.
            beam = hp.gauss_beam(np.radians(self.input_fwhm[channel] / 60), lmax=lmax)
            hp.almxfl(alm, cli(beam), inplace=True)

    def _map_to_alm_temperature(self, maps: List[np.ndarray]) -> np.ndarray:
        """
        Convert temperature maps to spherical harmonic coefficients (alms).

        Parameters
        ----------
        maps : List[np.ndarray]
            List of temperature maps.

        Returns
        -------
        np.ndarray
            Array of spherical harmonic coefficients.
        """
        print("Computing temperature alms")
        lmax = int(np.max(self.lmax_per_needlet))

        def process_map(i):
            temp_map = maps[i]
            if self.use_healpy:
                alm = hp.map2alm(temp_map[0], lmax=lmax)
            else:
                alm = self.hp.map2alm(temp_map[0], lmax, self.nthreads)
            if self.deconvolve:
                self._apply_common_beam(alm, i, lmax)
            return alm

        if self.parallel >= 1:
            with ThreadPoolExecutor(max_workers=len(maps)) as executor:
                alms = list(tqdm(executor.map(process_map, range(len(maps))),
                                 desc="Computing alms", total=len(maps), disable=not self.progress_bar))
        else:
            alms = [process_map(i) for i in tqdm(range(len(maps)), desc="Computing alms", disable=not self.progress_bar)]
        return np.array(alms)
    
    def _map_to_alm_polarization(self, maps: List[np.ndarray], field: int) -> np.ndarray:
        """
        Convert polarization maps to spherical harmonic coefficients (alms) in parallel.

        Parameters
        ----------
        maps : List[np.ndarray]
            List of polarization maps (Q and U components).
        field : int
            Integer specifying whether to compute E-mode (1) or B-mode (2).

        Returns
        -------
        np.ndarray
            Array of spherical harmonic coefficients.
        """
        mode = 'E' if field == 1 else 'B'
        print(f"Computing {mode}-mode alms")
        lmax = int(np.max(self.lmax_per_needlet))

        def process_map(i):
            pol_map = maps[i]
            if self.use_healpy:
                alm = hp.map2alm_spin(pol_map[1:], 2, lmax=lmax)[field - 1]
            else:
                alm = self.hp.map2alm(pol_map[1:], lmax, self.nthreads)[field - 1]
            if self.deconvolve:
                self._apply_common_beam(alm, i, lmax)
            return alm

        if self.parallel >= 1:
            with ThreadPoolExecutor(max_workers=len(maps)) as executor:
                alms = list(tqdm(executor.map(process_map, range(len(maps))),
                                 desc="Computing alms", total=len(maps), disable=not self.progress_bar))
        else:
            alms = [process_map(i) for i in tqdm(range(len(maps)), desc="Computing alms", disable=not self.progress_bar)]
        return np.array(alms)

    def map_to_alm(self, maps: List[np.ndarray], field: int) -> np.ndarray:
        """
        Convert maps to spherical harmonic coefficients (alms).

        Parameters
        ----------
        maps : List[np.ndarray]
            List of maps.
        field : int
            Field index (0 for temperature, 1 for E-mode, 2 for B-mode).

        Returns
        -------
        np.ndarray
            Array of spherical harmonic coefficients.
        """
        if field == 0:
            return self._map_to_alm_temperature(maps)
        return self._map_to_alm_polarization(maps, field)
    
    
    def compute_covariance_fwhm_nmodes(self):

        if not isinstance(self.needlet_filters, np.ndarray) or len(self.needlet_filters.shape) != 2:
            raise ValueError("self.needlet_filters must be a 2D NumPy array")
        nlmax = self.needlet_filters.shape[1] - 1
        l = np.arange(nlmax + 1)
        
        fwhm_results = []
        n_tot = 12 * self.nside ** 2
        theta_pix = np.pi / (np.sqrt(3) * self.nside)

        for b in range(self.needlet_filters.shape[0]):
            needlet_band = self.needlet_filters[b, :]
            
            total_num_modes = np.sum((2 * l + 1) * needlet_band**2)
            if total_num_modes < self.required_num_modes:
                raise ValueError(f"Band {b} has insufficient modes ({total_num_modes}) for {self.required_num_modes}")

            mode_fraction = self.required_num_modes / total_num_modes

            n_p_min = n_tot * mode_fraction
            
            fwhm_rad = theta_pix * np.sqrt(n_p_min / np.pi)
            fwhm_deg = fwhm_rad * 180 / np.pi
            
            ell_peak = np.argmax(needlet_band)
            scaling_factor = max(1.0, (nlmax - ell_peak) / (nlmax - l[0]))
            fwhm_deg = fwhm_deg * scaling_factor
            
            fwhm_results.append(fwhm_deg) #np.clip(fwhm_deg, 30, 200)
        
        self.fwhm = np.array(fwhm_results)

    
    def compute_covariance_fwhm_tol(self):
        if not isinstance(self.needlet_filters, np.ndarray) or len(self.needlet_filters.shape) != 2:
            raise ValueError("self.needlet_filters must be a 2D NumPy array")

        nlmax = len(self.needlet_filters[0]) - 1
        l = np.arange(nlmax + 1)
        fwhm_results = []

        if not hasattr(self, 'frequencies') or not hasattr(self, 'ilc_bias'):
            raise AttributeError("Attributes 'frequencies' and 'ilc_bias' must be defined")

        nchan = len(self.frequencies)

        for i in range(self.needlet_filters.shape[0]):
            nside_i = self.needlet_nside[i]
            needlet_band = self.needlet_filters[i, :]
            nmodes_band = np.sum((2 * l + 1) * needlet_band**2)

            if nmodes_band <= 0:
                raise ValueError(f"Band {i} has non-positive number of modes: {nmodes_band}")
            if self.ilc_bias <= 0:
                raise ValueError("ilc_bias must be positive")

            npix = hp.nside2npix(nside_i)

            # Number of pixels required to achieve the bias tolerance
            pps = np.sqrt((npix * (nchan - 1)) / (self.ilc_bias * nmodes_band))

            # Pixel size in radians
            pix_size_rad = np.sqrt(4.0 * np.pi / npix)

            # Statistical FWHM in radians
            fwhm_rad = pps * pix_size_rad

            # Convert FWHM to degrees
            fwhm_deg = np.rad2deg(fwhm_rad)

            fwhm_results.append(fwhm_deg)

        self.fwhm = np.array(fwhm_results)

    def compute_covariance_fwhm(self):
        """
        Compute the FWHM for each needlet filter based on the specified bias method.
        """
        if self.bias_method == 'nmodes':
            self.compute_covariance_fwhm_nmodes()
        elif self.bias_method == 'tol':
            self.compute_covariance_fwhm_tol()
        else:
            raise ValueError("Invalid bias method. Use 'nmodes' or 'tol'.")

    def find_scale(self, j: int) -> np.ndarray:
        """
        Compute the needlet scale for the j-th needlet filter in parallel.

        Parameters
        ----------
        j : int
            Index of the needlet filter.

        Returns
        -------
        np.ndarray
            Array of maps after applying the needlet filter.
        """
        nside = self.needlet_nside[j]
        loc_lmax = self.lmax_per_needlet[j]

        def process_alm(alm):
            # Apply the needlet filter
            filtered_alm = hp.almxfl(alm, self.needlet_filters[j])
            # Generate the filtered map
            if self.use_healpy:
                return hp.alm2map(filtered_alm, nside)
            else:
                loc_hp = HealpixDUCC(nside)
                filtered_alm = slice_alms(np.array([filtered_alm]), loc_lmax)[0]
                return loc_hp.alm2map(filtered_alm, loc_lmax, self.nthreads)

        # Use ThreadPoolExecutor to parallelize the loop

        if self.parallel >= 1:
            with ThreadPoolExecutor(max_workers=len(self.alms)) as executor:
                scale_maps = list(executor.map(process_alm, self.alms))
        else:
            scale_maps = [process_alm(alm) for alm in self.alms]

        return np.array(scale_maps)
    
    def find_weights(self, j: int, scale: np.ndarray) -> np.ndarray:
        """
        Compute the ILC weights for the j-th needlet filter in parallel.

        Parameters
        ----------
        j : int
            Index of the needlet filter.
        scale : np.ndarray
            Scale maps for the j-th needlet filter.

        Returns
        -------
        np.ndarray
            Array of ILC weights.
        """
        nside = self.needlet_nside[j]
        hp_loc = HealpixDUCC(nside)
        num_maps = scale.shape[0]
        one_vec = np.ones(num_maps)
        num_pixels = hp.nside2npix(nside)

        # Shared covariance matrix initialized with zeros
        covariance = np.zeros((num_pixels, num_maps, num_maps))

        def compute_covariance(i, k, j):
            # nside = max(1, self.needlet_nside[j]//4)
            # product = hp.ud_grade(scale[i] * scale[k],nside_out=nside,order_out="RING")
            # lmax = 2 * nside
            # fwhm_stat = np.radians(self.fwhm[j])
            # bl_stat = hp.gauss_beam(fwhm_stat, lmax=lmax)
            # thetas = np.arange(0,np.pi,0.002)
            # beam_stat = hp.bl2beam(bl_stat, thetas)
            # theta_ = 0.5 * fwhm_stat
            # dist = 0.5 * (np.tanh(30 * (thetas - 0.3 * theta_)) - np.tanh(30 * (thetas - 3 * theta_)))
            # dist /= np.max(dist)
            # dist[np.argmax(dist):]=1.
            # bl_stat = hp.beam2bl(dist * beam_stat,thetas, lmax = lmax)
            # alm_s = hp.map2alm(product, lmax=lmax,iter=1, use_weights=True)
            # alm_s = hp.almxfl(alm_s, bl_stat)
            # smoothed = hp.alm2map(alm_s, self.needlet_nside[j], lmax=lmax)


            product = scale[i] * scale[k]
            if self.use_healpy:
                product *= hp.ud_grade(self.mask, nside_out=self.needlet_nside[j], order_out="RING",power=-2)
                smoothed = hp.smoothing(product, np.radians(self.fwhm[j]))
            else:
                nside = self.needlet_nside[j]
                lmax = 3 * nside - 1
                product_alm = hp_loc.map2alm(product, lmax, self.nthreads)
                lmax = self.lmax_per_needlet[j]
                product_alm = slice_alms(np.array([product_alm]), lmax)[0]
                bl = hp.gauss_beam(np.radians(self.fwhm[j]), lmax=lmax)
                product_alm = hp.almxfl(product_alm, bl)
                smoothed = hp_loc.alm2map(product_alm, lmax, self.nthreads)
            
   
            return i, k, smoothed

        # Parallelize the computation of covariance elements
        if self.parallel >= 1:
            with ThreadPoolExecutor(max_workers=self.nthreads) as executor:
                futures = []
                for i in range(num_maps):
                    for k in range(i, num_maps):
                        futures.append(executor.submit(compute_covariance, i, k, j))

                for future in futures:
                    i, k, smoothed = future.result()
                    covariance[:, i, k] = smoothed
                    if i != k:
                        covariance[:, k, i] = smoothed
        else:
            for i in range(num_maps):
                for k in range(i, num_maps):
                    _, _, smoothed = compute_covariance(i, k, j)
                    covariance[:, i, k] = smoothed
                    if i != k:
                        covariance[:, k, i] = smoothed

        # Invert the covariance matrix
        inv_cov = np.linalg.inv(covariance)

        # Compute weights
        weights = (inv_cov @ one_vec).T / (one_vec @ inv_cov @ one_vec )
        local_mask = hp.ud_grade(self.mask, nside_out=self.needlet_nside[j], order_out="RING", power=-2)
        weights[:,local_mask == 0] = 0.0  # Set weights to zero where mask is zero

        return weights
    
    
    def component_separation(self, maps: List[np.ndarray], field: int=0) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Perform component separation using NILC in parallel.

        Parameters
        ----------
        maps : List[np.ndarray]
            List of input maps.
        field : int
            Field index (0 for temperature, 1 for E-mode, 2 for B-mode).

        Returns
        -------
        Tuple[np.ndarray, List[np.ndarray]]
            Final NILC-separated map and list of ILC weights for each needlet scale.
        """
        self.alms = self.map_to_alm(maps, field)
        self.compute_covariance_fwhm()
        nside = hp.get_nside(maps[0] if maps[0].ndim == 1 else maps[0][0])

        def process_needlet(j):
            scale = self.find_scale(j)
            weights = self.find_weights(j, scale)
            result_map = np.sum(scale * weights, axis=0)
            if self.use_healpy:
                filtered_alm = hp.map2alm(result_map)
                filtered_alm = hp.almxfl(filtered_alm, self.needlet_filters[j])
                filtered_map = hp.alm2map(filtered_alm, nside)
            else:
                loc_hp = HealpixDUCC(self.needlet_nside[j])
                loc_lmax = self.lmax_per_needlet[j]
                filtered_alm = loc_hp.map2alm(result_map, loc_lmax, self.nthreads)
                filtered_alm = hp.almxfl(filtered_alm, self.needlet_filters[j])
                filtered_map = self.hp.alm2map(filtered_alm, self.lmax_per_needlet[j], self.nthreads)
            return filtered_map, weights

        # Parallelize the processing of needlet scales
        ilc_map = np.zeros(hp.nside2npix(nside))
        weight_list = []

        if self.parallel >= 2:
            with ThreadPoolExecutor(max_workers=self.nthreads) as executor:
                futures = list(tqdm(executor.map(process_needlet, range(self.num_needlets)), desc="Processing Needlet Scales", total=self.num_needlets, disable=not self.progress_bar))
        else:
            futures = [process_needlet(j) for j in tqdm(range(self.num_needlets), desc="Processing Needlet Scales", disable=not self.progress_bar)]

        for filtered_map, weights in futures:
            ilc_map += filtered_map
            weight_list.append(weights)

        # Clear temporary variables for garbage collection
        self.alms = None
        gc.collect()

        return ilc_map, weight_list

    def calculate_residuals(self, maps: List[np.ndarray], weight_list: List[np.ndarray], field: int) -> np.ndarray:
        """
        Calculate the residuals using provided weights.

        Parameters
        ----------
        maps : List[np.ndarray]
            List of input maps.
        weight_list : List[np.ndarray]
            List of weights for each needlet scale.
        field : int
            Field index (0 for temperature, 1 for E-mode, 2 for B-mode).

        Returns
        -------
        np.ndarray
            Map of residuals.
        """
        self.alms = self.map_to_alm(maps, field)

        residual_map = 0

        for j in tqdm(range(self.num_needlets), desc="Calculating residuals", disable=not self.progress_bar):
            scale = self.find_scale(j)
            weights = weight_list[j]
            residuals = np.sum(scale * weights, axis=0)
            filtered_alm = hp.map2alm(residuals)
            filtered_alm = hp.almxfl(filtered_alm, self.needlet_filters[j])
            residual_map += hp.alm2map(filtered_alm, hp.get_nside(maps[0][0]))

        # Clear temporary variables for garbage collection
        self.alms = None
        #gc.collect()
        return residual_map