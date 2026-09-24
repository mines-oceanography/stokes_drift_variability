import numpy as np
import xarray as xr
import xrft


def helmholtz_decomposition(u, v):
    """
    Computes the Helmholtz decomposition of the u and v field using the specral method from _____.
    Returns the
    """

    # 1. FFT of u and v
    print("Computing FFTs...")
    U_fft = xrft.fft(u, dim=['x', 'y'], 
                     shift=True, true_amplitude=True,
                     detrend='linear', window='hann', window_correction=False
                     ).rename({"freq_x": "kx", "freq_y": "ky"})
    V_fft = xrft.fft(v, dim=['x', 'y'],
                     shift=True, true_amplitude=True,
                     detrend='linear', window='hann', window_correction=False
                     ).rename({"freq_x": "kx", "freq_y": "ky"})
    # Note: window_correction=True won't actually apply a window correction;
    # It is set to False here for clarity. The window correction is applied manually later.

    # 2. Helmholtz decomposition
    print("Performing Helmholtz decomposition...")
    kx = U_fft['kx']
    ky = U_fft['ky']
    KK, LL = np.meshgrid(kx, ky)

    THETA = np.arctan2(LL, KK)
    div_fft = U_fft * np.cos(THETA) + V_fft * np.sin(THETA) # P, potential, divergent, irrotatial, phi
    rot_fft = -U_fft * np.sin(THETA) + V_fft * np.cos(THETA) # S, solenoidal, vortical, rotational, psi

    # 3. Normalize to KE PSD with window correction
    print("Calculating KE PSDs...")
    C = 0.140625 # window correctionf factor for a 2D Hann window; set manually here for efficiency

    dkx, dky = float(U_fft.kx[1] - U_fft.kx[0]), float(U_fft.ky[1] - U_fft.ky[0])
    
    div_KE_psd = (np.abs(div_fft) ** 2) * dkx * dky / (2 * C)
    rot_KE_psd = (np.abs(rot_fft) ** 2) * dkx * dky / (2 * C)

    # 4. KE PSD
    KE_psd = div_KE_psd + rot_KE_psd

    # 5. Pack into dataset
    print("Packing results into dataset...")
    ds = xr.Dataset({
        "KE_psd": KE_psd,
        "div_KE_psd": div_KE_psd,
        "rot_KE_psd": rot_KE_psd,
        "div_fft": div_fft,
        "rot_fft": rot_fft,
        "U_fft": U_fft,
        "V_fft": V_fft,
    })

    print("Done.")
    return ds