"""
LibRadTran Interface for Thermal Atmospheric Correction

This module provides a Python interface to LibRadTran for computing
atmospheric transmittance and TOA brightness temperatures for
Landsat 8 Band 10 thermal imagery.

This is an alternative to MODTRAN for users without MODTRAN licenses.

Author: Generated for Landsat SST Algorithm
"""

import os
import subprocess
import tempfile
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List, Dict
import warnings


class LibRadTranThermal:
    """
    Interface for LibRadTran thermal radiative transfer calculations.

    This class provides methods to:
    - Convert ERA5 atmospheric profiles to LibRadTran format
    - Run LibRadTran uvspec for thermal calculations
    - Extract transmittance and TOA brightness temperature
    - Process batches of profiles to generate MODTRAN-compatible output

    Parameters
    ----------
    libradtran_dir : str, optional
        Path to LibRadTran installation directory.
        If not provided, uses LIBRADTRAN_DIR environment variable.
    srf_file : str, optional
        Path to spectral response function file for Landsat 8 Band 10.
        Default: 'Data/SRF/landsat8_band10_srf.txt'
    surface_emissivity : float, optional
        Ocean surface emissivity in thermal IR. Default: 0.99

    Example
    -------
    >>> lrt = LibRadTranThermal()
    >>> toa_bt, tau, jacobian = lrt.process_profile(
    ...     altitude_km, pressure_hpa, temperature_k, spec_humidity,
    ...     surface_temp_k=280.0
    ... )
    """

    # Physical constants
    C1 = 1.19104e8   # W um^4 / (m^2 sr) - first radiation constant
    C2 = 14387.77    # um K - second radiation constant
    LAMBDA_C = 10.9  # um - Band 10 center wavelength
    G = 9.80665      # m/s^2 - standard gravity
    M_AIR = 28.97    # g/mol - molecular weight of dry air
    M_H2O = 18.015   # g/mol - molecular weight of water

    def __init__(
        self,
        libradtran_dir: Optional[str] = None,
        srf_file: Optional[str] = None,
        surface_emissivity: float = 0.99
    ):
        # Set LibRadTran directory
        if libradtran_dir is None:
            libradtran_dir = os.environ.get('LIBRADTRAN_DIR')

        if libradtran_dir is None:
            raise ValueError(
                "LibRadTran directory not specified. "
                "Set LIBRADTRAN_DIR environment variable or pass libradtran_dir parameter."
            )

        self.libradtran_dir = Path(libradtran_dir)
        self.uvspec_bin = self.libradtran_dir / 'bin' / 'uvspec'
        self.data_dir = self.libradtran_dir / 'share' / 'libRadtran' / 'data'

        # Verify installation
        if not self.uvspec_bin.exists():
            raise FileNotFoundError(
                f"uvspec binary not found at {self.uvspec_bin}. "
                "Check LibRadTran installation."
            )

        # Set SRF file path
        if srf_file is None:
            # Default relative to this module
            module_dir = Path(__file__).parent
            srf_file = module_dir / 'Data' / 'SRF' / 'landsat8_band10_srf.txt'

        self.srf_file = Path(srf_file)
        if not self.srf_file.exists():
            raise FileNotFoundError(
                f"SRF file not found at {self.srf_file}. "
                "Download Landsat 8 Band 10 SRF data."
            )

        self.surface_emissivity = surface_emissivity

    def convert_specific_humidity_to_ppm(
        self,
        q_kg_kg: np.ndarray,
        p_hpa: np.ndarray = None,
        T_k: np.ndarray = None
    ) -> np.ndarray:
        """
        Convert specific humidity to water vapor mixing ratio in ppm.

        Parameters
        ----------
        q_kg_kg : array-like
            Specific humidity [kg H2O / kg moist air]
        p_hpa : array-like, optional
            Pressure [hPa] (not used in this conversion but kept for API compatibility)
        T_k : array-like, optional
            Temperature [K] (not used in this conversion but kept for API compatibility)

        Returns
        -------
        ppm : ndarray
            Water vapor mixing ratio [ppmv]
        """
        q = np.asarray(q_kg_kg)
        # Convert specific humidity to mixing ratio
        # r = q / (1 - q) where r is kg H2O / kg dry air
        r = q / (1 - q)
        # Convert to ppmv: multiply by ratio of molecular weights and 1e6
        ppm = r * (self.M_AIR / self.M_H2O) * 1e6
        return ppm

    def write_atmosphere_file(
        self,
        filepath: Path,
        altitude_km: np.ndarray,
        pressure_hpa: np.ndarray,
        temperature_k: np.ndarray,
        h2o_ppm: np.ndarray
    ):
        """
        Write atmospheric profile in LibRadTran atmosphere_file format.

        LibRadTran atmosphere file format (same as standard atm files like afglus.dat):
        - Column 1: altitude [km]
        - Column 2: pressure [hPa/mb]
        - Column 3: temperature [K]
        - Column 4: air number density [cm^-3]
        - Column 5: O3 [cm^-3] (placeholder, use standard profile)
        - Column 6: O2 [cm^-3] (placeholder, use standard profile)
        - Column 7: H2O [cm^-3]
        - Column 8: CO2 [cm^-3] (placeholder, use standard profile)
        - Column 9: NO2 [cm^-3] (placeholder, use standard profile)

        Altitude must be in DESCENDING order (from TOA to surface).

        Parameters
        ----------
        filepath : Path
            Output file path
        altitude_km : array-like
            Altitude [km]
        pressure_hpa : array-like
            Pressure [hPa]
        temperature_k : array-like
            Temperature [K]
        h2o_ppm : array-like
            Water vapor mixing ratio [ppmv]
        """
        alt = np.asarray(altitude_km)
        p = np.asarray(pressure_hpa)
        T = np.asarray(temperature_k)
        h2o = np.asarray(h2o_ppm)

        # LibRadTran expects altitude in DESCENDING order (TOA first)
        # Our MODTRAN data comes with altitude descending (TOA first), so no reversal needed
        # But verify and reverse if needed
        if alt[0] < alt[-1]:
            # Altitude is ascending (surface first), reverse to get TOA first
            alt = alt[::-1]
            p = p[::-1]
            T = T[::-1]
            h2o = h2o[::-1]

        # Calculate air number density from ideal gas law
        # Using Boltzmann constant k = 1.380649e-23 J/K
        # n_air [m^-3] = p[Pa] / (k * T)
        # n_air [cm^-3] = p[Pa] / (k * T) / 1e6
        #              = p[hPa] * 100 / (1.380649e-23 * T) / 1e6
        #              = p[hPa] * 7.2429e18 / T
        n_air = p * 7.2429e18 / T  # cm^-3

        # Convert H2O from ppm to number density [cm^-3]
        # h2o_density = n_air * h2o_ppm / 1e6
        h2o_density = n_air * h2o / 1e6  # cm^-3

        # Estimate O2 as 20.95% of air
        n_o2 = n_air * 0.2095

        # Use placeholder values for other gases (these will come from standard atmosphere)
        # O3: use typical mid-latitude values scaled by altitude
        # CO2: ~420 ppm
        n_co2 = n_air * 420e-6

        with open(filepath, 'w') as f:
            # Write header comment
            f.write("# Custom atmosphere profile for LibRadTran\n")
            f.write("# z(km)      p(mb)        T(K)    air(cm-3)    o3(cm-3)     o2(cm-3)    h2o(cm-3)    co2(cm-3)     no2(cm-3)\n")

            for i in range(len(alt)):
                # Estimate O3 based on typical profile (peak around 25 km)
                # This is a rough approximation; actual O3 will come from climatology
                if alt[i] > 20:
                    o3_vmr = 5e-6  # 5 ppm in stratosphere
                elif alt[i] > 10:
                    o3_vmr = 1e-6  # 1 ppm in upper troposphere
                else:
                    o3_vmr = 3e-8  # 30 ppb in lower troposphere
                n_o3 = n_air[i] * o3_vmr

                # NO2: very small, use placeholder
                n_no2 = n_air[i] * 1e-9  # ~1 ppb

                f.write(f"{alt[i]:10.3f} {p[i]:12.5e} {T[i]:10.2f} {n_air[i]:12.5e} "
                        f"{n_o3:12.5e} {n_o2[i]:12.5e} {h2o_density[i]:12.5e} "
                        f"{n_co2[i]:12.5e} {n_no2:12.5e}\n")

    def create_input_file(
        self,
        filepath: Path,
        atmosphere_file: Path,
        surface_temp_k: float,
        wavelength_min_nm: float = 8000,
        wavelength_max_nm: float = 16000,
        solver: str = 'disort'
    ):
        """
        Create LibRadTran uvspec input file for thermal calculation.

        Parameters
        ----------
        filepath : Path
            Output input file path
        atmosphere_file : Path
            Path to atmosphere profile file (same format as afglus.dat)
        surface_temp_k : float
            Surface temperature [K]
        wavelength_min_nm : float
            Minimum wavelength [nm]. Default: 8000 (8 µm)
        wavelength_max_nm : float
            Maximum wavelength [nm]. Default: 16000 (16 µm)
        solver : str
            Radiative transfer solver. 'disort' (accurate) or 'twostr' (fast)
        """
        with open(filepath, 'w') as f:
            # Data files path
            f.write(f"data_files_path {self.data_dir}\n")
            f.write("\n")

            # Atmospheric profile - use atmosphere_file format like standard profiles
            f.write(f"atmosphere_file {atmosphere_file}\n")
            f.write("\n")

            # Thermal source (no solar)
            f.write("source thermal\n")
            f.write("\n")

            # Surface properties - use Lambertian surface with emissivity
            f.write(f"albedo {1.0 - self.surface_emissivity:.4f}\n")
            f.write(f"sur_temperature {surface_temp_k:.3f}\n")
            f.write("\n")

            # Molecular absorption - use REPTRAN coarse for thermal
            # (medium resolution is missing lookup files for thermal)
            f.write("mol_abs_param reptran coarse\n")
            f.write("\n")

            # Radiative transfer solver
            f.write(f"rte_solver {solver}\n")
            f.write("\n")

            # Wavelength range (in nm for uvspec)
            f.write(f"wavelength {wavelength_min_nm:.0f} {wavelength_max_nm:.0f}\n")
            f.write("\n")

            # Apply Landsat 8 Band 10 SRF for band integration
            f.write(f"filter_function_file {self.srf_file}\n")
            f.write("\n")

            # Output brightness temperature directly
            f.write("output_quantity brightness\n")
            f.write("output_user lambda uu\n")
            f.write("output_process integrate\n")
            f.write("\n")

            # Viewing geometry (nadir looking up from surface toward sensor)
            f.write("umu 1.0\n")  # Upward radiance at TOA
            f.write("phi 0\n")
            f.write("\n")

            # Output at TOA
            f.write("zout toa\n")
            f.write("\n")

            # Quiet output
            f.write("quiet\n")

    def run_uvspec(
        self,
        input_file: Path,
        timeout: int = 120
    ) -> str:
        """
        Execute LibRadTran uvspec and return output.

        Parameters
        ----------
        input_file : Path
            Path to uvspec input file
        timeout : int
            Timeout in seconds. Default: 120

        Returns
        -------
        output : str
            uvspec stdout output
        """
        try:
            result = subprocess.run(
                [str(self.uvspec_bin)],
                stdin=open(input_file),
                capture_output=True,
                text=True,
                timeout=timeout
            )

            if result.returncode != 0:
                raise RuntimeError(
                    f"uvspec failed with return code {result.returncode}:\n"
                    f"stderr: {result.stderr}\n"
                    f"stdout: {result.stdout}"
                )

            return result.stdout

        except subprocess.TimeoutExpired:
            raise RuntimeError(f"uvspec timed out after {timeout}s")
        except FileNotFoundError:
            raise RuntimeError(f"uvspec not found at {self.uvspec_bin}")

    def radiance_to_brightness_temp(self, radiance_mw: float, wavelength_um: float = None) -> float:
        """
        Convert spectral radiance to brightness temperature using inverse Planck function.

        Parameters
        ----------
        radiance_mw : float
            Spectral radiance in mW/(m^2 sr nm)
        wavelength_um : float, optional
            Center wavelength in µm. Default: LAMBDA_C (10.9 µm)

        Returns
        -------
        brightness_temp : float
            Brightness temperature [K]
        """
        if wavelength_um is None:
            wavelength_um = self.LAMBDA_C

        # Convert radiance from mW/(m^2 sr nm) to W/(m^2 sr µm)
        # mW -> W: divide by 1000
        # per nm -> per µm: multiply by 1000
        # Net effect: no change in numerical value
        radiance_w = radiance_mw  # W/(m^2 sr µm)

        # Inverse Planck function: T = C2 / (λ * ln(C1/(λ^5 * L) + 1))
        # C1 in W µm^4 / (m^2 sr), C2 in µm K
        try:
            arg = self.C1 / (wavelength_um**5 * radiance_w) + 1
            if arg <= 1:
                return np.nan
            T = self.C2 / (wavelength_um * np.log(arg))
            return T
        except (ValueError, ZeroDivisionError):
            return np.nan

    def parse_radiance_output(self, output: str) -> float:
        """
        Parse uvspec radiance output and convert to brightness temperature.
        (Legacy function - kept for compatibility)
        """
        return self.parse_brightness_output(output)

    def parse_brightness_output(self, output: str) -> float:
        """
        Parse uvspec brightness temperature output.

        Parameters
        ----------
        output : str
            uvspec stdout

        Returns
        -------
        brightness_temp : float
            TOA brightness temperature [K]
        """
        lines = output.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                # Format: wavelength  brightness_temp
                parts = line.split()
                if len(parts) >= 2:
                    brightness_temp = float(parts[1])
                    return brightness_temp

        raise ValueError(f"Could not parse brightness temperature from output:\n{output}")

    def calculate_transmittance(
        self,
        surface_temp_k: float,
        toa_bt_k: float
    ) -> float:
        """
        Estimate atmospheric transmittance from surface and TOA temperatures.

        Uses the Planck function ratio at Band 10 center wavelength.
        This is an approximation that assumes negligible atmospheric emission
        contribution (valid for high transmittance conditions).

        Parameters
        ----------
        surface_temp_k : float
            Surface temperature [K]
        toa_bt_k : float
            TOA brightness temperature [K]

        Returns
        -------
        transmittance : float
            Atmospheric transmittance (0-1)
        """
        # Planck function at center wavelength
        def planck_radiance(T, wavelength_um=self.LAMBDA_C):
            return self.C1 / (wavelength_um**5 * (np.exp(self.C2 / (wavelength_um * T)) - 1))

        L_surface = planck_radiance(surface_temp_k)
        L_toa = planck_radiance(toa_bt_k)

        # tau = L_toa / (epsilon * L_surface)
        tau = L_toa / (self.surface_emissivity * L_surface)

        # Clamp to valid range
        return np.clip(tau, 0.0, 1.0)

    def calculate_transmittance_direct(
        self,
        altitude_km: np.ndarray,
        pressure_hpa: np.ndarray,
        temperature_k: np.ndarray,
        h2o_ppm: np.ndarray,
        surface_temp_k: float
    ) -> float:
        """
        Calculate transmittance directly by running LibRadTran twice.

        Runs with surface at two different temperatures and uses the
        difference to estimate transmittance more accurately.

        Parameters
        ----------
        altitude_km, pressure_hpa, temperature_k, h2o_ppm : array-like
            Atmospheric profile
        surface_temp_k : float
            Nominal surface temperature [K]

        Returns
        -------
        transmittance : float
            Atmospheric transmittance
        """
        # Run at nominal temperature
        bt1 = self._run_single_profile(
            altitude_km, pressure_hpa, temperature_k, h2o_ppm,
            surface_temp_k
        )

        # Run at elevated temperature
        delta_T = 1.0  # K
        bt2 = self._run_single_profile(
            altitude_km, pressure_hpa, temperature_k, h2o_ppm,
            surface_temp_k + delta_T
        )

        # dBT/dSST ≈ tau * epsilon for thermal
        jacobian = (bt2 - bt1) / delta_T
        tau = jacobian / self.surface_emissivity

        return np.clip(tau, 0.0, 1.0)

    def _run_single_profile(
        self,
        altitude_km: np.ndarray,
        pressure_hpa: np.ndarray,
        temperature_k: np.ndarray,
        h2o_ppm: np.ndarray,
        surface_temp_k: float
    ) -> float:
        """Run LibRadTran for a single profile and return TOA brightness temp."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Write atmosphere file
            atmosphere_file = tmpdir / 'profile.dat'
            self.write_atmosphere_file(
                atmosphere_file,
                altitude_km, pressure_hpa, temperature_k, h2o_ppm
            )

            # Create input file
            input_file = tmpdir / 'uvspec.inp'
            self.create_input_file(
                input_file,
                atmosphere_file,
                surface_temp_k
            )

            # Run uvspec
            output = self.run_uvspec(input_file)

            # Parse output
            return self.parse_brightness_output(output)

    def process_profile(
        self,
        altitude_km: np.ndarray,
        pressure_hpa: np.ndarray,
        temperature_k: np.ndarray,
        spec_humidity: np.ndarray,
        surface_temp_k: float,
        compute_jacobian: bool = True
    ) -> Tuple[float, float, float]:
        """
        Process a single atmospheric profile through LibRadTran.

        Parameters
        ----------
        altitude_km : array-like
            Altitude [km]
        pressure_hpa : array-like
            Pressure [hPa]
        temperature_k : array-like
            Temperature [K]
        spec_humidity : array-like
            Specific humidity [kg/kg]
        surface_temp_k : float
            Surface temperature [K]
        compute_jacobian : bool
            If True, compute dBT/dSST via finite difference (slower).
            If False, estimate from transmittance.

        Returns
        -------
        toa_bt : float
            TOA brightness temperature [K]
        transmittance : float
            Atmospheric transmittance
        jacobian : float
            dBT/dSST (brightness temperature sensitivity)
        """
        # Convert specific humidity to ppm
        h2o_ppm = self.convert_specific_humidity_to_ppm(spec_humidity)

        # Run at nominal surface temperature
        toa_bt = self._run_single_profile(
            altitude_km, pressure_hpa, temperature_k, h2o_ppm,
            surface_temp_k
        )

        if compute_jacobian:
            # Run at elevated temperature for Jacobian
            delta_T = 1.0
            toa_bt_plus = self._run_single_profile(
                altitude_km, pressure_hpa, temperature_k, h2o_ppm,
                surface_temp_k + delta_T
            )
            jacobian = (toa_bt_plus - toa_bt) / delta_T
            transmittance = jacobian / self.surface_emissivity
        else:
            # Estimate transmittance from brightness temperatures
            transmittance = self.calculate_transmittance(surface_temp_k, toa_bt)
            jacobian = transmittance * self.surface_emissivity

        return toa_bt, np.clip(transmittance, 0.0, 1.0), jacobian

    def process_month(
        self,
        atm_profiles_file: Path,
        sst_profiles_file: Path,
        output_file: Path,
        n_levels: int = 37,
        compute_jacobian: bool = True,
        verbose: bool = True
    ):
        """
        Process all atmospheric profiles for a month.

        Reads input files in the existing MODTRAN format and produces
        output in MODTRAN-compatible format.

        Parameters
        ----------
        atm_profiles_file : Path
            Input atmospheric profiles file (modtran_atmprofiles_XX.txt format)
        sst_profiles_file : Path
            Input SST profiles file (modtran_sstprofiles_XX.txt format)
        output_file : Path
            Output file path (MODTRAN output format)
        n_levels : int
            Number of atmospheric levels per profile. Default: 37
        compute_jacobian : bool
            If True, compute exact Jacobian. Default: True
        verbose : bool
            Print progress. Default: True
        """
        import pandas as pd

        # Read atmospheric profiles
        atm_data = pd.read_csv(
            atm_profiles_file,
            sep='\t',
            header=None,
            names=['altitude_km', 'pressure_hpa', 'temp_k', 'spec_humidity']
        )

        # Read SST profiles
        sst_data = pd.read_csv(
            sst_profiles_file,
            sep='\t',
            header=None,
            names=['sst_celsius']
        )

        n_atm_profiles = len(atm_data) // n_levels
        n_sst = len(sst_data)

        # Use minimum of atm profiles and SST values
        n_profiles = min(n_atm_profiles, n_sst)
        if n_atm_profiles != n_sst:
            warnings.warn(
                f"Count mismatch: {n_atm_profiles} atm profiles, {n_sst} SST values. "
                f"Processing {n_profiles} profiles."
            )

        if verbose:
            print(f"Processing {n_profiles} profiles...")

        results = []

        for i in range(n_profiles):
            # Extract profile
            start_idx = i * n_levels
            end_idx = start_idx + n_levels
            profile = atm_data.iloc[start_idx:end_idx]

            # Get SST (convert Celsius to Kelvin)
            sst_c = sst_data.iloc[i]['sst_celsius']
            surface_temp_k = sst_c + 273.15

            try:
                toa_bt, tau, jacobian = self.process_profile(
                    profile['altitude_km'].values,
                    profile['pressure_hpa'].values,
                    profile['temp_k'].values,
                    profile['spec_humidity'].values,
                    surface_temp_k,
                    compute_jacobian=compute_jacobian
                )

                results.append({
                    'wind_spd': 0,
                    'surface_t': surface_temp_k,
                    'toa_t': toa_bt,
                    'transmittance': tau,
                    'jacobian': jacobian
                })

            except Exception as e:
                warnings.warn(f"Profile {i} failed: {e}")
                results.append({
                    'wind_spd': 0,
                    'surface_t': surface_temp_k,
                    'toa_t': np.nan,
                    'transmittance': np.nan,
                    'jacobian': np.nan
                })

            if verbose and (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{n_profiles} profiles")

        # Write output in MODTRAN format
        with open(output_file, 'w') as f:
            for r in results:
                f.write(
                    f"{r['wind_spd']:02d} "
                    f"{r['surface_t']:.3f} "
                    f"{r['toa_t']:.3f} "
                    f"{r['transmittance']:.4f} "
                    f"{r['jacobian']:.4f}\n"
                )

        if verbose:
            print(f"Output written to: {output_file}")

        return results

    def process_month_parallel(
        self,
        atm_profiles_file: Path,
        sst_profiles_file: Path,
        output_file: Path,
        n_levels: int = 37,
        compute_jacobian: bool = True,
        n_workers: int = 6,
        verbose: bool = True
    ):
        """
        Process all atmospheric profiles for a month using parallel workers.

        Parameters
        ----------
        atm_profiles_file : Path
            Input atmospheric profiles file (modtran_atmprofiles_XX.txt format)
        sst_profiles_file : Path
            Input SST profiles file (modtran_sstprofiles_XX.txt format)
        output_file : Path
            Output file path (MODTRAN output format)
        n_levels : int
            Number of atmospheric levels per profile. Default: 37
        compute_jacobian : bool
            If True, compute exact Jacobian. Default: True
        n_workers : int
            Number of parallel workers. Default: 6
        verbose : bool
            Print progress. Default: True
        """
        import pandas as pd
        from concurrent.futures import ProcessPoolExecutor, as_completed

        # Read atmospheric profiles
        atm_data = pd.read_csv(
            atm_profiles_file,
            sep='\t',
            header=None,
            names=['altitude_km', 'pressure_hpa', 'temp_k', 'spec_humidity']
        )

        # Read SST profiles
        sst_data = pd.read_csv(
            sst_profiles_file,
            sep='\t',
            header=None,
            names=['sst_celsius']
        )

        n_atm_profiles = len(atm_data) // n_levels
        n_sst = len(sst_data)

        # Use minimum of atm profiles and SST values
        n_profiles = min(n_atm_profiles, n_sst)
        if n_atm_profiles != n_sst:
            warnings.warn(
                f"Count mismatch: {n_atm_profiles} atm profiles, {n_sst} SST values. "
                f"Processing {n_profiles} profiles."
            )

        if verbose:
            print(f"Processing {n_profiles} profiles with {n_workers} workers...")

        # Prepare profile data for parallel processing
        profile_args = []
        for i in range(n_profiles):
            start_idx = i * n_levels
            end_idx = start_idx + n_levels
            profile = atm_data.iloc[start_idx:end_idx]

            sst_c = sst_data.iloc[i]['sst_celsius']
            surface_temp_k = sst_c + 273.15

            profile_args.append((
                i,
                profile['altitude_km'].values,
                profile['pressure_hpa'].values,
                profile['temp_k'].values,
                profile['spec_humidity'].values,
                surface_temp_k,
                compute_jacobian,
                str(self.libradtran_dir),
                str(self.srf_file),
                self.surface_emissivity
            ))

        # Process in parallel
        results = [None] * n_profiles
        completed = 0

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(_process_single_profile_worker, args): args[0]
                for args in profile_args
            }

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    # Get surface temp from original args
                    surface_temp_k = profile_args[idx][5]
                    warnings.warn(f"Profile {idx} failed: {e}")
                    results[idx] = {
                        'wind_spd': 0,
                        'surface_t': surface_temp_k,
                        'toa_t': np.nan,
                        'transmittance': np.nan,
                        'jacobian': np.nan
                    }

                completed += 1
                if verbose and completed % 100 == 0:
                    print(f"  Completed {completed}/{n_profiles} profiles")

        # Write output in MODTRAN format
        with open(output_file, 'w') as f:
            for r in results:
                f.write(
                    f"{r['wind_spd']:02d} "
                    f"{r['surface_t']:.3f} "
                    f"{r['toa_t']:.3f} "
                    f"{r['transmittance']:.4f} "
                    f"{r['jacobian']:.4f}\n"
                )

        if verbose:
            print(f"Output written to: {output_file}")

        return results


def _process_single_profile_worker(args):
    """
    Worker function for parallel profile processing.

    This is a module-level function to enable pickling for multiprocessing.
    """
    (idx, altitude_km, pressure_hpa, temp_k, spec_humidity,
     surface_temp_k, compute_jacobian, libradtran_dir, srf_file, emissivity) = args

    # Create a new LibRadTran instance in this worker process
    lrt = LibRadTranThermal(
        libradtran_dir=libradtran_dir,
        srf_file=srf_file,
        surface_emissivity=emissivity
    )

    toa_bt, tau, jacobian = lrt.process_profile(
        altitude_km, pressure_hpa, temp_k, spec_humidity,
        surface_temp_k, compute_jacobian=compute_jacobian
    )

    return {
        'wind_spd': 0,
        'surface_t': surface_temp_k,
        'toa_t': toa_bt,
        'transmittance': tau,
        'jacobian': jacobian
    }


def validate_installation():
    """
    Validate LibRadTran installation and print diagnostic information.
    """
    print("LibRadTran Installation Validator")
    print("=" * 40)

    libradtran_dir = os.environ.get('LIBRADTRAN_DIR')

    if libradtran_dir is None:
        print("ERROR: LIBRADTRAN_DIR environment variable not set")
        return False

    print(f"LIBRADTRAN_DIR: {libradtran_dir}")

    libradtran_dir = Path(libradtran_dir)

    # Check uvspec
    uvspec = libradtran_dir / 'bin' / 'uvspec'
    if uvspec.exists():
        print(f"  uvspec binary: OK ({uvspec})")
    else:
        print(f"  uvspec binary: NOT FOUND ({uvspec})")
        return False

    # Check data directory
    data_dir = libradtran_dir / 'share' / 'libRadtran' / 'data'
    if data_dir.exists():
        print(f"  Data directory: OK ({data_dir})")
    else:
        print(f"  Data directory: NOT FOUND ({data_dir})")
        return False

    # Check REPTRAN data
    reptran_dir = data_dir / 'correlated_k' / 'reptran'
    if reptran_dir.exists():
        print(f"  REPTRAN data: OK")
    else:
        print(f"  REPTRAN data: NOT FOUND (required for thermal calculations)")
        print(f"    Expected at: {reptran_dir}")
        return False

    # Check SRF file
    module_dir = Path(__file__).parent
    srf_file = module_dir / 'Data' / 'SRF' / 'landsat8_band10_srf.txt'
    if srf_file.exists():
        print(f"  SRF file: OK ({srf_file})")
    else:
        print(f"  SRF file: NOT FOUND ({srf_file})")
        return False

    print("\nInstallation: OK")
    return True


if __name__ == '__main__':
    validate_installation()
