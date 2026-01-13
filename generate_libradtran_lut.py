#!/usr/bin/env python3
"""
Generate LibRadTran-based Atmospheric Correction LUTs

This script processes all 12 months of ERA5 atmospheric profiles
through LibRadTran to generate lookup tables for atmospheric correction.

The output files are in the same format as MODTRAN outputs, allowing
seamless integration with the existing SST retrieval workflow.

Usage:
    python generate_libradtran_lut.py [options]

Options:
    --months MONTHS     Comma-separated list of months to process (e.g., "01,02,03")
                        Default: all 12 months
    --data-dir DIR      Path to Data/AtmCorrection directory
                        Default: ./Data/AtmCorrection
    --output-prefix     Prefix for output files
                        Default: libradtran_atmprofiles_
    --fast              Use fast (but less accurate) RT solver
    --no-jacobian       Skip exact Jacobian calculation (faster)
    --parallel N        Number of parallel workers (not implemented yet)
    --validate          Run validation against existing MODTRAN outputs

Example:
    # Process all months
    python generate_libradtran_lut.py

    # Process specific months with fast solver
    python generate_libradtran_lut.py --months 01,02,03 --fast

    # Validate against MODTRAN
    python generate_libradtran_lut.py --months 01 --validate

Author: Generated for Landsat SST Algorithm
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from libradtran_interface import LibRadTranThermal, validate_installation


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate LibRadTran atmospheric correction LUTs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--months',
        type=str,
        default=None,
        help='Comma-separated list of months (01-12). Default: all months'
    )

    parser.add_argument(
        '--data-dir',
        type=str,
        default='Data/AtmCorrection',
        help='Path to AtmCorrection data directory'
    )

    parser.add_argument(
        '--output-prefix',
        type=str,
        default='libradtran_atmprofiles_',
        help='Prefix for output files'
    )

    parser.add_argument(
        '--fast',
        action='store_true',
        help='Use fast (twostr) solver instead of accurate (disort)'
    )

    parser.add_argument(
        '--no-jacobian',
        action='store_true',
        help='Skip exact Jacobian calculation (estimate from transmittance)'
    )

    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate against existing MODTRAN outputs'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Check files exist without processing'
    )

    parser.add_argument(
        '--parallel',
        type=int,
        default=1,
        metavar='N',
        help='Number of parallel workers. Default: 1 (sequential)'
    )

    parser.add_argument(
        '--use-fixed-sst',
        action='store_true',
        help='Use fixed SST files (*_fixed.txt) with corrected ordering'
    )

    return parser.parse_args()


def get_months(months_arg):
    """Parse months argument."""
    if months_arg is None:
        return [f'{m:02d}' for m in range(1, 13)]
    else:
        return [m.strip().zfill(2) for m in months_arg.split(',')]


def validate_against_modtran(libradtran_file, modtran_file):
    """
    Compare LibRadTran outputs against MODTRAN outputs.

    Returns dictionary of comparison statistics.
    """
    import pandas as pd

    cols = ['wind_spd', 'surface_t', 'toa_t', 'transmittance', 'jacobian']

    try:
        lrt = pd.read_csv(libradtran_file, sep=' ', header=None, names=cols)
        mod = pd.read_csv(modtran_file, sep=' ', header=None, names=cols)
    except Exception as e:
        return {'error': str(e)}

    if len(lrt) != len(mod):
        return {'error': f'Length mismatch: LibRadTran={len(lrt)}, MODTRAN={len(mod)}'}

    # Remove NaN rows
    valid_mask = ~(lrt['toa_t'].isna() | mod['toa_t'].isna())
    lrt = lrt[valid_mask]
    mod = mod[valid_mask]

    results = {}

    # TOA brightness temperature comparison
    toa_diff = lrt['toa_t'] - mod['toa_t']
    results['toa_t'] = {
        'mean_diff': toa_diff.mean(),
        'std_diff': toa_diff.std(),
        'max_abs_diff': toa_diff.abs().max(),
        'correlation': lrt['toa_t'].corr(mod['toa_t'])
    }

    # Transmittance comparison
    tau_diff = lrt['transmittance'] - mod['transmittance']
    results['transmittance'] = {
        'mean_diff': tau_diff.mean(),
        'std_diff': tau_diff.std(),
        'max_abs_diff': tau_diff.abs().max(),
        'correlation': lrt['transmittance'].corr(mod['transmittance'])
    }

    # Jacobian comparison
    jac_diff = lrt['jacobian'] - mod['jacobian']
    results['jacobian'] = {
        'mean_diff': jac_diff.mean(),
        'std_diff': jac_diff.std(),
        'max_abs_diff': jac_diff.abs().max(),
        'correlation': lrt['jacobian'].corr(mod['jacobian'])
    }

    # Valid samples
    results['n_samples'] = len(lrt)

    return results


def print_validation_results(results, month):
    """Print formatted validation results."""
    print(f"\nValidation Results for Month {month}")
    print("=" * 50)

    if 'error' in results:
        print(f"  ERROR: {results['error']}")
        return

    print(f"  Valid samples: {results['n_samples']}")

    for var in ['toa_t', 'transmittance', 'jacobian']:
        r = results[var]
        print(f"\n  {var.upper()}:")
        print(f"    Mean difference: {r['mean_diff']:.4f}")
        print(f"    Std difference:  {r['std_diff']:.4f}")
        print(f"    Max abs diff:    {r['max_abs_diff']:.4f}")
        print(f"    Correlation:     {r['correlation']:.6f}")

    # Pass/fail assessment
    print("\n  Assessment:")
    toa_ok = abs(results['toa_t']['mean_diff']) < 0.5 and results['toa_t']['std_diff'] < 1.0
    tau_ok = abs(results['transmittance']['mean_diff']) < 0.02 and results['transmittance']['std_diff'] < 0.05

    print(f"    TOA BT:       {'PASS' if toa_ok else 'FAIL'} (target: |bias| < 0.5 K, std < 1.0 K)")
    print(f"    Transmittance: {'PASS' if tau_ok else 'FAIL'} (target: |bias| < 0.02, std < 0.05)")


def main():
    args = parse_args()

    print("LibRadTran LUT Generator")
    print("=" * 50)

    # Validate installation
    if not validate_installation():
        print("\nERROR: LibRadTran installation validation failed.")
        print("Please check your installation and LIBRADTRAN_DIR environment variable.")
        sys.exit(1)

    # Set up paths
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        sys.exit(1)

    months = get_months(args.months)
    print(f"\nMonths to process: {', '.join(months)}")
    print(f"Data directory: {data_dir}")
    print(f"Output prefix: {args.output_prefix}")
    print(f"Fast mode: {args.fast}")
    print(f"Compute Jacobian: {not args.no_jacobian}")
    print(f"Parallel workers: {args.parallel}")
    print(f"Use fixed SST files: {args.use_fixed_sst}")

    # Determine SST file suffix
    sst_suffix = '_fixed' if args.use_fixed_sst else ''

    # Check input files exist
    print("\nChecking input files...")
    missing_files = []
    for month in months:
        atm_file = data_dir / f'modtran_atmprofiles_{month}.txt'
        sst_file = data_dir / f'modtran_sstprofiles_{month}{sst_suffix}.txt'

        if not atm_file.exists():
            missing_files.append(str(atm_file))
        if not sst_file.exists():
            missing_files.append(str(sst_file))

    if missing_files:
        print("ERROR: Missing input files:")
        for f in missing_files:
            print(f"  {f}")
        sys.exit(1)

    print("  All input files found.")

    if args.dry_run:
        print("\nDry run complete. Exiting.")
        return

    # Initialize LibRadTran interface
    try:
        lrt = LibRadTranThermal()
    except Exception as e:
        print(f"ERROR: Failed to initialize LibRadTran: {e}")
        sys.exit(1)

    # Process each month
    for month in months:
        print(f"\n{'=' * 50}")
        print(f"Processing month {month}...")
        print('=' * 50)

        atm_file = data_dir / f'modtran_atmprofiles_{month}.txt'
        sst_file = data_dir / f'modtran_sstprofiles_{month}{sst_suffix}.txt'
        output_file = data_dir / f'{args.output_prefix}{month}.bts+tau+dbtdsst.txt'

        try:
            if args.parallel > 1:
                lrt.process_month_parallel(
                    atm_file,
                    sst_file,
                    output_file,
                    compute_jacobian=not args.no_jacobian,
                    n_workers=args.parallel,
                    verbose=True
                )
            else:
                lrt.process_month(
                    atm_file,
                    sst_file,
                    output_file,
                    compute_jacobian=not args.no_jacobian,
                    verbose=True
                )

            print(f"  Output: {output_file}")

            # Validate if requested
            if args.validate:
                modtran_file = data_dir / f'modtran_atmprofiles_{month}.bts+tau+dbtdsst.txt'
                if modtran_file.exists():
                    results = validate_against_modtran(output_file, modtran_file)
                    print_validation_results(results, month)
                else:
                    print(f"  Validation skipped: MODTRAN file not found ({modtran_file})")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "=" * 50)
    print("Processing complete!")


if __name__ == '__main__':
    main()
