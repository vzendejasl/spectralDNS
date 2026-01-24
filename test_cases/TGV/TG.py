#!/usr/bin/env python3
from __future__ import print_function

import os
import sys
import glob
import warnings
import csv

import numpy as np
from numpy import pi, zeros, sum, float64, sin, cos, prod
import h5py
from mpi4py import MPI

from shenfun.fourier import energy_fourier
from shenfun import ShenfunFile, Array, Function
from spectralDNS import config, get_solver, solve

try:
    import matplotlib.pyplot as plt
except ImportError:
    warnings.warn("matplotlib not installed")
    plt = None


# -------------------------
# Spectrum helper (optional)
# -------------------------
def spectrum(solver, context):
    c = context
    E_hat = 0.5 * sum((c.U_hat * np.conj(c.U_hat)).real, axis=0)

    # Account for Hermitian symmetry in the last dimension
    if E_hat.shape[-1] > 1:
        E_hat[..., 1:-1] *= 2

    k_mag = np.sqrt(c.K2)
    k0 = 2.0 * pi / config.params.L[0]  # assume cubic domain

    k_idx = np.floor(k_mag / k0 + 0.5).astype(int)

    max_possible_k = int(np.max(config.params.N) / 2 * np.sqrt(3)) + 2
    bins = np.arange(max_possible_k) - 0.5

    Ek_local, _ = np.histogram(k_idx.flatten(), bins=bins, weights=E_hat.flatten())
    Ek = solver.comm.allreduce(Ek_local)

    centers = (bins[:-1] + bins[1:]) / 2
    return centers, Ek


# -------------------------
# Initialization
# -------------------------
def initialize(solver, context):
    if 'NS' in config.params.solver:
        initialize1(solver, context)
    else:
        initialize2(solver, context)
    config.params.t = 0.0
    config.params.tstep = 0


def initialize1(solver, context):
    U, X = context.U, context.X
    scale = 1.0 / config.params.L_ref
    U[0] = sin(scale * X[0]) * cos(scale * X[1]) * cos(scale * X[2])
    U[1] = -cos(scale * X[0]) * sin(scale * X[1]) * cos(scale * X[2])
    U[2] = 0
    solver.set_velocity(**context)


def initialize2(solver, context):
    U, X = context.U, context.X
    scale = 1.0 / config.params.L_ref
    U[0] = sin(scale * X[0]) * cos(scale * X[1]) * cos(scale * X[2])
    U[1] = -cos(scale * X[0]) * sin(scale * X[1]) * cos(scale * X[2])
    U[2] = 0
    solver.set_velocity(**context)
    solver.cross2(context.W_hat, context.K, context.U_hat)


# -------------------------
# Output naming
# -------------------------
def _format_re(value):
    text = "{:g}".format(value)
    return text.replace(".", "p")


def _output_dir(params):
    return "spectralDNS_tgv_Re{}NumPtsPerDir{}".format(
        _format_re(params.Re),
        int(params.N[0]),
    )

def _base_filename(params):
    return "tgv_out_Re{}NumPtsPerDir{}".format(
        _format_re(params.Re),
        int(params.N[0]),
    )

def _diagnostics_filename(params):
    return _base_filename(params) + ".csv"

def _turb_filename(params):
    return "tgv_out_turb_Re{}NumPtsPerDir{}.csv".format(
        _format_re(params.Re),
        int(params.N[0]),
    )

def _turb_continued_filename(params):
    return "tgv_out_turb_continued_Re{}NumPtsPerDir{}.csv".format(
        _format_re(params.Re),
        int(params.N[0]),
    )

def _turb_grid_filename(params):
    return "tgv_out_turb_grid_Re{}NumPtsPerDir{}.csv".format(
        _format_re(params.Re),
        int(params.N[0]),
    )


# -------------------------
# Restart support
# -------------------------
def _select_checkpoint_file(solver, context):
    params = config.params
    user_path = params.get("restart_file", "")
    if user_path:
        if os.path.exists(user_path):
            if solver.rank == 0:
                print(f"Restart: using user-specified checkpoint {user_path}")
            return user_path
        if solver.rank == 0:
            print(f"Restart: specified checkpoint not found: {user_path}")
        return None

    candidates = set()
    basename = context.hdf5file.filename
    for path in (basename + "_c.h5", basename + "_c"):
        if os.path.exists(path):
            candidates.add(path)

    search_dir = params.get("output_dir", ".")
    for path in glob.glob(os.path.join(search_dir, "*_c.h5")) + glob.glob(os.path.join(search_dir, "*_c")):
        if os.path.exists(path):
            candidates.add(path)

    best_path = None
    best_tstep = -1
    best_t = 0.0
    for path in sorted(candidates):
        try:
            with h5py.File(path, "r") as f:
                tstep = int(f.attrs.get("tstep", -1))
                t = float(f.attrs.get("t", 0.0))
        except Exception:
            continue
        if tstep > best_tstep:
            best_path = path
            best_tstep = tstep
            best_t = t

    if best_path is not None and solver.rank == 0:
        print(f"Restart: selected checkpoint {best_path} (tstep={best_tstep}, t={best_t:.16e})")

    return best_path


def _load_restart(solver, context):
    params = config.params
    chk_path = _select_checkpoint_file(solver, context)
    if chk_path is None:
        if solver.rank == 0:
            print("Restart: no checkpoint found, starting from initial conditions.")
        return False

    local_slice = None
    if hasattr(context, "T") and hasattr(context.T, "local_slice"):
        local_slice = context.T.local_slice()

    with h5py.File(chk_path, "r") as f:
        tstep = int(f.attrs.get("tstep", -1))
        t = float(f.attrs.get("t", 0.0))
        if "U" not in f or "3D" not in f["U"]:
            if solver.rank == 0:
                print("Restart: checkpoint missing U/3D dataset, skipping restart.")
            return False

        group = f["U/3D"]
        dset_key = "0" if "0" in group else None
        if dset_key is None:
            keys = [k for k in group.keys() if k.isdigit()]
            if not keys:
                if solver.rank == 0:
                    print("Restart: no numeric datasets in U/3D, skipping restart.")
                return False
            dset_key = sorted(keys, key=int)[0]

        if solver.rank == 0:
            print(f"Restart: located dataset U/3D/{dset_key}")

        dset = group[dset_key]
        if local_slice is not None and dset.shape != context.U_hat.shape:
            data = dset[(slice(None),) + local_slice]
        else:
            data = dset[...]

    if data.shape != context.U_hat.shape:
        if solver.rank == 0:
            print(f"Restart: dataset shape {data.shape} does not match U_hat shape {context.U_hat.shape}, skipping restart.")
        return False

    context.U_hat[...] = data
    params.t = t
    params.tstep = tstep
    params.filemode = "a"
    params.restarted = True

    if solver.rank == 0:
        print(f"Restart: loaded t={params.t:.16e}, tstep={params.tstep}")

    # Advance scheduled snapshot indices so we don't immediately re-write old times
    snapshot_times_fields = params.get("snapshot_times_fields", None)
    if snapshot_times_fields is not None:
        while (params.snapshot_index_fields < len(snapshot_times_fields) and
               params.t >= snapshot_times_fields[params.snapshot_index_fields] - params.dt * 0.01):
            params.snapshot_index_fields += 1

    snapshot_times_spectrum = params.get("snapshot_times_spectrum", None)
    if snapshot_times_spectrum is not None:
        while (params.snapshot_index_spectrum < len(snapshot_times_spectrum) and
               params.t >= snapshot_times_spectrum[params.snapshot_index_spectrum] - params.dt * 0.01):
            params.snapshot_index_spectrum += 1

    return True


# -------------------------
# Diagnostics / plotting
# -------------------------
k = []
w = []
im1 = None
kold = zeros(1)

# Global temporary array for gradients to avoid reallocation if possible
# Note: In a real persistent application we might attach this to context
_grad_u = None

def update(context):
    global k, w, im1, _grad_u
    c = context
    params = config.params
    solver = config.solver
    t_phys = params.t

    diag_interval = params.get("diag_interval", 1)
    plot_step = params.get("plot_step", 0)

    if (params.tstep % diag_interval == 0) or (plot_step > 0 and params.tstep % plot_step == 0):
        U = solver.get_velocity(**c)
        curl = solver.get_curl(**c)
        if 'NS' in params.solver:
            P = solver.get_pressure(**c)

    if plt is not None:
        if plot_step > 0 and params.tstep % plot_step == 0 and solver.rank == 0:
            if im1 is None:
                plt.figure()
                im1 = plt.contourf(c.X[1][:, :, 0], c.X[0][:, :, 0], U[0, :, :, 10], 100)
                plt.colorbar(im1)
                plt.draw()
                globals().update(im1=im1)
            else:
                ax = im1.axes
                ax.clear()
                im1 = ax.contourf(c.X[1][:, :, 0], c.X[0][:, :, 0], U[0, :, :, 10], 100)
                globals().update(im1=im1)
            plt.pause(1e-6)

    if params.tstep == 1:
        from spectralDNS.utilities import reset_profile
        try:
            reset_profile(profile)
            print("Reset profile")
        except Exception:
            pass

    if params.tstep % diag_interval == 0:
        
        # 1. Basic stats
        ww = solver.comm.allreduce(sum(curl.astype(float64) * curl.astype(float64)) / prod(params.N) / 2)
        kk = solver.comm.allreduce(sum(U.astype(float64) * U.astype(float64)) / prod(params.N) / 2)

        # 2. Detailed Turbulence Stats
        
        local_diss = np.zeros_like(U[0], dtype=float64)
        
        # Longitudinal moments and component sums
        sum_long_d2 = 0.0 
        sum_long_d3 = 0.0 
        sum_long_d4 = 0.0
        
        sum_u2_comp1 = np.sum(U[0]**2)
        sum_u2_comp3 = np.sum(U[2]**2)
        
        sum_grad_u1_sq = 0.0 
        sum_grad_u3_sq = 0.0
        
        # Evaluation of L-inf norms
        u_inf_loc = np.max(np.abs(U))
        p_inf_loc = np.max(np.abs(P)) if 'NS' in params.solver else 0.0
        
        # Evaluation of CFL
        # CFL = max(|u_i|) * dt / dx
        dx = params.L[0] / params.N[0]
        
        # Temporary spectral buffers
        grad_hat = Function(c.T)
        grad_phys = Array(c.T)
        grad_hat_j = Function(c.T) 
        grad_phys_j = Array(c.T)
        
        local_diss[:] = 0.0
        
        # Diagonals: du/dx, dv/dy, dw/dz
        for i in range(3):
            grad_hat[:] = 1j * c.K[i] * c.U_hat[i]
            grad_phys = grad_hat.backward(grad_phys)
            
            # Dissipation part
            local_diss += grad_phys.real**2
            
            # Skew/Flat part
            d_val = grad_phys.real
            d2 = d_val**2
            sum_long_d2 += np.sum(d2)
            sum_long_d3 += np.sum(d2 * d_val)
            sum_long_d4 += np.sum(d2 * d2)
            
            # D3/D1 part
            if i == 0: sum_grad_u1_sq += np.sum(d2)
            if i == 2: sum_grad_u3_sq += np.sum(d2)
        
        # Off-diagonals
        pairs = [(0,1), (0,2), (1,2)]
        for (i, j) in pairs:
            grad_hat[:] = 1j * c.K[j] * c.U_hat[i]
            grad_phys = grad_hat.backward(grad_phys)
            val_ij = grad_phys.real
            
            grad_hat_j[:] = 1j * c.K[i] * c.U_hat[j]
            grad_phys_j = grad_hat_j.backward(grad_phys_j)
            val_ji = grad_phys_j.real
            
            S_ij = 0.5 * (val_ij + val_ji)
            local_diss += 2.0 * S_ij**2 
            
            if i == 0: sum_grad_u1_sq += np.sum(val_ij**2)
            elif i == 2: sum_grad_u3_sq += np.sum(val_ij**2)
                
            if j == 0: sum_grad_u1_sq += np.sum(val_ji**2)
            elif j == 2: sum_grad_u3_sq += np.sum(val_ji**2)
            
        local_diss *= 2.0 * params.nu
        
        # Global reductions
        max_diss = solver.comm.allreduce(np.max(local_diss), op=MPI.MAX)
        total_diss = solver.comm.allreduce(np.sum(local_diss), op=MPI.SUM)
        avg_diss = total_diss / prod(params.N)
        
        glob_sum_long_d2 = solver.comm.allreduce(sum_long_d2, op=MPI.SUM)
        glob_sum_long_d3 = solver.comm.allreduce(sum_long_d3, op=MPI.SUM)
        glob_sum_long_d4 = solver.comm.allreduce(sum_long_d4, op=MPI.SUM)
        
        glob_sum_u2_c1 = solver.comm.allreduce(sum_u2_comp1, op=MPI.SUM)
        glob_sum_u2_c3 = solver.comm.allreduce(sum_u2_comp3, op=MPI.SUM)
        
        glob_sum_grad_u1_sq = solver.comm.allreduce(sum_grad_u1_sq, op=MPI.SUM)
        glob_sum_grad_u3_sq = solver.comm.allreduce(sum_grad_u3_sq, op=MPI.SUM)
        
        u_inf = solver.comm.allreduce(u_inf_loc, op=MPI.MAX)
        p_inf = solver.comm.allreduce(p_inf_loc, op=MPI.MAX)
        
        vol_points = prod(params.N)
        cfl = u_inf * params.dt / dx
        
        # Calculate derived quantities
        avg_long_d2 = (glob_sum_long_d2 / vol_points) / 3.0
        avg_long_d3 = (glob_sum_long_d3 / vol_points) / 3.0
        avg_long_d4 = (glob_sum_long_d4 / vol_points) / 3.0
        
        skewness = avg_long_d3 / (avg_long_d2**1.5) if avg_long_d2 > 0 else 0.0
        flatness = avg_long_d4 / (avg_long_d2**2.0) if avg_long_d2 > 0 else 0.0
        
        D3_over_D1 = glob_sum_grad_u3_sq / glob_sum_grad_u1_sq if glob_sum_grad_u1_sq > 0 else 0.0
        E3_over_E1 = glob_sum_u2_c3 / glob_sum_u2_c1 if glob_sum_u2_c1 > 0 else 0.0
        
        avg_SijSij = (total_diss / (2.0 * params.nu)) / vol_points if params.nu > 0 else 0.0
        
        # Derived turbulence scales
        if avg_diss > 1e-20:
            kolmLenScl = (params.nu**3 / max_diss)**0.25 if max_diss > 0 else 0.0
            avg_kolmLenScl = (params.nu**3 / avg_diss)**0.25
            avg_lambda = np.sqrt(10.0 * params.nu * kk / avg_diss)
            kolmTimeScl = np.sqrt(params.nu / max_diss) if max_diss > 0 else 0.0
            avg_kolmTimeScl = np.sqrt(params.nu / avg_diss)
        else:
            kolmLenScl = 0.0
            avg_kolmLenScl = 0.0
            avg_lambda = 0.0
            kolmTimeScl = 0.0
            avg_kolmTimeScl = 0.0

        u_rms = np.sqrt(2.0/3.0 * kk)
        Re_taylor = u_rms * avg_lambda / params.nu if params.nu > 0 else 0.0
        
        # Grid metrics
        kmax = pi / dx
        kmax_eta = kmax * kolmLenScl
        hmin_eta = dx / kolmLenScl if kolmLenScl > 0 else 0.0
        avg_kmax_eta = kmax * avg_kolmLenScl
        avg_hmin_eta = dx / avg_kolmLenScl if avg_kolmLenScl > 0 else 0.0
        
        # PI_nu estimate (approximate vel_curl_ke as 2*enstrophy in incompressible)
        vel_curl_ke = 2.0 * ww
        PI_nu = pow(avg_diss, 0.5) / (avg_kolmLenScl * pow(vel_curl_ke, 0.75)) if avg_kolmLenScl > 0 and vel_curl_ke > 0 else 0.0
        PI_nu_min = pow(max_diss, 0.5) / (kolmLenScl * pow(vel_curl_ke, 0.75)) if kolmLenScl > 0 and vel_curl_ke > 0 else 0.0

        # Save to CSVs
        if solver.rank == 0:
            header_fmt = "%26s," * 3 + "%26s\n"
            data_fmt = "    %22.16e," * 3 + "    %22.16e\n"
            
            # Main Diagnostics
            diag_file = params.diagnostics_filename
            file_exists = os.path.isfile(diag_file)
            with open(diag_file, "a") as f:
                if not file_exists:
                    f.write("%26s,%26s,%26s,%26s,%26s,%26s,%26s\n" % ("Time", "Cycle", "u_inf", "p_inf", "KineticEnergy", "Enstrophy", "CFL"))
                f.write("    %22.16e,    %22.16e,    %22.16e,    %22.16e,    %22.16e,    %22.16e,    %22.16e\n" % 
                        (t_phys, float(params.tstep), u_inf, p_inf, float(kk), float(ww), cfl))

            # Turb Stats
            turb_file = os.path.join(params.output_dir, _turb_filename(params))
            file_exists = os.path.isfile(turb_file)
            with open(turb_file, "a") as f:
                if not file_exists:
                    f.write("%26s,%26s,%26s,%26s,%26s,%26s,%26s,%26s,%26s,%26s,%26s\n" % 
                            ("Time", "Cycle", "MaxDissipation", "AvgDissipation", "MinKolmLen", "TaylorLen", "AvgKolmLen", "MinKolmTime", "AvgKolmTime", "ReTaylor", "u_rms"))
                f.write(("    %22.16e,"*10 + "    %22.16e\n") % 
                        (t_phys, float(params.tstep), max_diss, avg_diss, kolmLenScl, avg_lambda, avg_kolmLenScl, kolmTimeScl, avg_kolmTimeScl, Re_taylor, u_rms))
            
            # Turb Continued Stats
            turb_cont_file = os.path.join(params.output_dir, _turb_continued_filename(params))
            file_exists = os.path.isfile(turb_cont_file)
            with open(turb_cont_file, "a") as f:
                if not file_exists:
                    f.write("%26s,%26s,%26s,%26s,%26s,%26s,%26s\n" % ("Time", "Cycle", "avg_SijSij", "Skewness", "Flatness", "D31", "E31"))
                f.write("    %22.16e,    %22.16e,    %22.16e,    %22.16e,    %22.16e,    %22.16e,    %22.16e\n" % 
                        (t_phys, float(params.tstep), avg_SijSij, skewness, flatness, D3_over_D1, E3_over_E1))

            # Turb Grid Stats
            turb_grid_file = os.path.join(params.output_dir, _turb_grid_filename(params))
            file_exists = os.path.isfile(turb_grid_file)
            with open(turb_grid_file, "a") as f:
                if not file_exists:
                    f.write("%26s,%26s,%26s,%26s,%26s,%26s,%26s,%26s\n" % 
                            ("Time", "Cycle", "Kmax*eta", "hmin/eta", "Avg_PI_NU", "Min_PI_NU", "Kmax*eta(Avg)", "hmin/eta(Avg)"))
                f.write(("    %22.16e,"*7 + "    %22.16e\n") % 
                        (t_phys, float(params.tstep), kmax_eta, hmin_eta, PI_nu, PI_nu_min, avg_kmax_eta, avg_hmin_eta))

            # Print to stdout for user
            if params.tstep == 0:
                print("{:>26} {:>26} {:>26} {:>26}".format("Time", "Cycle", "KineticEnergy", "Enstrophy"))
            print("{:26.16e} {:26.16e} {:26.16e} {:26.16e}".format(
                t_phys, float(params.tstep), float(kk), float(ww)
            ))

        # Trigger spectrum calculation
        should_compute = False
        snapshot_times_spectrum = params.get('snapshot_times_spectrum', None)
        if snapshot_times_spectrum is not None:
            while (params.snapshot_index_spectrum < len(snapshot_times_spectrum) and
                   params.t >= snapshot_times_spectrum[params.snapshot_index_spectrum] - params.dt * 0.01):
                should_compute = True
                params.snapshot_index_spectrum += 1
        elif params.get('compute_spectrum', 0) > 0 and params.tstep % params.compute_spectrum == 0:
            should_compute = True
        elif params.get('snapshot_step', -1) >= 0 and params.tstep == params.snapshot_step:
            should_compute = True

        if should_compute:
            centers, Ek = spectrum(solver, context)
            if solver.rank == 0:
                # Sanity Check
                total_Ek = np.sum(Ek)
                diff = abs(kk - total_Ek)
                print("Sanity Check (t={:.4f}): KE_sim = {:.16e}, Sum(E_k) = {:.16e}, Diff = {:.6e}".format(t_phys, kk, total_Ek, diff))

                # Save to HDF5
                spec_h5 = os.path.join(params.output_dir, "energy_spectrum.h5")
                with h5py.File(spec_h5, "a") as f:
                    grp = f.require_group("spectra")
                    if str(params.tstep) in grp:
                        del grp[str(params.tstep)]
                    dset = grp.create_dataset(str(params.tstep), data=Ek)
                    dset.attrs["time"] = t_phys

                # Save to a text file for easy reading
                out_dir = os.path.join(params.output_dir, "spectrum")
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)

                txt_filename = os.path.join(out_dir, "spectrum_{:06d}.txt".format(params.tstep))
                header = "Time: {:26.16e}\n{:>24},{:>26}".format(t_phys, "k", "E(k)")
                np.savetxt(txt_filename, np.column_stack((centers, Ek)), header=header, delimiter=",")
                print("Saved spectrum to {}".format(txt_filename))

        if 'NS' not in params.solver:
            c.U_hat = c.U.forward(c.U_hat)


def regression_test(context):
    params = config.params
    solver = config.solver
    U = solver.get_velocity(**context)
    curl = solver.get_curl(**context)
    wv = solver.comm.allreduce(sum(curl.astype(float64) * curl.astype(float64)) / prod(params.N) / 2)
    kv = solver.comm.allreduce(sum(U.astype(float64) * U.astype(float64)) / prod(params.N) / 2)
    config.solver.MemoryUsage('End')
    if solver.rank == 0:
        try:
            assert round(wv.item() - 0.375249930801, params.ntol) == 0, wv
            assert round(kv.item() - 0.124953117517, params.ntol) == 0, kv
        except AssertionError as e:
            print(f"Regression check failed: {e}. This is expected if you changed simulation parameters.")


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":

    # Defaults; CLI will override
    config.update(
        {
            'dt': 0.01,
            'T': 0.1,
            'L': [1.0, 1.0, 1.0],
            'M': [5, 5, 5],
            'planner_effort': {'fft': 'FFTW_ESTIMATE', 'rfftn': 'FFTW_ESTIMATE', 'irfftn': 'FFTW_ESTIMATE'},
        },
        "triplyperiodic",
    )

    config.triplyperiodic.add_argument("--diag_interval", type=int, default=1)
    config.triplyperiodic.add_argument("--compute_spectrum", type=int, default=0,
                                       help="Interval to compute energy spectrum (0 to disable)")
    config.triplyperiodic.add_argument("--write_result_dt", type=float, default=0.0,
                                       help="Write result every T seconds (overrides write_result if > 0)")
    config.triplyperiodic.add_argument("--plot_step", type=int, default=0)
    config.triplyperiodic.add_argument("--snapshot-time", type=float, default=-1.0,
                                       help="Physical time to force a single snapshot (fields + spectrum)")
    config.triplyperiodic.add_argument("--num_snapshots", type=int, default=0,
                                       help="Number of field snapshots to save evenly distributed over T")
    config.triplyperiodic.add_argument("--num_spectrum_snapshots", type=int, default=0,
                                       help="Number of spectrum snapshots to save evenly distributed over T")
    config.triplyperiodic.add_argument("--Re", type=float, default=100.0,
                                       help="Reynolds number (defines viscosity)")
    config.triplyperiodic.add_argument("--problem", type=int, default=2, choices=[1, 2],
                                       help="1 => [0,2*pi]^3 (dt unchanged), 2 => [0,1]^3 (dt scaled by 1/(2*pi))")
    config.triplyperiodic.add_argument("--restart-file", type=str, default="",
                                       help="Optional checkpoint file to restart from (defaults to latest found).")
    config.triplyperiodic.add_argument("--N", default=[32, 32, 32], nargs=3, help="Mesh size. Trumps M.")

    sol = get_solver(update=update, regression_test=regression_test, mesh="triplyperiodic")

    # Characteristic scales/domain
    if config.params.problem == 1:
        domain_length = 2 * pi
        L_ref = 1.0
        time_scale = 1.0
    else:
        domain_length = 1.0
        L_ref = 1.0 / (2 * pi)
        time_scale = L_ref

    config.params.L = [domain_length, domain_length, domain_length]
    config.params.time_scale = time_scale
    config.params.L_ref = L_ref

    U_ref = 1.0
    config.params.nu = (U_ref * L_ref) / config.params.Re

    input_dt = config.params.dt
    config.params.dt = input_dt * time_scale

    input_T = config.params.T
    config.params.T = input_T

    config.params.snapshot_step = -1

    # Snapshot scheduling (FIELDS)
    if config.params.num_snapshots > 0:
        times = [i * config.params.T / config.params.num_snapshots for i in range(0, config.params.num_snapshots + 1)]
        if config.params.snapshot_time >= 0 and config.params.snapshot_time <= config.params.T:
            times.append(config.params.snapshot_time)
        times.sort()
        config.params.snapshot_times_fields = times
        config.params.snapshot_index_fields = 0
        config.params.write_result = 2000000000
        if sol.rank == 0:
            print(f"num_snapshots={config.params.num_snapshots} -> Scheduled {len(times)} field snapshots by time.")
    elif config.params.snapshot_time >= 0 and config.params.snapshot_time <= config.params.T:
        config.params.snapshot_times_fields = [config.params.snapshot_time]
        config.params.snapshot_index_fields = 0
        config.params.write_result = 2000000000

    # Regular write interval via dt (if requested)
    if config.params.write_result_dt > 0 and config.params.dt > 0:
        config.params.write_result = int(config.params.write_result_dt / config.params.dt)
        if config.params.write_result < 1:
            config.params.write_result = 1
        if sol.rank == 0:
            print(f"Setting write_result to {config.params.write_result} steps (every {config.params.write_result_dt}s physical)")

    # Snapshot scheduling (SPECTRUM)
    if config.params.num_spectrum_snapshots > 0:
        times = [i * config.params.T / config.params.num_spectrum_snapshots for i in range(0, config.params.num_spectrum_snapshots + 1)]
        if config.params.snapshot_time >= 0 and config.params.snapshot_time <= config.params.T:
            times.append(config.params.snapshot_time)
        times.sort()
        config.params.snapshot_times_spectrum = times
        config.params.snapshot_index_spectrum = 0
        config.params.compute_spectrum = 2000000000
        if sol.rank == 0:
            print(f"num_spectrum_snapshots={config.params.num_spectrum_snapshots} -> Scheduled {len(times)} spectrum snapshots by time.")
    elif config.params.snapshot_time >= 0 and config.params.snapshot_time <= config.params.T:
        config.params.snapshot_times_spectrum = [config.params.snapshot_time]
        config.params.snapshot_index_spectrum = 0
        config.params.compute_spectrum = 2000000000

    if sol.rank == 0:
        domain_label = "[0, 2*pi]^3" if config.params.problem == 1 else "[0, 1]^3"
        print(f"--- Taylor-Green Vortex Setup ({domain_label}) ---")
        print(f"Reference Length (L): {L_ref:.6f}")
        print("Reference Velocity (U): 1.0")
        print(f"Reynolds Number (Re): {config.params.Re:.2f}")
        print(f"Computed Viscosity (nu): {config.params.nu:.6g}")
        print(f"Input dt (User): {input_dt:g}")
        print(f"Simulation dt (Used): {config.params.dt:g}")
        print(f"End Time T (Physical): {input_T:g}")
        if config.params.snapshot_time >= 0:
            print(f"Snapshot requested at physical t={config.params.snapshot_time:g}")
        print("---------------------------------------------")

    context = sol.get_context()

    # Output directory/filenames
    config.params.output_dir = _output_dir(config.params)
    if sol.rank == 0 and not os.path.exists(config.params.output_dir):
        os.makedirs(config.params.output_dir)

    config.params.base_filename = _base_filename(config.params)
    config.params.diagnostics_filename = os.path.join(config.params.output_dir, _diagnostics_filename(config.params))

    context.hdf5file.filename = os.path.join(config.params.output_dir, config.params.base_filename)

    # Store curl too
    context.hdf5file.results['data'].update({'curl': [context.curl]})

    def update_components(**c):
        sol.get_velocity(**c)
        sol.get_pressure(**c)
        sol.get_curl(**c)

    context.hdf5file.update_components = update_components

    # Custom HDF5 update
    def custom_update(params, **kw):
        self = context.hdf5file

        if self.cfile is None:
            if params.get("restarted", False):
                params.filemode = "a"

            self.cfile = ShenfunFile(self.filename + '_c', self.checkpoint['space'], mode=params.filemode)
            self.cfile.open()
            if 'tstep' not in self.cfile.f.attrs:
                self.cfile.f.attrs.create('tstep', 0)
            if 't' not in self.cfile.f.attrs:
                self.cfile.f.attrs.create('t', 0.0)
            self.cfile.close()

        if self.wfile is None:
            self.wfile = ShenfunFile(self.filename + '_w', self.results['space'], mode=params.filemode)

        checkpoint_written = False
        should_write = False

        snapshot_times_fields = params.get('snapshot_times_fields', None)
        if snapshot_times_fields is not None:
            snapshot_index_fields = params.get('snapshot_index_fields', 0)
            while (snapshot_index_fields < len(snapshot_times_fields) and
                   params.t >= snapshot_times_fields[snapshot_index_fields] - params.dt * 0.01):
                should_write = True
                snapshot_index_fields += 1
            params.snapshot_index_fields = snapshot_index_fields
        else:
            if params.tstep % params.write_result == 0:
                should_write = True

        if should_write:
            self.update_components(**kw)
            self.wfile.write(params.tstep, self.results['data'], as_scalar=False)
            for key, val in self.checkpoint['data'].items():
                self.cfile.write(int(key), val)
            self.cfile.open()
            self.cfile.f.attrs['tstep'] = params.tstep
            self.cfile.f.attrs['t'] = params.t
            self.cfile.close()
            checkpoint_written = True
            if sol.rank == 0:
                print(f"Checkpoint written at step {params.tstep} (t={params.t:.16e})")

        kill = self.check_if_kill()
        if (params.tstep % params.checkpoint == 0 or kill) and not checkpoint_written:
            for key, val in self.checkpoint['data'].items():
                self.cfile.write(int(key), val)
                self.cfile.open()
                self.cfile.f.attrs['tstep'] = params.tstep
                self.cfile.f.attrs['t'] = params.t
                self.cfile.close()
            if sol.rank == 0:
                print(f"Checkpoint written at step {params.tstep} (t={params.t:.16e})")
            if kill:
                sys.exit(1)

    context.hdf5file.update = custom_update

    # Restart or initialize
    restarted = _load_restart(sol, context)
    if not restarted:
        initialize(sol, context)
        update(context)
        context.hdf5file.update(config.params, **context)

    # Solve
    solve(sol, context)

    # Final save
    if sol.rank == 0:
        print("Done.")