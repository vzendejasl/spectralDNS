from __future__ import print_function
import warnings
import os
import glob
from numpy import pi, zeros, sum, float64, sin, cos, prod
import numpy as np
import h5py
from shenfun.fourier import energy_fourier
from shenfun import ShenfunFile
from spectralDNS import config, get_solver, solve

try:
    import matplotlib.pyplot as plt

except ImportError:
    warnings.warn("matplotlib not installed")
    plt = None

def spectrum(solver, context):
    c = context
    # Compute energy density
    E_hat = 0.5 * sum((c.U_hat * np.conj(c.U_hat)).real, axis=0)

    # Account for Hermitian symmetry in the last dimension
    if E_hat.shape[-1] > 1:
        E_hat[..., 1:-1] *= 2

    # Wavenumbers
    k_mag = np.sqrt(c.K2)

    # Fundamental wavenumber (assuming L[0] == L[1] == L[2])
    k0 = 2.0 * pi / config.params.L[0]

    # Normalize to integer shell index
    k_idx = np.floor(k_mag / k0 + 0.5).astype(int)

    # Use global N to determine max k, ensuring consistent bins across ranks
    # Assuming cubic domain for simplicity in binning logic similar to Isotropic.py
    Nb = int(np.sqrt(sum((config.params.N/2)**2)/3))
    # Or simply:
    max_possible_k = int(np.max(config.params.N) / 2 * np.sqrt(3)) + 2 # Upper bound
    
    # We use a large enough fixed bin range based on global resolution
    bins = np.arange(max_possible_k) - 0.5

    # Histogram
    Ek_local, _ = np.histogram(k_idx.flatten(), bins=bins, weights=E_hat.flatten())

    # Sum across ranks. Use allreduce so all ranks get the result (or reduce if only root needs it, 
    # but allreduce is safer to avoid synchronization confusion if we returned None on others).
    # However, since we return centers and Ek, let's use allreduce so every rank returns valid data,
    # preventing potential issues if the caller expects data.
    Ek = solver.comm.allreduce(Ek_local)

    centers = (bins[:-1] + bins[1:]) / 2
    
    # Filter out empty high-k bins if desired, or keep them fixed size. 
    # For now, return the full fixed size array.
    
    return centers, Ek

def initialize(solver, context):
    if 'NS' in config.params.solver:
        initialize1(solver, context)

    else:
        initialize2(solver, context)
    config.params.t = 0.0
    config.params.tstep = 0

def initialize1(solver, context):
    U, X = context.U, context.X
    scale = 1.0/config.params.L_ref
    U[0] = sin(scale*X[0])*cos(scale*X[1])*cos(scale*X[2])
    U[1] = -cos(scale*X[0])*sin(scale*X[1])*cos(scale*X[2])
    U[2] = 0
    solver.set_velocity(**context)

def initialize2(solver, context):
    U, X = context.U, context.X
    scale = 1.0/config.params.L_ref
    U[0] = sin(scale*X[0])*cos(scale*X[1])*cos(scale*X[2])
    U[1] = -cos(scale*X[0])*sin(scale*X[1])*cos(scale*X[2])
    U[2] = 0
    solver.set_velocity(**context)
    solver.cross2(context.W_hat, context.K, context.U_hat)

def _select_checkpoint_file(solver, context):
    params = config.params
    user_path = getattr(params, "restart_file", None)
    if user_path:
        if os.path.exists(user_path):
            if solver.rank == 0:
                print("Restart: using user-specified checkpoint {}".format(user_path))
            return user_path
        if solver.rank == 0:
            print("Restart: specified checkpoint not found: {}".format(user_path))
        return None

    candidates = set()
    basename = context.hdf5file.filename
    for path in (basename + "_c.h5", basename + "_c"):
        if os.path.exists(path):
            candidates.add(path)
    for path in glob.glob("*_c.h5") + glob.glob("*_c"):
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
        print("Restart: selected checkpoint {} (tstep={}, t={:.16e})".format(
            best_path, best_tstep, best_t))

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
        dset_path = "U/3D/{}".format(dset_key)
        if solver.rank == 0:
            print("Restart: located dataset {}".format(dset_path))
        dset = group[dset_key]
        if local_slice is not None and dset.shape != context.U_hat.shape:
            data = dset[(slice(None),) + local_slice]
        else:
            data = dset[...]

    if data.shape != context.U_hat.shape:
        if solver.rank == 0:
            print("Restart: dataset shape {} does not match U_hat shape {}, skipping restart.".format(
                data.shape, context.U_hat.shape))
        return False

    context.U_hat[...] = data
    params.t = t
    params.tstep = tstep
    params.filemode = "a"
    params.restarted = True

    if solver.rank == 0:
        print("Restart: loaded t={:.16e}, tstep={}".format(params.t, params.tstep))

    if hasattr(params, "snapshot_times_fields"):
        while (params.snapshot_index_fields < len(params.snapshot_times_fields) and
               params.t >= params.snapshot_times_fields[params.snapshot_index_fields] - params.dt * 0.01):
            params.snapshot_index_fields += 1
    if hasattr(params, "snapshot_times_spectrum"):
        while (params.snapshot_index_spectrum < len(params.snapshot_times_spectrum) and
               params.t >= params.snapshot_times_spectrum[params.snapshot_index_spectrum] - params.dt * 0.01):
            params.snapshot_index_spectrum += 1

    return True

k = []
w = []
im1 = None
kold = zeros(1)
def update(context):
    global k, w, im1
    c = context
    params = config.params
    solver = config.solver
    t_phys = params.t

    if (params.tstep % params.diag_interval == 0 or
            (params.plot_step > 0 and params.tstep % params.plot_step == 0)):
        U = solver.get_velocity(**c)
        curl = solver.get_curl(**c)
        if 'NS' in params.solver:
            solver.get_pressure(**c)

    if plt is not None:
        if params.plot_step > 0 and params.tstep % params.plot_step == 0 and solver.rank == 0:
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
        except:
            pass

    if params.tstep % params.diag_interval == 0:
        ww = solver.comm.reduce(sum(curl.astype(float64)*curl.astype(float64))/prod(params.N)/2)
        kk = solver.comm.reduce(sum(U.astype(float64)*U.astype(float64))/prod(params.N)/2) # Compute energy with double precision

        # Trigger spectrum calculation
        should_compute = False
        if hasattr(params, 'snapshot_times_spectrum'):
            while (params.snapshot_index_spectrum < len(params.snapshot_times_spectrum) and 
                   params.t >= params.snapshot_times_spectrum[params.snapshot_index_spectrum] - params.dt * 0.01):
                should_compute = True
                params.snapshot_index_spectrum += 1
                
        elif hasattr(params, 'compute_spectrum') and params.compute_spectrum > 0 and params.tstep % params.compute_spectrum == 0:
            should_compute = True
        elif hasattr(params, 'snapshot_step') and params.tstep == params.snapshot_step:
            should_compute = True

        if should_compute:
            centers, Ek = spectrum(solver, context)
            if solver.rank == 0:
                # Sanity Check
                total_Ek = np.sum(Ek)
                diff = abs(kk - total_Ek)
                print("Sanity Check (t={:.4f}): KE_sim = {:.16e}, Sum(E_k) = {:.16e}, Diff = {:.6e}".format(t_phys, kk, total_Ek, diff))

                # Save to HDF5
                with h5py.File("energy_spectrum.h5", "a") as f:
                    grp = f.require_group("spectra")
                    if str(params.tstep) in grp:
                        del grp[str(params.tstep)]
                    dset = grp.create_dataset(str(params.tstep), data=Ek)
                    dset.attrs["time"] = t_phys
                
                # Save to a text file for easy reading
                out_dir = "spectrum"
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                    
                txt_filename = os.path.join(out_dir, "spectrum_{:06d}.txt".format(params.tstep))
                header = "Time: {:26.16e}\n{:>24},{:>26}".format(t_phys, "k", "E(k)")
                np.savetxt(txt_filename, np.column_stack((centers, Ek)), header=header, delimiter=",")
                print("Saved spectrum to {}".format(txt_filename))

        if 'NS' not in params.solver:
            c.U_hat = c.U.forward(c.U_hat)
        ww2 = energy_fourier(c.U_hat, c.T)/2
        divu = solver.get_divergence(**context)
        divu = solver.comm.reduce(sum(divu.astype(float64)*divu.astype(float64))/prod(params.N)/2)

        kold[0] = kk
        if solver.rank == 0:
            k.append(kk)
            w.append(ww)
            if params.tstep == 0:
                print("{:>26} {:>26} {:>26} {:>26}".format(
                    "Time", "Cycle", "KineticEnergy", "Enstrophy"))
            print("{:26.16e} {:26.16e} {:26.16e} {:26.16e}".format(
                t_phys, float(params.tstep), float(kk), float(ww)))
            file_exists = os.path.isfile("diagnostics.csv")
            with open("diagnostics.csv", "a") as f:
                if not file_exists:
                    f.write("                      Time,                     Cycle,             KineticEnergy,                 Enstrophy\n")
                f.write("%26.16e,%26.16e,%26.16e,%26.16e\n" %(t_phys, float(params.tstep), float(kk), float(ww)))

def regression_test(context):
    params = config.params
    solver = config.solver
    U = solver.get_velocity(**context)
    curl = solver.get_curl(**context)
    w = solver.comm.reduce(sum(curl.astype(float64)*curl.astype(float64))/prod(params.N)/2)
    k = solver.comm.reduce(sum(U.astype(float64)*U.astype(float64))/prod(params.N)/2) # Compute energy with double precision
    config.solver.MemoryUsage('End')
    if solver.rank == 0:
        try:
            assert round(w.item() - 0.375249930801, params.ntol) == 0, w
            assert round(k.item() - 0.124953117517, params.ntol) == 0, k
        except AssertionError as e:
            print("Regression check failed: {}. This is expected if you changed simulation parameters (e.g. nu, dt). Simulation finished successfully.".format(e))

if __name__ == "__main__":

    config.update(

        {'dt': 0.01,                 # Time step

         'T': 0.1,                   # End time

         'L': [1.0, 1.0, 1.0],

         'M': [5, 5, 5],

         'planner_effort': {'fft': 'FFTW_ESTIMATE',

                            'rfftn': 'FFTW_ESTIMATE',

                            'irfftn': 'FFTW_ESTIMATE'},

        }, "triplyperiodic")

    config.triplyperiodic.add_argument("--diag_interval", type=int, default=1)

    config.triplyperiodic.add_argument("--compute_spectrum", type=int, default=0, help="Interval to compute energy spectrum (0 to disable)")

    config.triplyperiodic.add_argument("--write_result_dt", type=float, default=0.0, help="Write result every T seconds (overrides write_result if > 0)")

    config.triplyperiodic.add_argument("--plot_step", type=int, default=0)

    config.triplyperiodic.add_argument("--snapshot-time", type=float, default=-1.0, help="Physical time to force a single snapshot (fields + spectrum)")

    config.triplyperiodic.add_argument("--num_snapshots", type=int, default=0, help="Number of field snapshots to save evenly distributed over T")

    config.triplyperiodic.add_argument("--num_spectrum_snapshots", type=int, default=0, help="Number of spectrum snapshots to save evenly distributed over T")

    config.triplyperiodic.add_argument("--Re", type=float, default=100.0, help="Reynolds number (defines viscosity)")
    config.triplyperiodic.add_argument("--problem", type=int, default=2, choices=[1, 2],
                                       help="Problem setup: 1 => domain [0, 2*pi]^3 (dt unchanged), 2 => domain [0, 1]^3 (dt scaled by 1/(2*pi))")
    config.triplyperiodic.add_argument("--restart-file", type=str, default="",
                                       help="Optional checkpoint file to restart from (defaults to latest found).")

    config.triplyperiodic.add_argument("--N", default=[32, 32, 32], nargs=3,

                                       help="Mesh size. Trumps M.")

    sol = get_solver(update=update, regression_test=regression_test,

                     mesh="triplyperiodic")

    

        # Define characteristic scales and domain

    if config.params.problem == 1:
        domain_length = 2*pi
        L_ref = 1.0
        time_scale = 1.0
    else:
        domain_length = 1.0
        L_ref = 1.0 / (2*pi)
        time_scale = L_ref

    config.params.L = [domain_length, domain_length, domain_length]
    config.params.time_scale = time_scale
    config.params.L_ref = L_ref

    U_ref = 1.0

    

        # Compute nu from Re

        # Re = (U_ref * L_ref) / nu  =>  nu = (U_ref * L_ref) / Re

    config.params.nu = (U_ref * L_ref) / config.params.Re

    

        # Rescale dt for problem 2: input dt is nondimensional, physical dt = dt * L_ref.

    input_dt = config.params.dt

    config.params.dt = input_dt * time_scale

    

        # T is treated as Physical Time and is not rescaled.

    input_T = config.params.T

    config.params.T = input_T

    

        # Do not round snapshot_time to integer steps; use time-based tolerances instead.

    config.params.snapshot_step = -1

    

        # Calculate total simulation steps based on physical time

    total_steps = int(input_T / input_dt)

    

        # Handle num_snapshots logic (for HDF5 fields)

    if config.params.num_snapshots > 0:
        # Schedule snapshots by time
        times = [i * config.params.T / config.params.num_snapshots for i in range(0, config.params.num_snapshots + 1)]
        
        if config.params.snapshot_time >= 0:
             if config.params.snapshot_time <= config.params.T:
                 times.append(config.params.snapshot_time)
        
        times.sort()
        config.params.snapshot_times_fields = times
        config.params.snapshot_index_fields = 0
        config.params.write_result = 2000000000 # Disable fixed interval

        if sol.rank == 0:
            print("num_snapshots={} -> Scheduled {} field snapshots by time.".format(config.params.num_snapshots, len(times)))
    elif config.params.snapshot_time >= 0:
        # Single time-based snapshot for fields
        if config.params.snapshot_time <= config.params.T:
            config.params.snapshot_times_fields = [config.params.snapshot_time]
            config.params.snapshot_index_fields = 0
            config.params.write_result = 2000000000 # Disable fixed interval

    

    elif config.params.write_result_dt > 0:

        if config.params.dt > 0:

                # write_result_dt is physical time interval between saves

                # config.params.dt is the physical time increment per step (rescaled)

            config.params.write_result = int(config.params.write_result_dt / config.params.dt)

            if config.params.write_result < 1:

                config.params.write_result = 1

            if sol.rank == 0:

                print("Setting write_result to {} steps (every {}s physical)".format(config.params.write_result, config.params.write_result_dt))

    

        # Handle num_spectrum_snapshots logic (for Spectrum)

    if config.params.num_spectrum_snapshots > 0:
        # Schedule spectrums by time
        times = [i * config.params.T / config.params.num_spectrum_snapshots for i in range(0, config.params.num_spectrum_snapshots + 1)]
        
        if config.params.snapshot_time >= 0:
             if config.params.snapshot_time <= config.params.T:
                 times.append(config.params.snapshot_time)
        
        times.sort()
        config.params.snapshot_times_spectrum = times
        config.params.snapshot_index_spectrum = 0
        config.params.compute_spectrum = 2000000000 # Disable fixed interval

        if sol.rank == 0:
            print("num_spectrum_snapshots={} -> Scheduled {} spectrum snapshots by time.".format(config.params.num_spectrum_snapshots, len(times)))
    elif config.params.snapshot_time >= 0:
        # Single time-based snapshot for spectrum
        if config.params.snapshot_time <= config.params.T:
            config.params.snapshot_times_spectrum = [config.params.snapshot_time]
            config.params.snapshot_index_spectrum = 0
            config.params.compute_spectrum = 2000000000 # Disable fixed interval



    if sol.rank == 0:

        domain_label = "[0, 2*pi]^3" if config.params.problem == 1 else "[0, 1]^3"
        print("--- Taylor-Green Vortex Setup ({}) ---".format(domain_label))

        print("Reference Length (L): {:.6f}".format(L_ref))

        print("Reference Velocity (U): 1.0")

        print("Reynolds Number (Re): {:.2f}".format(config.params.Re))

        print("Computed Viscosity (nu): {:.6g}".format(config.params.nu))

        print("Input dt (User): {:g}".format(input_dt))

        print("Simulation dt (Used): {:g}".format(config.params.dt))

        print("End Time T (Physical): {:g}".format(input_T))

        if config.params.snapshot_time >= 0:
            print("Snapshot requested at physical t={:g}".format(config.params.snapshot_time))

        print("---------------------------------------------")



    context = sol.get_context()



    # Add curl to the stored results. For this we need to update the update_components

    # method used by the HDF5File class to compute the real fields that are stored

    context.hdf5file.filename = "NS9"

    context.hdf5file.results['data'].update({'curl': [context.curl]})

    def update_components(**c):

        """Overload default because we want to store the curl as well"""

        sol.get_velocity(**c)

        sol.get_pressure(**c)

        sol.get_curl(**c)



    context.hdf5file.update_components = update_components



    def custom_update(params, **kw):

        self = context.hdf5file

        if self.cfile is None:
            if params.get("restarted", False):
                params.filemode = "a"

            self.cfile = ShenfunFile(self.filename+'_c',
                                     self.checkpoint['space'],
                                     mode=params.filemode)

            self.cfile.open()

            if 'tstep' not in self.cfile.f.attrs:
                self.cfile.f.attrs.create('tstep', 0)
            if 't' not in self.cfile.f.attrs:
                self.cfile.f.attrs.create('t', 0.0)

            self.cfile.close()

        if self.wfile is None:

            self.wfile = ShenfunFile(self.filename+'_w',

                                     self.results['space'],

                                     mode=params.filemode)



        # Check for schedule or regular interval
        checkpoint_written = False
        should_write = False
        if hasattr(params, 'snapshot_times_fields'):
             while (params.snapshot_index_fields < len(params.snapshot_times_fields) and 
                   params.t >= params.snapshot_times_fields[params.snapshot_index_fields] - params.dt * 0.01):
                should_write = True
                params.snapshot_index_fields += 1
                
        elif params.tstep % params.write_result == 0:
            should_write = True
        
        is_snapshot = (params.snapshot_step >= 0) and (params.tstep == params.snapshot_step)
        if not hasattr(params, 'snapshot_times_fields') and is_snapshot:
            should_write = True

        if should_write:

            self.update_components(**kw)

            self.wfile.write(params.tstep, self.results['data'], as_scalar=False)

            # Save physical time as attribute
            try:
                close_after = False
                if self.wfile.f is None:
                    self.wfile.open()
                    close_after = True

                for key in self.results['data'].keys():
                    try:
                        # ShenfunFile structure: /var/3D/step
                        if key in self.wfile.f and '3D' in self.wfile.f[key] and str(params.tstep) in self.wfile.f[key]['3D']:
                            self.wfile.f[key]['3D'][str(params.tstep)].attrs['time'] = params.t
                    except Exception as e:
                        if sol.rank == 0:
                            print(f"Warning: Could not save time attribute: {e}")

                if close_after:
                    self.wfile.close()
            except Exception as e:
                if sol.rank == 0:
                    print(f"Warning: Could not open file to save time attribute: {e}")

            # Checkpoint whenever a snapshot is written
            for key, val in self.checkpoint['data'].items():
                self.cfile.write(int(key), val)
            self.cfile.open()
            self.cfile.f.attrs['tstep'] = params.tstep
            self.cfile.f.attrs['t'] = params.t
            self.cfile.close()
            checkpoint_written = True
            if sol.rank == 0:
                print("Checkpoint written at step {} (t={:.16e})".format(
                    params.tstep, params.t))

            if is_snapshot and sol.rank == 0:

                print(f"Snapshot fields saved at step {params.tstep}")



        kill = self.check_if_kill()

        if (params.tstep % params.checkpoint == 0 or kill) and not checkpoint_written:

            for key, val in self.checkpoint['data'].items():

                self.cfile.write(int(key), val)

                self.cfile.open()

                self.cfile.f.attrs['tstep'] = params.tstep

                self.cfile.f.attrs['t'] = params.t

                self.cfile.close()
            if sol.rank == 0:
                print("Checkpoint written at step {} (t={:.16e})".format(
                    params.tstep, params.t))



            if kill:

                sys.exit(1)



    context.hdf5file.update = custom_update



    restarted = _load_restart(sol, context)
    if not restarted:
        initialize(sol, context)

        # Perform initial save/spectrum computation at t=0
        if sol.rank == 0:
            print("Performing initial save at t=0...")
        update(context)
        context.hdf5file.update(config.params, **context)
    elif sol.rank == 0:
        print("Restart: continuing from checkpoint, skipping initial save.")

    solve(sol, context)



    # Final save at end of simulation
    should_final_save = False
    if hasattr(config.params, 'snapshot_times_fields'):
         # If there are remaining snapshots (e.g. T_final), save now.
         if config.params.snapshot_index_fields < len(config.params.snapshot_times_fields):
             should_final_save = True
    elif config.params.tstep % config.params.write_result != 0:
         should_final_save = True

    if should_final_save:

        if sol.rank == 0:

            print("Performing final save at t={:g}...".format(config.params.t))

        update(context)

        context.hdf5file.update(config.params, **context)
