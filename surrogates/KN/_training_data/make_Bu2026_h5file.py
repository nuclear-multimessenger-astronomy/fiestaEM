from fiesta.utils import convert_POSSIS_outputs_to_h5



possis_dirs = ["/work/koehn1/possis_runs_Bu2025/batch2/outputs",
               "/work/koehn1/possis_runs_Bu2025/batch1/outputs",
               "/work/koehn1/possis_runs_Bu2026/lower_dyn_masses/outputs", 
               "/work/koehn1/possis_runs_Bu2026/higher_dyn_velocities/outputs",
               "/work/koehn1/possis_runs_Bu2026/higher_dyn_velocities_batch2/outputs",
               "/work/koehn1/possis_runs_Bu2026/higher_dyn_velocities_pleiadi/outputs",
               "/work/koehn1/possis_runs_Bu2026/lower_wind_masses/outputs",
               "/work/koehn1/possis_runs_Bu2026/higher_wind_masses/outputs", 
               ]

convert_POSSIS_outputs_to_h5(possis_dirs=possis_dirs,
                             outfile="Bu2026_raw_data.h5")
