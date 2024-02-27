
#run preprocessing.py main function with args
exec(open('model/preprocessing.py').read())

#run connection_duration_a.py
exec(open('model/plotting_meta/connection_duration_a.py').read())
#run connection_duration_b.py
exec(open('model/plotting_meta/connection_duration_b.py').read())
#run cov_time.py
exec(open('model/plotting_meta/cov_time.py').read())
#run cov.py
exec(open('model/plotting_meta/cov.py').read())
#run current_difference.py
exec(open('model/plotting_meta/current_difference.py').read())
#run features_v_half_minutes.py
exec(open('model/plotting_meta/features_v_half_minutes.py').read())
#run hour_connected.py
exec(open('model/plotting_meta/hour_connected.py').read())
#run time_encoding_plot.py 
exec(open('model/plotting_meta/time_encoding_plot.py').read())
#run voltage_difference.py
exec(open('model/plotting_meta/voltage_difference.py').read())


#run preproc_split.py
exec(open('model/preproc_split.py').read())

#run ts_clustering.py
exec(open('model/ts_clustering.py').read())
#run ts_clustering_plotting.py
exec(open('model/ts_clustering_plotting.py').read())
#run ts_eval.py
exec(open('model/ts_eval.py').read())
#run ts_eval_plotting.py
exec(open('model/ts_eval_plotting.py').read())