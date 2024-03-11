from preprocessing import Preprocessing
from preproc_split import PreprocSplit
from plotting_meta.meta_plotting import MetaPlotting
from plotting_df.df_plotting import DfPlotting
from plotting_df.extracted_plotting import ExtractedPlotting
from plotting_df.filtered_plotting import FilteredPlotting
from ts_clustering import TsClustering
from ts_clustering_experimental import TsClusteringExperimental
from ts_clustering_plotting import TsClusteringPlotting
from ts_eval import TsEval
from regression import Regression

def Main(preprocessing, preproc_split, plotting_meta, plotting_df, plotting_extracted, plotting_filtered, ts_clustering,
         ts_clustering_experimental, ts_clustering_plotting, ts_eval, regression, ts_sample_value):
    print("Main function called")
    if preprocessing:
        from load_dans import all_data as data
        print("Preprocessing data")
        Preprocessing(data, ts_samples=ts_sample_value, meta_lower_bound=60, empty_charge=60, streak_percentage=0.2,
                    should_filter_1911001328A_2_and_1911001328A_1=True, export_meta=True, export_extracted=True, export_filtered=False,
                    export_all=True, export_specific_id=False, id_to_export="1911001328A_2", strict_charge_extract=True, diffs=False)
        
    if preproc_split:
        print("Splitting preprocessed data")
        PreprocSplit(get_immediate_features=True, get_intermediate_features=True, get_final_features=True)

    if plotting_meta:
        print("Plotting meta data")
        MetaPlotting(connectiondurationa=True, connectiondurationa_threshold=8640, connectiondurationb=True, connectiondurationb_threshold=8640
                , covtime=True, cov=True, currentdifference=True, featuresvhalfminutes=True, hourconnected=True, timeencodingplot=True,
                voltage_difference=True)
    
    if plotting_df:
        print("Plotting df data")
        DfPlotting()

    if plotting_extracted:
        print("Plotting extracted data")
        ExtractedPlotting(ts_samples=ts_sample_value)

    if plotting_filtered:
        print("Plotting filtered data")
        FilteredPlotting()

    if ts_clustering:
        print("Clustering time series")
        TsClustering(num_cores=-1, num_clusters_1_phase_range=range(2, 16), num_clusters_3_phase_range=range(2, 16), use_all_3_phase_data=True,
                     distance_metric='dtw', split_phases=True, ts_samples=ts_sample_value)
        
    if ts_clustering_experimental:
        print("Clustering time series experimental")
        TsClusteringExperimental(num_cores=-1, num_clusters=3, distance_metric='dtw', ts_samples=20, use_current_type=True,
                                 use_all_3_phase_data=True, Use_time=False, use_charging_point=False, use_floor=True, use_weekend=True,
                                 use_maxvoltage=False, use_maxcurrent=False, use_energy_uptake=False, use_average_voltage=False,
                                 use_average_current=False)
        
    if ts_clustering_plotting:
        print("Plotting time series clustering")
        TsClusteringPlotting(phase="1-Phase", ts_samples=ts_sample_value, tot_clusters=10)

    if ts_eval:
        print("Evaluating time series clustering")
        TsEval(ts_samples=ts_sample_value)

    if regression:
        print("Performing regression")
        Regression(num_cores=-1, ts_samples=ts_sample_value, include_ts_clusters=True, phase="3-Phase", clusters=10,
                   test_size=0.2, random_state=42, n_estimators=100)

    print("Main function finished")
    
Main(preprocessing=False, preproc_split=False, plotting_meta=False, plotting_df=False, plotting_extracted=False, plotting_filtered=False,
     ts_clustering=False, ts_clustering_experimental=False, ts_clustering_plotting=False, ts_eval=False, regression=True, ts_sample_value=60)