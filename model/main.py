from preprocessing import Preprocessing
from plotting_meta.meta_plotting import MetaPlotting
from plotting_df.df_plotting import DfPlotting
from plotting_df.extracted_plotting import ExtractedPlotting
from plotting_df.filtered_plotting import FilteredPlotting
from ts_clustering import TsClustering

def Main(preprocessing, plotting_meta, plotting_df, plotting_extracted, plotting_filtered, ts_clustering):
    print("Main function called")
    if preprocessing:
        from load_dans import all_data as data
        print("Preprocessing data")
        Preprocessing(data, ts_samples=10, meta_lower_bound=60, empty_charge=60, streak_percentage=0.2,
                    should_filter_1911001328A_2_and_1911001328A_1=True, export_meta=True, export_extracted=True, export_filtered=False,
                    export_all=True, export_specific_id=False, id_to_export="1911001328A_2", strict_charge_extract=True, diffs=False)

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
        ExtractedPlotting()

    if plotting_filtered:
        print("Plotting filtered data")
        FilteredPlotting()

    if ts_clustering:
        print("Clustering time series")
        TsClustering(num_cores=-1, num_clusters_1_phase_range=range(2, 3), num_clusters_3_phase_range=range(2, 3), use_all_3_phase_data=True,
                     distance_metric='dtw', ts_samples=10)

    print("Main function finished")
    
Main(preprocessing=True, plotting_meta=False, plotting_df=False, plotting_extracted=False, plotting_filtered=False, ts_clustering=True)
    
