from load_dans import all_data as data
from preprocessing import Preprocessing
from plotting_meta.meta_plotting import MetaPlotting
from plotting_df.df_plotting import DfPlotting
from plotting_df.extracted_plotting import ExtractedPlotting
from plotting_df.filtered_plotting import FilteredPlotting

def Main(preprocessing, plotting_meta, plotting_df, plotting_extracted, plotting_filtered):
    print("Main function called")

    if preprocessing:
        print("Preprocessing data")
        Preprocessing(data, ts_samples=2, meta_lower_bound=60, empty_charge=60, streak_percentage=0.2,
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
    
Main(preprocessing=True, plotting_meta=True, plotting_df=True, plotting_extracted=True, plotting_filtered=True)
    
