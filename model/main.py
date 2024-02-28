from load_dans import all_data as data
from preprocessing import Preprocessing
from plotting_meta.meta_plotting import MetaPlotting

Preprocessing(data, ts_samples=2, meta_lower_bound=60, empty_charge=60, streak_percentage=0.2,
              should_filter_1911001328A_2_and_1911001328A_1=True, export_meta=True, export_extracted=True, export_filtered=False,
              export_all=True, export_specific_id=False, id_to_export="1911001328A_2", strict_charge_extract=True, diffs=False)

MetaPlotting(connectiondurationa=True, connectiondurationa_threshold=8640, connectiondurationb=True, connectiondurationb_threshold=8640
             , covtime=True, cov=True, currentdifference=True, featuresvhalfminutes=True, hourconnected=True, timeencodingplot=True,
             voltage_difference=True)