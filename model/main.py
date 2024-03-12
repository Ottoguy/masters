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
from deep_regression import DeepLearningRegression
import pandas as pd

def Main(preprocessing, preproc_split, plotting_meta, plotting_df, plotting_extracted, plotting_filtered, ts_clustering,
         ts_clustering_experimental, ts_clustering_plotting, ts_eval, regression, deep_regression, ts_sample_value):
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
        #Regression(num_cores=-1, ts_samples=ts_sample_value, include_ts_clusters=True, phase="3-Phase", clusters=15,
                   #test_size=0.3, random_state=42, n_estimators=200)
        # Set the ranges of values for clusters, test_size, and n_estimators
        cluster_values = [3, 5, 8, 10, 12, 15]  # Update with your desired values
        test_size_values = [0.2, 0.4, 0.6]  # Update with your desired values
        n_estimators_values = [100, 200, 300]  # Update with your desired values

        # Create an empty DataFrame to store the results
        results_df = pd.DataFrame(columns=['Clusters', 'Test_Size', 'N_Estimators', 'MSE_Clusters'])

        # Iterate over values
        for clusters in cluster_values:
            for test_size in test_size_values:
                for n_estimators in n_estimators_values:
                    # Call the Regression function
                    mse_clusters = Regression(num_cores=-1, ts_samples=ts_sample_value, include_ts_clusters=True,
                                            phase="3-Phase", clusters=clusters, test_size=test_size,
                                            random_state=42, n_estimators=n_estimators)

                    # Record the results in the DataFrame
                    results_df = pd.concat([results_df, pd.DataFrame({
                        'Clusters': [clusters],
                        'Test_Size': [test_size],
                        'N_Estimators': [n_estimators],
                        'MSE_Clusters': [mse_clusters]
                    })], ignore_index=True)

        # Sort the DataFrame by 'MSE_Clusters' column
        results_df = results_df.sort_values(by='MSE_Clusters')

        # Save the sorted DataFrame to a CSV file
        csv_filename = 'regression_results_sorted.csv'
        results_df.to_csv(csv_filename, index=False)

        print(f"Results saved to {csv_filename}")
    
    if deep_regression:
        print("Performing deep regression")

        # Set the ranges of values for hyperparameters
        cluster_values = [15]  # Update with your desired values
        epochs_values = [400]  # Update with your desired values
        batch_size_values = [32]  # Update with your desired values
        layer1_units_values = [128]  # Update with your desired values
        layer2_units_values = [64]  # Update with your desired values
        dropout_rate_values = [0.4]  # Update with your desired values

        # Create an empty DataFrame to store the results
        results_df_dl = pd.DataFrame(columns=['Clusters', 'Test_Size', 'Epochs', 'Batch_Size', 'Layer1_Units', 'Layer2_Units', 'Dropout_Rate', 'MSE_Clusters_DL'])

        # Iterate over hyperparameter values
        for clusters in cluster_values:
            for epochs in epochs_values:
                for batch_size in batch_size_values:
                    for layer1_units in layer1_units_values:
                        for layer2_units in layer2_units_values:
                            for dropout_rate in dropout_rate_values:
                                # Call the DeepLearningRegression function
                                mse_clusters_dl = DeepLearningRegression(num_cores=-1, ts_samples=ts_sample_value, include_ts_clusters=True,
                                                                        clusters=clusters, test_size=0.3,
                                                                        random_state=42, epochs=epochs, batch_size=batch_size,
                                                                        layer1_units=layer1_units, layer2_units=layer2_units,
                                                                        dropout_rate=dropout_rate)

                                # Record the results in the DataFrame
                                results_df_dl = pd.concat([results_df_dl, pd.DataFrame({
                                    'Clusters': [clusters],
                                    'Epochs': [epochs],
                                    'Batch_Size': [batch_size],
                                    'Layer1_Units': [layer1_units],
                                    'Layer2_Units': [layer2_units],
                                    'Dropout_Rate': [dropout_rate],
                                    'MSE_Clusters_DL': [mse_clusters_dl]
                                })], ignore_index=True)

        # Sort the DataFrame by 'MSE_Clusters_DL' column
        results_df_dl = results_df_dl.sort_values(by='MSE_Clusters_DL')

        # Save the sorted DataFrame to a CSV file
        csv_filename_dl = 'deep_regression_results_sorted.csv'
        results_df_dl.to_csv(csv_filename_dl, index=False)

        print(f"Results saved to {csv_filename_dl}")

    print("Main function finished")
    
Main(preprocessing=False, preproc_split=False, plotting_meta=False, plotting_df=False, plotting_extracted=False, plotting_filtered=False,
     ts_clustering=False, ts_clustering_experimental=False, ts_clustering_plotting=False, ts_eval=False, regression=False,
     deep_regression=True, ts_sample_value=60)