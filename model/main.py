from preprocessing import Preprocessing
from preproc_split import PreprocSplit
from plotting_meta.meta_plotting import MetaPlotting
from plotting_df.df_plotting import DfPlotting
from plotting_df.extracted_plotting import ExtractedPlotting
from plotting_df.filtered_plotting import FilteredPlotting
from ts_clustering import TsClustering
from ts_clustering_plotting import TsClusteringPlotting
from ts_eval import TsEval
from regression import Regression
from deep_regression import DeepLearningRegression
from dl_merge import DLMerge
import pandas as pd
import os
from datetime import datetime
import time

def Main(preprocessing, preproc_split, plotting_meta, plotting_df, plotting_extracted, plotting_filtered, ts_clustering,
          ts_clustering_plotting, ts_eval, regression, deep_regression, ts_sample_value, merge_dl):
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
        
    if ts_clustering_plotting:
        print("Plotting time series clustering")
        TsClusteringPlotting(phase="1-Phase", ts_samples=ts_sample_value, tot_clusters=10)

    if ts_eval:
        print("Evaluating time series clustering")
        TsEval(ts_samples=ts_sample_value)
    
    if deep_regression:
        print("Performing deep regression")

        #How many samples should we use for the time series
        ts_sample_values = [60]  # Update with your desired values
        # Set the ranges of values for hyperparameters
        cluster_values = [10]  # Update with your desired values
        #750+ epochs best
        epochs_values = [1111]  # Update with your desired values
        #64 useless, 16 best 2024-03-17
        batch_size_values = [4, 8, 16]  # Update with your desired values
        #256 best, 32 not good 2024-03-17
        layer1_units_values = [192, 256, 320]  # Update with your desired values
        #64 seems to be as good as any higher value 2024-03-17
        layer2_units_values = [48, 64, 96]  # Update with your desired values
        #This layer maybe does not improve the model (1 is the same as not having the layer) 2024-03-18
        layer3_units_values = [1]  # Update with your desired values
        #0.3, 0.4, or 0.5 does not seem to matter much 2024-03-17
        dropout_rate_values = [0.4]  # Update with your desired values
        # Define the features to exclude one at a time
        #features_to_exclude = ['ChargingPoint','Floor','Weekend','TimeConnected_sin','TimeConnected_cos', 'MaxVoltage', 'MaxCurrent',
        #                   'FullyCharged', 'Current_Type', 'Energy_Uptake', 'AverageVoltageDifference', 'AverageCurrentDifference']
        features_to_exclude = ['None']
        #tanh performed the best for activation layer 1 2024-03-16
        activation_functions_layer1 = ['tanh', 'relu']
        #RELU performed by far the best for activation layer 2 2024-03-15
        activation_functions_layer2 = ['relu']
        #RELU performed by far the best for activation layer 3 2024-03-18
        activation_functions_layer3 = ['relu']
        #Embedding gives almost universally worse results 2024-03-16
        should_embed_features = [False]
        train_immediate=False
        train_barebones=False

        # Create an empty DataFrame to store the results
        results_df_dl = pd.DataFrame(columns=['RMSE_Clusters'])

        # Calculate total number of iterations
        total_iterations = (
            len(cluster_values)
            * len(epochs_values)
            * len(batch_size_values)
            * len(layer1_units_values)
            * len(layer2_units_values)
            * len(layer3_units_values)
            * len(dropout_rate_values)
            * len(features_to_exclude)
            * len(activation_functions_layer1)
            * len(activation_functions_layer2)
            * len(activation_functions_layer3)
            * len(should_embed_features)
        )

        total_epochs = 0

        for epoch in epochs_values:
            total_epochs += epoch

        start_time = time.time()
        
        print(f"Will iterate over {total_iterations} iterations for {total_epochs} epochs in total")

        # Iterate over hyperparameter values
        for ts_sample_value in ts_sample_values:
            for clusters in cluster_values:
                for epochs in epochs_values:
                    for batch_size in batch_size_values:
                        for layer1_units in layer1_units_values:
                            for layer2_units in layer2_units_values:
                                for layer3_units in layer3_units_values:
                                    for dropout_rate in dropout_rate_values:
                                        for feature_to_exclude in features_to_exclude:
                                            for activation_function_layer1 in activation_functions_layer1:
                                                for activation_function_layer2 in activation_functions_layer2:
                                                    for activation_function_layer3 in activation_functions_layer3:
                                                        for should_embed in should_embed_features:
                                                            # Call the DeepLearningRegression function
                                                            rmse_barebones_dl, rmse_immediate_dl, rmse_intermediate_dl, rmse_clusters_dl, mae_barebones_dl, mae_immediate_dl, mae_intermediate_dl, mae_clusters_dl = DeepLearningRegression(
                                                                                                    ts_samples=ts_sample_value,
                                                                                                    clusters=clusters, test_size=0.3,
                                                                                                    random_state=42, epochs=epochs, batch_size=batch_size,
                                                                                                    layer1_units=layer1_units, layer2_units=layer2_units, layer3_units=layer3_units,
                                                                                                    dropout_rate=dropout_rate, feature_to_exclude=feature_to_exclude, 
                                                                                                    layer1activation=activation_function_layer1, layer2activation=activation_function_layer2,
                                                                                                    layer3activation=activation_function_layer3, should_embed=should_embed,
                                                                                                    train_immediate=train_immediate, train_barebones=train_barebones)

                                                            # Record the results in the DataFrame
                                                            results_df_dl = pd.concat([results_df_dl, pd.DataFrame({
                                                                'RMSE_Clusters': [rmse_clusters_dl],
                                                                'RMSE_Intermediate': [rmse_intermediate_dl],
                                                                'RMSE_Immediate': [rmse_immediate_dl],
                                                                'RMSE_Barebones': [rmse_barebones_dl],
                                                                'MAE_Clusters': [mae_clusters_dl],
                                                                'MAE_Intermediate': [mae_intermediate_dl],
                                                                'MAE_Immediate': [mae_immediate_dl],
                                                                'MAE_Barebones': [mae_barebones_dl],
                                                                'TS_Samples': [ts_sample_value],
                                                                'Clusters': [clusters],
                                                                'Epochs': [epochs],
                                                                'Batch_Size': [batch_size],
                                                                'Layer1_Units': [layer1_units],
                                                                'Layer2_Units': [layer2_units],
                                                                'Layer3_Units': [layer3_units],
                                                                'Layer1Activation': [activation_function_layer1],
                                                                'Layer2Activation': [activation_function_layer2],
                                                                'Layer3Activation': [activation_function_layer3],
                                                                'Dropout_Rate': [dropout_rate],
                                                                'ExcludedFeature': [feature_to_exclude],
                                                                'ShouldEmbed': [should_embed]
                                                            })], ignore_index=True)

        # Sort the DataFrame by 'RMSE_intermediate_DL' column
        results_df_dl = results_df_dl.sort_values(by='RMSE_Clusters')

        output_folder = 'prints/dl_overview/'
        # Create a folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        # Get the current date and time
        current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Create the file name
        print(f"Creating the file {current_datetime}.csv")
        output_file = f"{output_folder}/{current_datetime}.csv"
        # Print desired_rows to a CSV file
        results_df_dl.to_csv(output_file, index=False)
        #Print path to the created file
        print(f"Results saved to {output_file}")

        end_time = time.time()
        execution_time = end_time - start_time
        print("Execution time: {} seconds".format(execution_time))
        print("Code execution completed.")
    
    if merge_dl:
        print("Merging deep learning results")
        DLMerge()

    print("Main function finished")
    
Main(preprocessing=False, preproc_split=False, plotting_meta=False, plotting_df=False, plotting_extracted=False, plotting_filtered=False,
     ts_clustering=False, ts_clustering_plotting=False, ts_eval=False, regression=False,
     deep_regression=True, ts_sample_value=60, merge_dl=True)