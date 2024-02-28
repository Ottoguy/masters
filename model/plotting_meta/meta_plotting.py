from plotting_meta.connection_duration_a import ConnectionDurationA

def MetaPlotting(connectiondurationa, connectiondurationa_threshold):
    # Set the threshold (half-minutes) for disregarding EVs in the ConnectionDurationA plot
    # 5760 is 48hrs
    # 8640 is 72hrs
    if connectiondurationa:
        ConnectionDurationA(threshold=connectiondurationa_threshold)