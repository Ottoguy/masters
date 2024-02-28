from plotting_meta.connection_duration_a import ConnectionDurationA
from plotting_meta.connection_duration_b import ConnectionDurationB
from plotting_meta.cov_time import CovTime
from plotting_meta.cov import Cov
from plotting_meta.current_difference import CurrentDifference
from plotting_meta.features_v_half_minutes import FeaturesVHalfMinutes
from plotting_meta.hour_connected import HourConnected
from plotting_meta.time_encoding_plot import TimeEncodingPlot
from plotting_meta.voltage_difference import VoltageDifference

def MetaPlotting(connectiondurationa, connectiondurationa_threshold, connectiondurationb, connectiondurationb_threshold, covtime, cov,
                 currentdifference, featuresvhalfminutes, hourconnected, timeencodingplot, voltage_difference):
    # Set the threshold (half-minutes) for disregarding EVs in the ConnectionDurationA plot
    # 5760 is 48hrs
    # 8640 is 72hrs
    if connectiondurationa:
        print("Meta plotting ConnectionDurationA")
        ConnectionDurationA(threshold=connectiondurationa_threshold)
    if connectiondurationb:
        print("Meta plotting ConnectionDurationB")
        ConnectionDurationB(threshold=connectiondurationb_threshold)
    if covtime:
        print("Meta plotting CovTime")
        CovTime()
    if cov:
        print("Meta plotting Cov")
        Cov()
    if currentdifference:
        print("Meta plotting CurrentDifference")
        CurrentDifference()
    if featuresvhalfminutes:
        print("Meta plotting FeaturesVHalfMinutes")
        FeaturesVHalfMinutes()
    if hourconnected:
        print("Meta plotting HourConnected")
        HourConnected()
    if timeencodingplot:
        print("Meta plotting TimeEncodingPlot")
        TimeEncodingPlot()
    if voltage_difference:
        print("Meta plotting VoltageDifference")
        VoltageDifference()
    print("Meta plotting done")