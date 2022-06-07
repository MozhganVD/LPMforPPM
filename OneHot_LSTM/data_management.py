import pm4py
from pm4py.algo.filtering.log.attributes import attributes_filter
from pm4py.algo.filtering.log.end_activities import end_activities_filter


def filter_activities_with_start_name(start_name, data_address):
    log = pm4py.read_xes(data_address)
    activities = attributes_filter.get_attribute_values(log, "concept:name")
    W_activities = [k for k, v in activities.items() if k.startswith(start_name)]

    filtered_log = attributes_filter.apply_events(log, W_activities,
                                                  parameters={
                                                      attributes_filter.Parameters.CASE_ID_KEY: "case:concept:name",
                                                      attributes_filter.Parameters.ATTRIBUTE_KEY: "concept:name",
                                                      attributes_filter.Parameters.POSITIVE: True})
    return filtered_log


# process_model = pm4py.discover_bpmn_inductive(W_log)
# #
# pm4py.view_bpmn(process_model)


# W_log = filter_activities_with_start_name('W', '../Datasets/12689204/BPI_Challenge_2012/BPI_Challenge_2012.xes')
# df = pm4py.convert_to_dataframe(W_log)
# df.to_csv('../Datasets/12689204/BPI_Challenge_2012/BPI_Challenge_2012_W_Activities.csv')

##all data to csv
log = pm4py.read_xes('../../Datasets/Road_Traffic_Fine_Management_Process/Road_Traffic_Fine_Management_Process.xes')
# df = pm4py.convert_to_dataframe(log)
# end_activities = end_activities_filter.get_end_activities(df)
end_activities = end_activities_filter.get_end_activities(log)
filtered_log = end_activities_filter.apply(log, ["Send for Credit Collection"])
df = pm4py.convert_to_dataframe(filtered_log)
# filtered_df = end_activities_filter.apply(df, ["pay compensation"],
#                                           parameters={end_activities_filter.Parameters.CASE_ID_KEY: "case:concept:name",
#                                                       end_activities_filter.Parameters.ACTIVITY_KEY: "concept:name"})
df.to_csv('../../Datasets/Road_Traffic_Fine_Management_Process/TrafficFine_Credit.csv')
