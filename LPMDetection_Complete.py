import glob
import time
import numpy as np
import os
import pm4py
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.petri_net.importer import importer as pnml_import
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments
from pm4py.objects.petri_net.utils import petri_utils
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.utils import reachability_graph
from pm4py.visualization.transition_system import visualizer as ts_visualizer
from pm4py.objects.petri_net.exporter import exporter as pnml_exporter
from pm4py.algo.conformance.alignments.petri_net.algorithm import Parameters
from pm4py.algo.filtering.log.attributes import attributes_filter
import pandas as pd
from utils import DatasetManager


# parameters = {Parameters.PARAM_ALIGNMENT_RESULT_IS_SYNC_PROD_AWARE: True}


def custom_alignment(net, im, fm, log):
    model_cost_function = dict()
    sync_cost_function = dict()
    for t in net.transitions:
        # if the label is not None, we have a visible transition
        if t.label is not None:
            # associate cost 1000 to each move-on-model associated to visible transitions
            model_cost_function[t] = 100000000000000
            # associate cost 0 to each move-on-log
            sync_cost_function[t] = 0
        else:
            # associate cost 1 to each move-on-model associated to hidden transitions
            model_cost_function[t] = 0

    parameters = {
        alignments.Variants.VERSION_STATE_EQUATION_A_STAR.value.Parameters.PARAM_MODEL_COST_FUNCTION: model_cost_function,
        alignments.Variants.VERSION_STATE_EQUATION_A_STAR.value.Parameters.PARAM_SYNC_COST_FUNCTION: sync_cost_function,
        Parameters.PARAM_ALIGNMENT_RESULT_IS_SYNC_PROD_AWARE: True}
    aligned_traces = alignments.apply_log(log, net, im, fm, parameters=parameters)
    return aligned_traces


def wrap_lpm(net, im, fm):
    ts = reachability_graph.construct_reachability_graph(net, im, use_trans_name=True)

    # add a backloop from the final mark to the initial one
    final_places_names = list()
    initial_places_names = list()
    count_add = 1
    t_1 = PetriNet.Transition("n_add_" + str(count_add), None)
    count_add += 1
    net.transitions.add(t_1)

    for p in fm:
        petri_utils.add_arc_from_to(p, t_1, net)
        final_places_names.append(p.name)
    for p in im:
        petri_utils.add_arc_from_to(t_1, p, net)
        initial_places_names.append(p.name)

    # add a silent transition towards a new final marking --> not needed
    # t_2=PetriNet.Transition("n_add_"+str(count_add),None)
    # count_add+=1
    # net.transitions.add(t_2)
    new_final_place = PetriNet.Place("p_fin")
    net.places.add(new_final_place)
    #     for p in fm:
    #         fm[p]=0
    #         petri_utils.add_arc_from_to(p, t_2, net)

    #     petri_utils.add_arc_from_to(t_2,new_final_place,net)
    final_marking = Marking()
    final_marking[new_final_place] = 1

    # add a shortcircuit to the final place to every marking of the model

    for state in ts.states:
        marking_labels = [f'n{nodenumber[:-1]}' for nodenumber in str(state).split('n') if nodenumber]
        t_s = PetriNet.Transition("n_add_" + str(count_add), None)
        count_add += 1
        net.transitions.add(t_s)
        petri_utils.add_arc_from_to(t_s, new_final_place, net)
        for marking_p in marking_labels:
            # if marking_p not in final_places_names and marking_p not in initial_places_names:
            for p in net.places:
                if p.name == marking_p:
                    petri_utils.add_arc_from_to(p, t_s, net)

    # gviz = ts_visualizer.apply(ts, parameters={ts_visualizer.Variants.VIEW_BASED.value.Parameters.FORMAT: "png"})
    # ts_visualizer.view(gviz)
    return net, im, final_marking


####### start here ###########

# import log once
folder_address = "../../Datasets/000_Experimemts/BPIC2011/f1/"
Original_log = xes_importer.apply(folder_address + 'BPIC11_f1_Trunc36.xes')
Original_dataframe = log_converter.apply(Original_log, variant=log_converter.Variants.TO_DATA_FRAME)

# Original_dataframe['event_nr'] = ""
# idx = 0
# for case in Original_dataframe['case:concept:name'].unique():
#     case_length = len(Original_dataframe[Original_dataframe['case:concept:name']==case])
#     for i in range(1, case_length+1):
#         Original_dataframe.at[idx, 'event_nr'] = i
#         idx += 1

Original_dataframe['Case_duplicate'] = Original_dataframe['case:concept:name']
start = time.time()
case_id = "case:concept:name"
event_nr_col = "event_nr"
manager = DatasetManager('production')
# generating prefixes traces
Prefixes_dataframe = manager.generate_prefix_data(Original_dataframe, 2, 36)
Prefixes_log = log_converter.apply(Prefixes_dataframe, variant=log_converter.Variants.TO_EVENT_LOG)
# for each LPM [note: this is a prototype, I'm doing it for one]:
LPMs_file = glob.glob(folder_address + "Discriminative/Non_Similar_LPMs/*.pnml")
model_move_counter = set()
for net_file in LPMs_file:
    lpm_number = os.path.basename(net_file).split(".")[0].split("_")[1]
    # lpm_number = os.path.basename(net_file).split(".")[0]
    lpm_col = "LPM_%s" % lpm_number
    print(lpm_number)
    # net_file = "C:/Users/20211286/Documents/PhD/Datasets/BPIC2012/bpic2012_O_Accepted/LPMs/20.apnml"
    t_names = list()
    lpm_acts = list()
    net, im, fm = pnml_import.apply(net_file)

    for t in net.transitions:
        t_names.append(t)
        if t.label != None:
            lpm_acts.append(t.label)

    number_act_lpm = len(lpm_acts)
    if number_act_lpm > 2:
        print(number_act_lpm)
    lpm_acts.sort()
    # gviz = pn_visualizer.apply(net, im, fm)
    # pn_visualizer.view(gviz)
    # build a new event log where for each trace only events corresponding to LPMs models are kept.
    print('start filtering data')
    tracefilter_log_pos = attributes_filter.apply_events(Prefixes_log, [transition.label for transition in t_names if
                                                                        not transition.label is None],
                                                         parameters={
                                                             attributes_filter.Parameters.ATTRIBUTE_KEY: 'concept:name',
                                                             attributes_filter.Parameters.POSITIVE: True})

    # we keep all the cases which have a number of activities greater than or equal to the number of activities in LPM
    filtered_log = pm4py.filter_log(lambda x: len(x) >= number_act_lpm, tracefilter_log_pos)
    # remove cases with duplicate activities, we need to keep cases which has a unique number of activities greater
    # than activities in LPM

    for x in filtered_log:
        uniques_events = []
        for ev in x:
            uniques_events.append(ev._dict['concept:name'])
        if len(np.unique(uniques_events)) < number_act_lpm:
            target_case = x[0]._dict['Case_duplicate']
            filtered_log = attributes_filter.apply(filtered_log, [target_case],
                                                   parameters={
                                                       attributes_filter.Parameters.ATTRIBUTE_KEY: "Case_duplicate",
                                                       attributes_filter.Parameters.POSITIVE: False})

    # aligned_traces = custom_alignment(net_2, im_2, fm_2, tracefilter_log_pos)  # generate the aligned traces;
    print('start creating alignment checking')
    aligned_traces = custom_alignment(net, im, fm, filtered_log)
    Final_log = []
    print('start creating lpm feature')
    print(len(aligned_traces))
    for ii, aligned_trace in enumerate(aligned_traces):
        # print(ii)
        Case = log_converter.apply(filtered_log[ii], variant=log_converter.Variants.TO_DATA_FRAME)
        Case[case_id] = filtered_log[ii].attributes['concept:name']
        Case[lpm_col] = np.False_
        aligned_acts = set()
        for jj, el in enumerate(aligned_trace["alignment"]):
            if ">>" in el[1]:
                continue
            else:
                aligned_acts.add(el[1][0])

        idx_counter = 0
        if len(aligned_acts) >= number_act_lpm:
            for jj, el in enumerate(aligned_trace["alignment"]):
                if ">>" in el[1]:
                    if None in el[1]:
                        continue
                    else:
                        # if el[1][0] == ">>":
                        #     # print('move on model')
                        #     model_move_counter.add(ii)
                        # else:
                        idx_counter += 1
                        continue
                else:
                    Case.at[idx_counter, lpm_col] = True
                    idx_counter += 1

        Final_log.append(Case)

    if len(Final_log) < 2:
        continue
    Tracefilter_LPMs_log = pd.concat(Final_log, axis=0)
    Prefixes_dataframe[lpm_col] = np.False_
    # All_Cases = Original_log_LPMs[case_id].unique()
    LPMs_Cases = Tracefilter_LPMs_log[case_id].unique()
    for case in LPMs_Cases:
        # print(case)
        True_cases = Tracefilter_LPMs_log[Tracefilter_LPMs_log[case_id] == case]
        True_events = True_cases.loc[True_cases[lpm_col] == True]
        for event_number in True_events[event_nr_col]:
            target = int(Prefixes_dataframe.index[(Prefixes_dataframe[case_id] == case) &
                                                  (Prefixes_dataframe[event_nr_col] == event_number)].values)

            Prefixes_dataframe.at[target, lpm_col] = True

print(model_move_counter)
print(time.time() - start)
Prefixes_dataframe.to_csv(folder_address + "Discriminative/BPIC11_f1_Trunc36_Discriminative_LPMs.csv", index=False)
