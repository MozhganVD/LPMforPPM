import glob

import numpy as np
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments
from pm4py.algo.conformance.alignments.petri_net.algorithm import Parameters
from pm4py.objects.petri_net.importer import importer as pnml_import
from pm4py.objects.process_tree import pt_operator
from pm4py.objects.process_tree import util as pt_util
from pm4py.visualization.petri_net import visualizer as pn_viz
from pm4py.visualization.process_tree import visualizer as pt_viz
from pm4py.algo.simulation.tree_generator import simulator as pt_gen
from pm4py.objects.conversion.process_tree import converter as pt_conv
from pm4py.objects.conversion.wf_net import converter as wf_net_converter
from pm4py.objects.process_tree.exporter import exporter as ptml_exporter
import pm4py.objects.process_tree.utils.generic as generic
from pm4py.objects.petri_net import semantics
from pm4py.objects.process_tree.obj import Operator

import re
import sys
import statistics
import networkx as nx
import time
import numpy

"""
il codice assume che le attivitià abbiano nomi univoci
"""


# In[4]:


def make_visible(data):
    def replacement(match):
        if match.group(6) != '$invisible$':
            return match.group(0)
        orig_ident = match.group(2)
        orig_name = match.group(4)
        orig_activity = match.group(6)
        ident = orig_ident
        name = f'{orig_ident} {orig_name}'.replace(' ',
                                                   '-')  # matches regular expression '^n[0-9]+-tau-' or contains '-tau-'
        activity = name
        return match.group(1) + ident + match.group(3) + name + match.group(5) + activity + match.group(7)

    return re.compile(
        '(<transition id=")([^"]*)(">.*?<name><text>)(.*?)(</text></name>.*?activity=")([^"]*)(".*?</transition>)').sub(
        replacement, data)


# OLD function; keeping it for reference reason. DO NOT USE IT
def set_features(node, out, curr_features):
    if len(node._get_children()) == 0:
        out[node._get_label()] = curr_features
        return out
    (n_p, n_c, l_s, l_e) = curr_features
    op = node._get_operator()
    # a node can be operator or a leaf
    if str(op) == "+":
        n_p = n_p + len(node.children) - 1
    elif str(op) == "X":
        n_c = n_c + len(node.children) - 1
    elif str(op) == "*":
        # I have to check the children to detect strictly loopable activities
        tau_child = False
        op_child = False
        for child in node.children:
            if (str(child._get_label()) == "None" or "-tau-" in str(child._get_label())) and str(
                    child._get_operator()) == "None":
                tau_child = True
            elif (str(child._get_label()) == "None" or "-tau-" in str(child._get_label())) and str(
                    child._get_operator()) != "None":
                op_child = True

        if tau_child == False or op_child == True:
            l_e = 1
        elif tau_child == True and op_child == False:
            l_s = 1

    for child in node.children:
        out = set_features(child, out, (
        n_p, n_c, l_s, l_e))  # paralleli, choice , strictly loop, extended loop quindi  [0,N], [0,N], [0,1], [0,1]
    return out


# In[5]:


# new functions to extract features
def merge_dictionaries(dictionaries):
    return {key: value for dictionary in dictionaries for (key, value) in dictionary.items()}


def depth_first(aggregate, leaf):
    def inner(tree):
        return aggregate(tree, map(inner, tree.children)) if tree.children else leaf(tree)

    return inner


def maximum_degree_of_parallelism_aggregate(tree, subresults):
    subresults = list(subresults)
    if tree.operator == Operator.PARALLEL:
        subdegrees = [max(subresult.values()) for subresult in subresults]
        total_degree = sum(subdegrees)
        return {node: total_degree - subdegree + degree for (subresult, subdegree) in zip(subresults, subdegrees) for
                (node, degree) in subresult.items()}
    return merge_dictionaries(subresults)


def degree_of_choice_aggregate(tree, subresults):
    if tree.operator == Operator.XOR:
        return {node: degree + len(tree.children) - 1 for subresult in subresults for (node, degree) in
                subresult.items()}
    return merge_dictionaries(subresults)


def strictly_loopable_aggregate(tree, subresults):
    if tree.operator == Operator.LOOP and all(child.operator is None for child in tree.children):
        return {node: True for subresult in subresults for node in subresult}
    return merge_dictionaries(subresults)


def loopable_aggregate(tree, subresults):
    if tree.operator == Operator.LOOP:
        return {node: True for subresult in subresults for node in subresult}
    return merge_dictionaries(subresults)


def collate_dictionaries(dictionaries):
    keys = {key for dictionary in dictionaries for key in dictionary.keys()}
    return {key: tuple(dictionary[key] for dictionary in dictionaries) for key in keys}


def feature_map(tree):
    return {key: (
    maximum_degree_of_parallelism, degree_of_choice, int(strictly_loopable), int(loopable and not strictly_loopable))
            for (key, (maximum_degree_of_parallelism, degree_of_choice, loopable, strictly_loopable)) in
            collate_dictionaries((depth_first(maximum_degree_of_parallelism_aggregate, lambda node: {node: 1})(tree),
                                  depth_first(degree_of_choice_aggregate, lambda node: {node: 1})(tree),
                                  depth_first(loopable_aggregate, lambda node: {node: False})(tree),
                                  depth_first(strictly_loopable_aggregate, lambda node: {node: False})(tree))).items()}


# ### Optionality

# In[6]:


def search_or(node, o_list, act_list, open_xor=None):
    if len(node._get_children()) == 0:
        i = 0
        for e in o_list:
            if e[0] and node._get_label():
                e.append(node._get_label())
                if node.parent == open_xor:
                    e[0] = False
                elif str(node.parent._get_operator()) != "X":
                    e[0] = False
            if act_list[i][0]:
                act_list[i].append(node._get_label())
            i += 1
        return o_list, act_list

    check = True
    op = node._get_operator()
    if str(op) == "X":
        for a in act_list:
            if a[0]:  # se trovo un or già aperto quelli sotto non li prendo
                check = False
                break
    else:
        check = False

    if check:
        c = len(o_list)
        o_list.append([False])
        act_list.append([True])
        open_xor = node
        for child in node.children:
            o_list[c][0] = True
            search_or(child, o_list, act_list, open_xor)
        act_list[c][0] = False
    else:
        for child in node.children:
            search_or(child, o_list, act_list, )

    return o_list, act_list


# In[7]:


def frequency(aligned_traces, a_dict, tau_dict, o_list, act_list):
    start_or = []  # tutte le label di attività di inizio ramo opzionale
    tot_label = []  # tutte le label di attività che si trovano in un ramo opzionale
    act_plus = []  # per attività non presenti nel modello
    for i in range(0, len(o_list)):
        start_or += o_list[i]
        tot_label += act_list[i]
        act_list[i] = [o_list[i], act_list[i]]  # così riesco a scorrere i dati di un or nella stessa lista

    # struttura di a_dict[a] => [0, True]

    for a_trace in aligned_traces:
        for move in a_trace['alignment']:
            if move[0][1] != '>>' and (move[1][0] != '>>' or move[1][1] is None):
                # print(move[1][0])
                # move sincrona
                t_label = move[1][1]
                if not t_label:
                    t_name = move[0][1]
                    t_label = tau_dict[t_name]
                if not a_dict[t_label][1]:
                    continue
                a_dict[t_label][0] += 1
                if t_label in tot_label:
                    a_dict[t_label][1] = False
                if t_label not in start_or:
                    continue
                for i in range(0, len(act_list)):  # se arriva qui significa che è all'inizio di un ramo or
                    if t_label not in act_list[i][0]:
                        continue
                    for elem in act_list[i][1]:
                        a_dict[elem][1] = True
                    break
            elif move[0][1] == ">>":
                # move on log
                t_label = move[1][0]
                if t_label not in a_dict and t_label not in act_plus:
                    act_plus.append(t_label)
            else:
                # move on model
                t_label = move[1][1]
                if t_label not in start_or:
                    continue
                for i in range(0, len(act_list)):  # se arriva qui significa che è all'inizio di un ramo or
                    if t_label not in act_list[i][0]:
                        continue
                    for elem in act_list[i][1]:
                        a_dict[elem][1] = True
                    break
                continue

    return a_dict, act_plus


# In[8]:


def optionality(tree, net, out, id_c, aligned_traces):
    # or_list = lista identificativa degli or, contengono solo le prime attività di ogni ramo
    # act_list = ogni lista contiene tutte le attività che fanno parte di quell'or
    or_list, act_list = search_or(tree, [[0]], [[0]])
    or_list.pop(0)
    act_list.pop(0)

    t_dict = {}
    t_dict_tau = {}
    for t in net.transitions:
        if "-tau-" in t.label or "Inv" in t.label:
            t_dict_tau[t.name] = t.label
        t_dict[t.label] = [0, True]

    for i in range(0, len(or_list)):
        or_list[i].pop(0)
        act_list[i].pop(0)
    a_list = act_list.copy()
    t_dict, act_plus = frequency(aligned_traces, t_dict, t_dict_tau, or_list, a_list)

    for i in range(0, len(or_list)):
        or_list[i].insert(0, 0)

    for t in t_dict:  # sommo tutte le frequenze delle prime attività dei rami
        # t_label = t_dict[t][1]
        for e_or in or_list:
            if t in e_or:
                e_or[0] += t_dict[t][0]
                break

    # per ogni transazione del modello che si trova in out calcolo la metrica
    opz_dict = {}
    for t in net.transitions:
        choice = out[t.label][id_c]
        if choice == 1:
            opz = (1, 1)
        else:  # se la choice > 1 allora calcolo la metrica
            f_curr = t_dict[t.label][0]  # f(dataset)
            f_or = 0
            for i in range(0, len(act_list)):
                if t.label in act_list[i]:
                    f_or = or_list[i][0]
                    break
            if f_or:
                opz = (round(1 / choice, 4), round(f_curr / f_or, 4))
            else:
                opz = (round(1 / choice, 4), 'not in log')

        opz_dict[t.label] = opz

    return opz_dict, act_plus


# ## Parallelism

# In[9]:


def open_close(node):
    p = node.parent
    i = 0
    for n in p.children:  # cerco la posizione del nodo
        if n == node:
            break
        i += 1
    o = p.children[i - 1]._get_label()
    c = p.children[i + 1]._get_label()
    return o, c


# In[10]:


def search_parallelism(node, p_list):
    if len(node._get_children()) == 0:
        for paral in p_list:
            if paral[0] > 0 and node._get_label():  # aggiungo nei parallelismi "aperti"
                c = p_list.index(paral)
                p_list[c].append(node._get_label())
        return p_list

    op = node._get_operator()
    # se è un and cerco il parallelismo, in qualsiasi altra situazione continuo a scorrere
    if str(op) == "+":
        o, c = open_close(node)
        p_list.append(
            [len(node.children), o, c])  # aggiungo una nuova lista e indico quanti rami deve scorrere per chiuderlo
        i = len(p_list) - 1
        for child in node.children:
            if p_list[i][0] > 0:
                search_parallelism(child, p_list)
                p_list[i][0] -= 1
    else:
        for child in node.children:
            search_parallelism(child, p_list)
    return p_list


# In[11]:


def para_model(net, sp_list, o_c_list):
    G = create_graph(net)
    paral_list = []
    for i in range(0, len(sp_list)):
        paral_list.append([o_c_list[i][0], o_c_list[i][1]])

    for t in net.transitions:
        for i in range(0, len(sp_list)):
            if t.label == paral_list[i][0]:
                paral_list[i].pop(0)
                paral_list[i].insert(0, t.name)
                break
            elif t.label == paral_list[i][1]:
                paral_list[i].pop(1)
                paral_list[i].insert(1, t)  # aggiungo la chiusura come trans da calcolare
                break
            elif t.label in sp_list[i]:
                paral_list[i].append(t)
                break

    paral_activity = {}
    for p in paral_list:
        start = p.pop(0)
        act_dist = long_path(G, start, p)
        longest = max(act_dist.values())  # può essere della chiusura o no

        for activity in p:
            a = activity.label
            if 'Inv' in a or '-tau-' in a:
                continue
            position = act_dist[a]
            paral_activity[a] = round(position / longest, 4)

    return paral_activity


# In[13]:


def para_log(aligned_traces, sp_list, p_activity, pt_dict):
    for a_trace in aligned_traces:
        # per ogni traccia aggiungo tutti i parallelismi trovati
        paral_event_list = []
        paral_list = []
        before_event = False
        for move in a_trace['alignment']:
            if move[0][1] == '>>' or move[1][1] == None:
                continue

            t_name = move[1][1]
            if t_name in p_activity:
                for x in sp_list:  # cerco il parallelismo al quale appartiene
                    if t_name in x:
                        break
                # if paral:  # se è in un parallelismo è un set valido
                before_event = True
                if x not in paral_list:  # se quel parallelismo non era già aperto
                    paral_list.append(x)
                    set_event = [t_name]
                else:
                    set_event.append(t_name)
            else:
                if before_event:  # non è un'evento in un parallelismo, devo vedere se ne chiude uno
                    paral_event_list.append(set_event)
                    l = len(set_event)
                    i = 1
                    for event in set_event:
                        pt = i / l  # calcolo pt (posizione/numero elem del parall)
                        pt_dict[event].append(pt)
                        i += 1
                    set_event.clear()
                before_event = False

    return pt_dict


# In[14]:


def parallelism(tree, net, out, id_par, aligned_traces):
    p_list = search_parallelism(tree, [[0]])
    p_list.pop(0)

    sp_list = []  # nuova lista di set
    p_activity = set()  # set di attività che sono parte di un parallelismo
    pt_dict = {}  # dizionario per salvare i pt
    for x in out.keys():
        if x[0] != 'n':
            pt_dict[x] = []

    # elimino i tau e trasformo le liste in set
    open_close_list = []
    for x in p_list:
        x.pop(0)
        t_open = x.pop(0)
        t_close = x.pop(0)
        open_close_list.append((t_open, t_close))
        y = set()
        for e in x:
            if 'tau' in e:
                continue
            else:
                y.add(e)
                p_activity.add(e)
        sp_list.append(y)

    superset = set()
    for i in range(0, len(sp_list)):
        for j in range(0, len(sp_list)):
            if sp_list[j].issuperset(sp_list[i]) and sp_list[j] != sp_list[i]:
                superset.add(i)

    superset = tuple(sorted(superset, reverse=True))

    for x in superset:
        open_close_list.remove(open_close_list[x])
        sp_list.remove(sp_list[x])

    pt_dict = para_log(aligned_traces, sp_list, p_activity, pt_dict)

    paral_log = {}
    for elem in pt_dict:
        if len(pt_dict[elem]):
            pt_avg = round(sum(pt_dict[elem]) / len(pt_dict[elem]), 4)
            pt_var = round(statistics.variance(pt_dict[elem]), 4)
        else:
            pt_avg = 0
            pt_var = 0
        paral_log[elem] = (round(1 / out[elem][id_par], 4), pt_avg, pt_var)

    paral_mod = para_model(net, sp_list, open_close_list)
    return paral_mod, paral_log


# ## Distance

# In[15]:


## NOTA: utilizzare la rete originale
def create_graph(net):
    G = nx.MultiDiGraph()
    for t in net.transitions:
        G.add_node(t.name, label=t.label, type="t")
    for p in net.places:
        G.add_node(p.name, label=p.name, type="p")
    for a in net.arcs:
        G.add_edge(a.source.name, a.target.name)
    return G


# In[16]:


def long_path(G, s, t_list):
    # va bene se s e t sono dei places, poi non vengono conteggiati nel for
    # t_list = t_list[::-1]
    act_list = []
    for t in t_list:
        act_list.append(t.label)

    longest = {}
    for t in t_list:
        if t.label in longest.keys():
            continue
        path = nx.all_simple_paths(G, source=s, target=t.name)

        for n in path:
            count = 0
            for node in n:
                lab = G.nodes[node]['label']
                if G.nodes[node]['type'] == 't':
                    if not lab or '-tau-' in lab or 'Inv' in lab:
                        continue
                    count += 1
                else:
                    continue
                if lab not in longest.keys() and lab in act_list:
                    longest[lab] = count
                elif lab in longest.keys() and longest[lab] < count:
                    longest[lab] = count
                else:
                    continue
    return longest


# In[17]:


def dist_log(aligned_traces, pt_dict):
    for a_trace in aligned_traces:
        trace = []
        for move in a_trace['alignment']:
            if move[0][1] == '>>' or move[1][0] == '>>':
                continue
            t_label = move[1][1]
            trace.append(t_label)
        l = len(trace)
        i = 1
        for t in trace:
            pt = i / l  # controllare che l non sia < i nell'ultimo elem
            pt_dict[t].append(pt)
            i += 1

    return pt_dict


# In[18]:


def avg_var_pt(out, aligned_traces, sp_list=None, p_activity=None):
    pt_dict = {}  # dizionario per salvare i pt
    for x in out.keys():
        if 'Inv' not in x and '-tau-' not in x:
            pt_dict[x] = []

    if sp_list is None:
        pt_dict = dist_log(aligned_traces, pt_dict)
    else:
        pt_dict = para_log(aligned_traces, sp_list, p_activity, pt_dict)

    m_log = {}
    for elem in pt_dict:
        if len(pt_dict[elem]) > 1:
            pt_avg = round(sum(pt_dict[elem]) / len(pt_dict[elem]), 4)
            pt_var = round(statistics.variance(pt_dict[elem]), 4)
        elif len(pt_dict[elem]) == 1:
            pt_avg = pt_dict[elem][0]
            pt_var = pt_dict[elem][0]
        else:
            pt_avg = 0
            pt_var = 0
        m_log[elem] = [pt_avg, pt_var]

    return m_log


# In[19]:


def distance(out, net, im, aligned_traces):
    G = create_graph(net)
    s = im.popitem()[0].name  # place's name of initial marking
    # e = fm.popitem()[0].name  # place's name of final marking

    print("Inizio calcolo della distanza... ")
    longest = long_path(G, s, net.transitions)
    longest_path = max(longest.values())
    dist_dict = avg_var_pt(out, aligned_traces)

    for elem in dist_dict:
        position = longest[elem]
        dist_dict[elem].insert(0, round(position / longest_path, 4))

    return dist_dict


# ## Main

# In[22]:

folder_address = "../../../../Datasets/IKNL/dataset_K21.242/All_Episodes/"
LPMs_file = glob.glob(folder_address + "LPMs_all/*.pnml")

Unique_acts = set()
behaviour_per_acts = dict()
Overlapped_LPMs = []
direct_loop_error = 0
sequence_error = 0

for file in LPMs_file:
    Is_overlapped = 0
    print(file)
    net, initial_marking, final_marking = pnml_import.apply(file)
    try:
        tree = wf_net_converter.apply(net, initial_marking, final_marking)
        tree_2 = generic.fold(tree)
        op = tree_2._get_operator()
        LPMs_characters = feature_map(tree_2)
        for item in LPMs_characters:
            if item.label is not None:
                if item.label in Unique_acts:
                    Is_overlapped = 1
                Unique_acts.add(item.label)
                if item.label not in behaviour_per_acts.keys():
                    behaviour_per_acts[item.label] = {'parallel': LPMs_characters[item][0] - 1,
                                                      'choice': LPMs_characters[item][1] - 1,
                                                      'direct loop': LPMs_characters[item][2],
                                                      'indirect loop': LPMs_characters[item][3]}
                    if LPMs_characters[item][0] == 1:
                        behaviour_per_acts[item.label]['sequence'] = 1
                    else:
                        behaviour_per_acts[item.label]['sequence'] = 0
                else:
                    behaviour_per_acts[item.label]['parallel'] += LPMs_characters[item][0] - 1
                    behaviour_per_acts[item.label]['choice'] += LPMs_characters[item][1] - 1
                    behaviour_per_acts[item.label]['direct loop'] += LPMs_characters[item][2]
                    behaviour_per_acts[item.label]['indirect loop'] += LPMs_characters[item][3]
                    if LPMs_characters[item][0] == 1:
                        behaviour_per_acts[item.label]['sequence'] += 1

        Overlapped_LPMs.append(Is_overlapped)
        print(LPMs_characters)
    except:
        transitions = net.transitions
        for tr in transitions:
            if tr.label is not None:
                Unique_acts.add(tr.label)
                if tr.label not in behaviour_per_acts.keys():
                    behaviour_per_acts[tr.label] = {'parallel': 0,
                                                    'choice': 0,
                                                    'direct loop': 1,
                                                    'indirect loop': 0,
                                                    'sequence': 1}
                else:
                    behaviour_per_acts[tr.label]['direct loop'] += 1
                    behaviour_per_acts[tr.label]['sequence'] += 1

        # gviz = pn_visualizer.apply(net, initial_marking, final_marking)
        # pn_visualizer.view(gviz)


parallel = 0
choice = 0
sequence = 0
Dloop = 0
iDloop = 0

for item in behaviour_per_acts:
    parallel += behaviour_per_acts[item]['parallel']
    choice += behaviour_per_acts[item]['choice']
    sequence += behaviour_per_acts[item]['sequence']
    Dloop += behaviour_per_acts[item]['direct loop']
    iDloop += behaviour_per_acts[item]['indirect loop']

parallel = parallel / len(Unique_acts)
choice = choice / len(Unique_acts)
sequence += sequence_error
sequence = sequence / len(Unique_acts)
Dloop += direct_loop_error
Dloop = Dloop / len(Unique_acts)
iDloop = iDloop / len(Unique_acts)

print('sequence behavior: ', sequence / (parallel + choice + sequence + Dloop + iDloop))
print('choice behavior: ', choice / (parallel + choice + sequence + Dloop + iDloop))
print('parallel behavior: ', parallel / (parallel + choice + sequence + Dloop + iDloop))
print('Direct loop behavior: ', Dloop / (parallel + choice + sequence + Dloop + iDloop))
print('indirect loop behavior: ', iDloop / (parallel + choice + sequence + Dloop + iDloop))
print('Overlapped LPMs:', np.mean(Overlapped_LPMs))
print('done')
print(len(Unique_acts))

# gviz = pn_visualizer.apply(net, initial_marking, final_marking)
# pn_visualizer.view(gviz)