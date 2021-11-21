# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from os import error
from django import template
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse, HttpResponseRedirect
from django.template import loader
from django.urls import reverse
import pandas as pd
from django.contrib.auth.models import User

from .forms import DataForm
from .models import Data

from .main import cal_pop_fitness, select_mating_pool, crossover, mutation
@login_required(login_url="/login/")
def index(request):
    
    context = {
        'segment': 'index',
    }
    html_template = loader.get_template('index.html')
    return HttpResponse(html_template.render(context, request))

@login_required(login_url="/login/")
def profile(request):
    data = request.user.datas
    print(data)
    initial_dict = {
        'user' : request.user.id,
        "nuissance" : data.nuissance,
        "bruit" : data.bruit,
        "impacts" : data.impacts,
        "geographique" : data.geographique,
        "equipement" : data.equipement,
        "accessibilite" : data.accessibilite,
        "climat" : data.climat
    }
    print(data)
    if request.method == 'POST':
        instance = Data.objects.get(user = request.user)
        form = DataForm(request.POST or None, instance=instance)
        if form.is_valid():
            form_object = form.save(commit=False)
            
            form_object.save()


    else:
        form = DataForm(initial = initial_dict)
    context = {
        'segment': 'profile',
        'form' : form
    }
    html_template = loader.get_template('profile.html')
    return HttpResponse(html_template.render(context, request))

import subprocess

@login_required(login_url="/login/")
def promethee(request):
    # Inputs of the equationf or the first decision maker (ECONOMISTE).
    ECONOMISTE = [0.178, 0.294, 0.0616, 0.0616, 0.0616, 0.1738, 0.1738]
    POLITICIEN = [0.0751, 0.1363, 0.1363, 0.0616, 0.172, 0.172, 0.172]
    Rep_de_lenv = [0.0496, 0.0708, 0.1731, 0.1893, 0.1893, 0.1752, 0.1527]
    Rep_de_publ = [0.1957, 0.1379, 0.1379, 0.1379, 0.1379, 0.1645, 0.1645]
    decideurs = User.objects.filter(is_staff = False)
    count = Data.objects.filter(user__in=decideurs, nuissance__isnull=False).count()
    final_sorts = []
    if request.method == 'POST':
        print('HIIIIIIIIIIIII')
        admin_profile = User.objects.get(username = request.user)
        for decideur in decideurs:
            # subprocess.Popen('python ',user_profile.datasDecid.last().mp , user_profile.datasDecid.last().weights)
            import sys
            import numpy
            import csv
            import numpy as np
            import time
            import os
            from pathlib import Path
            BASE_DIR = Path(__file__).resolve().parent.parent
            print(BASE_DIR)
            np.set_printoptions(precision=3)
            print("PROMETHEE 2 METHOD")

            print("##################################################")

            workpath = os.path.dirname(os.path.abspath(__file__))
            print('workplace', workpath)
            with open('media/Matrice_de_performance/etude_suisse.csv') as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                Matrix = np.array(list(csv_reader))

            print('Matrice de performance',Matrix)

            print("STEP 1 : Normalize the Evaluation Matrix")
            array_Matrix  = np.array(Matrix)

            Alternative_matix = array_Matrix[2:,1:].astype(np.single)
            print('Alternative_matix \n',Alternative_matix)

            labels = array_Matrix[0,1:]
            print('labels \n',labels)

            Alternatives = array_Matrix[2:,0]
            print('Names \n',Alternatives)

            # TODO! Add beneficial and non benificial criterias
            maximisation = array_Matrix[1,1:]
            print('Beneficial or Not  \n',maximisation)

            # Get min and max for each criteria
            min_criteria_array = Alternative_matix.min(axis=0)
            print('min_criteria_array \n',min_criteria_array)

            max_criteria_array = Alternative_matix.max(axis=0)
            print('max_criteria_array \n',max_criteria_array)


            for i in range(len(Alternative_matix)):
                for j in range(len(Alternative_matix[i])):
                    if maximisation[j] == 'yes':
                        Alternative_matix[i][j] = (max_criteria_array[j]-Alternative_matix[i][j])/(max_criteria_array[j]-min_criteria_array[j])
                    else:
                        Alternative_matix[i][j] = (Alternative_matix[i][j]-min_criteria_array[j])/(max_criteria_array[j]-min_criteria_array[j])
            print('Alternative_matix \n',Alternative_matix)
            time.sleep(3)

            print("STEP 2 : Calculate Evaluative ieme per the othere {m1-m2 | m1-m3 | ....}")
            # Create the Alternatives Possibilities array[m1-m2,........]
            def all_alternatives(Alternatives):
                Alternative_possibilities = []
                for i in range(len(Alternatives)):
                    for j in range(len(Alternatives)):
                        if i == j:
                            pass
                        else:
                            Alternative_possibilities.append(Alternatives[i]+'-'+Alternatives[j])
                return np.array(Alternative_possibilities).reshape(len(Alternative_possibilities),1)
            Alternative_possibilities = all_alternatives(Alternatives)
            print('Alternative_possibilities \n', Alternative_possibilities)
            time.sleep(3)

            # create the matrix of all variables possibilities:
            def all_variables(matrix):
                new_matrix = []
                for i in range(len(matrix)):
                    for j in range(len(matrix)):
                        if i == j:
                            pass
                        else:
                            new_matrix.append(matrix[i]-matrix[j])
                return np.array(new_matrix)

            variables_possibilities = all_variables(Alternative_matix)
            print('variables_possibilities \n', variables_possibilities)
            time.sleep(3)
            print('Alternative_possibilities shape \n', Alternative_possibilities.shape)
            print('variables_possibilities shape \n', variables_possibilities.shape)

            # concatenate the Names and variables related 
            the_all_matrix = np.hstack([Alternative_possibilities, variables_possibilities])
            print('The All Matrix \n', the_all_matrix)
            time.sleep(3)
            print("STEP 3 : Calculate the PREFERENCE Function")
            # Create an updated matrix that return 0 if value is negative or equal to 0 
            # else keep value as it it
            def changetozeros(matrix):
                for i in range(len(matrix)) :  
                    for j in range(len(matrix[i])) :  
                        if matrix[i][j] < 0 :
                            matrix[i][j] = 0
                return matrix

            Preference_matrix = changetozeros(variables_possibilities)
            print('PREFERENCE_matrix \n', Preference_matrix)
            time.sleep(3)
            # concatenate the Names and preferences related 
            the_Preference_matrix = np.hstack([Alternative_possibilities, Preference_matrix])
            print('the_Preference_matrix \n', the_Preference_matrix)

            weights = decideur.datas.get_weights(   )

            print('weights \n', weights)
            array_weights = np.asarray(weights, dtype='float')
            print('array_weights \n', array_weights)
            time.sleep(3)
            # lets create a fucntion to mult the weights with the matrix of preferences variables
            def mult_matrix_vect(matrix, weight):
                for i in range(len(matrix)) :  
                    for j in range(len(matrix[i])) :  
                        matrix[i][j] = matrix[i][j]* weight[j]
                return matrix
            # TODO: Check this multyplie function
            def show_mult_matrix_vect(matrix, weight):
                data = []
                for i in range(len(matrix)) :  
                
                    for j in range(len(matrix[i])) : 
                    
                        data.append('{}*{}'.format(weight[j],matrix[i][j]))
                return np.array(data)

            Agregate_preference_matrix = mult_matrix_vect(Preference_matrix, array_weights)
            show_calculation = show_mult_matrix_vect(Preference_matrix, array_weights)

            print('show_calculation \n', show_calculation)
            print('Agregate_preference_matrix \n', Agregate_preference_matrix)
            time.sleep(3)
            # lets add a column to sum these aggregated preferences
            def add_aggregated_preferences_line(matrix):
                average_line_weight = []
                
                for i in range(len(matrix)) :
                    sum = 0  
                    for j in range(len(matrix[i])) :
                        sum = sum + matrix[i][j] 
                    average_line_weight.append(sum)
                    
                matrix = np.vstack([matrix.transpose(), average_line_weight]).transpose()
                return matrix

            Agregate_preference_matrix_with_sum = add_aggregated_preferences_line(Agregate_preference_matrix)
            print('Agregate_preference_matrix_with_sum \n', Agregate_preference_matrix_with_sum)
            time.sleep(3)
            aggrsums = Agregate_preference_matrix_with_sum[:,-1]
            print(aggrsums)
            # take only the aggragated sum values(LAST column) and create aggregated preference Function(matrix)
            def create_aggregated_matrix(matrix, aggr):
                # retrieve only the aggregated column(list)
                aggregate_column = np.array(matrix[:, -1].transpose())
                agrs = aggr.tolist()
                print(aggregate_column)
                print("type of aggregate_column")
                print(type(aggregate_column))
            #  aggregated_matrix  = [[len(Alternatives), len(Alternatives) ]]
                #hada el hmar ghadi ylez madam les valeurs yethattou
            # print(np.array(aggregated_matrix).shape)
                for i in range(len(aggregated_matrix)) :  
                    for j in range(len(aggregated_matrix[i])) :       
                        if i == j:
                            aggregated_matrix[i][j] = 0        
                        else:  
                            aggregated_matrix[i][j]= agrs[0]
                            agrs.pop(0) 
                        
                            
                            # aggregated_matrix.append(aggregate_column[j])
                # print('lol',aggregated_matrix)
                print(np.array(aggregated_matrix).shape)
                return aggregated_matrix
                
            aggregated_matrix = np.zeros((len(Alternatives), len(Alternatives)))

            print("len alternatives")
            created_aggregated_matrix = create_aggregated_matrix(aggregated_matrix, aggrsums)

            print("HADA created_aggregated_matrix")
            print(created_aggregated_matrix)
            time.sleep(3)
            duplicated = created_aggregated_matrix
            #flot entrant w sortant
            def sumColumn(matrice):
                return [sum(col) for col in zip(*matrice)] 

            sommeeecolonne= sumColumn(created_aggregated_matrix)

            sumrows = np.sum(created_aggregated_matrix, axis = 1)
            #we need to deivde those calculated values on the number of alternatives -1
            newsommecolonne = []
            newsumrow= []
            for x in sommeeecolonne:
                newsommecolonne.append(x /(len(created_aggregated_matrix) - 1))

            for x in sumrows:
                newsumrow.append(x /(len(created_aggregated_matrix) - 1))
            
            print("flots entrants \n" , newsommecolonne)
            print("flots sortants \n" , newsumrow)

            created_aggregated_matrix = np.vstack([created_aggregated_matrix, newsumrow])
            print("updated matrix with columns ")
            print(created_aggregated_matrix)

            newsommecolonne.append(0)
            created_aggregated_matrix= np.vstack([created_aggregated_matrix.transpose(), newsommecolonne]).transpose()
            print("created_aggregated_matrix kamel\n", created_aggregated_matrix)


            #here i'll be using a function to calculate the flots 
            def calculateflows(matrix):
                diffs=[]
                for i in range(len(matrix)):
                    diffs.append(matrix[i,-1] - matrix[-1, i])
                return diffs

            print("flowscreated_aggregated_matrix")
            differencesflots = calculateflows(created_aggregated_matrix)
            print(differencesflots)


            alt = np.append(Alternatives, " ")
            duplicated = np.vstack([alt, created_aggregated_matrix.transpose()])

            talyabachtetsetef  = np.vstack([duplicated, differencesflots]).transpose()
            print("sma3")

            print("##############")
            with numpy.printoptions(threshold=numpy.inf):
                print(talyabachtetsetef[:-1,:])

            # Sort 2D numpy array by first column
            sortedArr = talyabachtetsetef[talyabachtetsetef[:,-1].argsort()]
            print('Sorted 2D Numpy Array')
            print("##############")
            with numpy.printoptions(threshold=numpy.inf):
                print(np.flipud(sortedArr))
            print("Final Sort is : ")
            print(sortedArr[:,0])
            final_sorts.append([decideur,sortedArr[:,0]])
    context = {
        'segment': 'promethee',
        'decideurs' : decideurs,
        'count' : count,
        'final_sort' : final_sorts
    }
    html_template = loader.get_template('promethee.html')
    return HttpResponse(html_template.render(context, request))


import numpy as np
import pandas as pd
from apyori import apriori

@login_required(login_url="/login/")
def apriori(request):
    context = {'segment': 'apriori'}
    import sys
    from itertools import chain, combinations
    from collections import defaultdict
    from optparse import OptionParser

    def subsets(arr):
        """ Returns non empty subsets of arr"""
        return chain(*[combinations(arr, i + 1) for i, a in enumerate(arr)])

    def returnItemsWithMinSupport(itemSet, transactionList, minSupport, freqSet):
        """calculates the support for items in the itemSet and returns a subset
        of the itemSet each of whose elements satisfies the minimum support"""
        _itemSet = set()
        localSet = defaultdict(int)

        for item in itemSet:
            for transaction in transactionList:
                if item.issubset(transaction):
                    freqSet[item] += 1
                    localSet[item] += 1

        for item, count in localSet.items():
            support = float(count) / len(transactionList)

            if support >= minSupport:
                _itemSet.add(item)

        return _itemSet


    def joinSet(itemSet, length):
        """Join a set with itself and returns the n-element itemsets"""
        return set(
            [i.union(j) for i in itemSet for j in itemSet if len(i.union(j)) == length]
        )

    def getItemSetTransactionList(data_iterator):
        transactionList = list()
        itemSet = set()
        for record in data_iterator:
            transaction = frozenset(record)
            transactionList.append(transaction)
            for item in transaction:
                itemSet.add(frozenset([item]))  # Generate 1-itemSets
        return itemSet, transactionList

    def runApriori(data_iter, minSupport, minConfidence):
        """
        run the apriori algorithm. data_iter is a record iterator
        Return both:
        - items (tuple, support)
        - rules ((pretuple, posttuple), confidence)
        """
        itemSet, transactionList = getItemSetTransactionList(data_iter)

        freqSet = defaultdict(int)
        largeSet = dict()
        # Global dictionary which stores (key=n-itemSets,value=support)
        # which satisfy minSupport

        assocRules = dict()
        # Dictionary which stores Association Rules

        oneCSet = returnItemsWithMinSupport(itemSet, transactionList, minSupport, freqSet)

        currentLSet = oneCSet
        k = 2
        while currentLSet != set([]):
            largeSet[k - 1] = currentLSet
            currentLSet = joinSet(currentLSet, k)
            currentCSet = returnItemsWithMinSupport(
                currentLSet, transactionList, minSupport, freqSet
            )
            currentLSet = currentCSet
            k = k + 1

        def getSupport(item):
            """local function which Returns the support of an item"""
            return float(freqSet[item]) / len(transactionList)

        toRetItems = []
        for key, value in largeSet.items():
            toRetItems.extend([(tuple(item), getSupport(item)) for item in value])

        toRetRules = []
        for key, value in list(largeSet.items())[1:]:
            for item in value:
                _subsets = map(frozenset, [x for x in subsets(item)])
                for element in _subsets:
                    remain = item.difference(element)
                    if len(remain) > 0:
                        confidence = getSupport(item) / getSupport(element)
                        if confidence >= minConfidence:
                            toRetRules.append(((tuple(element), tuple(remain)), confidence))
        return toRetItems, toRetRules


    def printResults(items, rules):
        """prints the generated itemsets sorted by support and the confidence rules sorted by confidence"""
        for item, support in sorted(items, key=lambda x: x[1]):
            print("item: %s , %.3f" % (str(item), support))
        print("\n------------------------ RULES:")
        for rule, confidence in sorted(rules, key=lambda x: x[1]):
            pre, post = rule
            print("Rule: %s ==> %s , %.3f" % (str(pre), str(post), confidence))


    def to_str_results(items, rules):
        """prints the generated itemsets sorted by support and the confidence rules sorted by confidence"""
        i, r = [], []
        for item, support in sorted(items, key=lambda x: x[1]):
            x = "item: %s , %.3f" % (str(item), support)
            i.append(x)

        for rule, confidence in sorted(rules, key=lambda x: x[1]):
            pre, post = rule
            x = "Rule: %s ==> %s , %.3f" % (str(pre), str(post), confidence)
            r.append(x)

        return i, r


    def dataFromFile(fname):
        """Function which reads from the file and yields a generator"""
        with open(fname, "rU") as file_iter:
            for line in file_iter:
                line = line.strip().rstrip(",")  # Remove trailing comma
                record = frozenset(line.split(","))
                yield record

    if request.method == 'POST':
        print('HIIIIIIIIIIIII')
        optparser = OptionParser()
        optparser.add_option(
            "-f", 
            "--inputFile", 
            dest="input", 
            help="filename containing csv", 
            default=None
        )
        optparser.add_option(
            "-s",
            "--minSupport",
            dest="minS",
            help="minimum support value",
            default=0.15,
            type="float",
        )
        optparser.add_option(
            "-c",
            "--minConfidence",
            dest="minC",
            help="minimum confidence value",
            default=0.6,
            type="float",
        )

        (options, args) = optparser.parse_args()

        inFile = None
        if options.input is None:

            inFile = sys.stdin
        elif options.input is not None:
            inFile = dataFromFile(options.input)
        else:
            print("No dataset filename specified, system with exit\n")
            sys.exit("System will exit")

        minSupport = options.minS
        minConfidence = options.minC

        items, rules = runApriori('media/Matrice_de_performance/etude_suisse.csv', minSupport, minConfidence)
        print(rules)
        print('HI')
        printResults(items, rules)


    html_template = loader.get_template('apriori.html')
    return HttpResponse(html_template.render(context, request))

import numpy
@login_required(login_url="/login/")
def ag(request):
    ECONOMISTE = [0.178, 0.294, 0.0616, 0.0616, 0.0616, 0.1738, 0.1738]
    POLITICIEN = [0.0751, 0.1363, 0.1363, 0.0616, 0.172, 0.172, 0.172]
    rep_de_lenv = [0.0496, 0.0708, 0.1731, 0.1893, 0.1893, 0.1752, 0.1527]
    rep_de_publ = [0.1957, 0.1379, 0.1379, 0.1379, 0.1379, 0.1645, 0.1645]
    html_template = loader.get_template('ag.html')
    result = None

    decideurs = User.objects.filter(is_staff = False)
    result_for_each_decideur = []
    print(decideurs)
    decideur_datas = Data.objects.filter(
        user__in=decideurs, 
        nuissance__isnull=False,
        impacts__isnull=False,)
    count = decideur_datas.count()
    print(count)
    if request.method == 'POST':
        print('HIIIIIIIIIIIII')
        
        for decideur_data in decideur_datas:

            choice = decideur_data.get_weights()
            # Number of the weights we are looking to optimize.
            num_weights = 7

            """
            Genetic algorithm parameters:
                Mating pool size
                Population sizex    
            """

            import pandas as pd

            data = pd.read_csv('apps/app/etude_suisse.csv')
            nuissance = data.iloc[:, 1]
            bruit = data.iloc[:, 2]
            impact = data.iloc[:, 3]
            geotechnique = data.iloc[:, 4]
            equipement = data.iloc[:, 5]
            accessibility = data.iloc[:, 6]
            climat = data.iloc[:, 7]




            population_data = data.iloc[:, 1:]
            sol_per_pop = len(population_data)
            num_parents_mating = 4

            # Defining the population size.
            pop_size = (sol_per_pop,num_weights) # The population will have sol_per_pop chromosome where each chromosome has num_weights genes.

            data2 = pd.read_csv('apps/app/etude_suisse2.csv', header=None, index_col=0)

            #Creating the initial population.
            new_population = numpy.array(data2)
            print(new_population)

            num_generations = 5
            for generation in range(num_generations):
                print("Generation : ", generation+1)
                # Measing the fitness of each chromosome in the population.
                fitness =  cal_pop_fitness(choice, new_population)

                # Selecting the best parents in the population for mating.
                parents =  select_mating_pool(new_population, fitness, 
                                                num_parents_mating)

                # Generating next generation using crossover.
                offspring_crossover =  crossover(parents,
                                                offspring_size=(pop_size[0]-parents.shape[0], num_weights))

                # Adding some variations to the offsrping using mutation.
                offspring_mutation =  mutation(offspring_crossover)

                # Creating the new population based on the parents and offspring.
                new_population[0:parents.shape[0], :] = parents
                new_population[parents.shape[0]:, :] = offspring_mutation

                # The best result in the current iteration.
                print("Best result : ", numpy.min(numpy.sum(new_population* choice, axis=1)))

            # Getting the best solution after iterating finishing all generations.
            #At first, the fitness is calculated for each solution in the final generation.
            fitness =  cal_pop_fitness( choice, new_population)
            # Then return the index of that solution corresponding to the best fitness.
            best_match_idx = numpy.where(fitness == numpy.min(fitness))

            result = []
            for sublist in new_population[best_match_idx, :]:
                for item in sublist:
                    for it in item:
                        print(it)
                        result.append(it)
            
            print("Best solution : ", result)
            condi1 = data.iloc[:, 1] == result[0]
            condi2 = data.iloc[:, 2] == result[1]
            condi3 = data.iloc[:, 3] == result[2]
            condi4 = data.iloc[:, 4] == result[3]
            condi5 = data.iloc[:, 5] == result[4]
            condi6 = data.iloc[:, 6] == result[5]
            condi7 = data.iloc[:, 7] == result[6]
            

            id_zone = data.loc[(condi1)&(condi2)&(condi3)&(condi4)]
            # TODO update the if condition for each decideur
            print('id_zone: ', id_zone.iloc[:, 0])
            result_for_each_decideur.append((decideur_data.user, id_zone.iloc[:, 0]))

        context = {'segment': 'ag',
                    'solution' : result,
                    'id_zone' : id_zone.iloc[:, 0],
                    'count' : count,
                    'final_result' : result_for_each_decideur,
                    'decideurs': decideurs
                }
        print("Best solution fitness : ", fitness[best_match_idx])
        return HttpResponse(html_template.render(context, request))
    context = {'segment': 'ag',
    'count' : count,
    'decideurs': decideurs}
    return HttpResponse(html_template.render(context, request))

@login_required(login_url="/login/")
def pages(request):
    context = {}
    # All resource paths end in .html.
    # Pick out the html file name from the url. And load that template.
    try:

        load_template = request.path.split('/')[-1]

        if load_template == 'admin':
            return HttpResponseRedirect(reverse('admin:index'))
        context['segment'] = load_template

        html_template = loader.get_template(load_template)
        return HttpResponse(html_template.render(context, request))

    except template.TemplateDoesNotExist:

        html_template = loader.get_template('page-404.html')
        return HttpResponse(html_template.render(context, request))

    except:
        html_template = loader.get_template('page-500.html')
        return HttpResponse(html_template.render(context, request))
