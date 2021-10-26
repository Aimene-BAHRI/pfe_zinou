# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from django import template
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse, HttpResponseRedirect
from django.template import loader
from django.urls import reverse
import pandas as pd

from .main import cal_pop_fitness, select_mating_pool, crossover, mutation
@login_required(login_url="/login/")
def index(request):
    context = {'segment': 'index'}

    html_template = loader.get_template('index.html')
    return HttpResponse(html_template.render(context, request))

@login_required(login_url="/login/")
def promethee(request):

    if request.method == 'POST':
        print('HIIIIIIIIIIIII')
        file = request.FILES['docfile'] or None
         # Inputs of the equationf or the first decision maker (ECONOMISTE).
        ECONOMISTE = [0.178, 0.294, 0.0616, 0.0616, 0.0616, 0.1738, 0.1738]
        POLITICIEN = [0.0751, 0.1363, 0.1363, 0.0616, 0.172, 0.172, 0.172]
        Rep_de_lenv = [0.0496, 0.0708, 0.1731, 0.1893, 0.1893, 0.1752, 0.1527]
        Rep_de_publ = [0.1957, 0.1379, 0.1379, 0.1379, 0.1379, 0.1645, 0.1645]

        data = pd.read_csv(file, header=None, index_col=0).iloc[1:,:]
        print(data)
    context = {'segment': 'promethee'}
    html_template = loader.get_template('promethee.html')
    return HttpResponse(html_template.render(context, request))


import numpy as np
import pandas as pd
from apyori import apriori

@login_required(login_url="/login/")
def apriori(request):
    context = {'segment': 'apriori'}
    if request.method == 'POST':
        print('HIIIIIIIIIIIII')
        file = request.FILES['docfile'] or None
         # Inputs of the equationf or the first decision maker (ECONOMISTE).
        ECONOMISTE = [0.178, 0.294, 0.0616, 0.0616, 0.0616, 0.1738, 0.1738]
        POLITICIEN = [0.0751, 0.1363, 0.1363, 0.0616, 0.172, 0.172, 0.172]
        Rep_de_lenv = [0.0496, 0.0708, 0.1731, 0.1893, 0.1893, 0.1752, 0.1527]
        Rep_de_publ = [0.1957, 0.1379, 0.1379, 0.1379, 0.1379, 0.1645, 0.1645]

        data = pd.read_csv(file, header=None, index_col=0)

        print(data.shape)

        records = []
        for i in range(0, 651):
            records.append([str(data.values[i, j]) for j in range(0, 7)])
        print('test')
        association = list(apriori(records, min_support = 0.5, min_confidence = 0.7, min_lift = 1.2, min_length = 7))

    html_template = loader.get_template('apriori.html')
    return HttpResponse(html_template.render(context, request))

import numpy
@login_required(login_url="/login/")
def ag(request):
    context = {'segment': 'ag'}
    html_template = loader.get_template('ag.html')
    result = None
    if request.method == 'POST':
        print('HIIIIIIIIIIIII')
        file = request.FILES['docfile'] or None
        # Inputs of the equationf or the first decision maker (ECONOMISTE).
        ECONOMISTE = [0.178, 0.294, 0.0616, 0.0616, 0.0616, 0.1738, 0.1738]
        POLITICIEN = [0.0751, 0.1363, 0.1363, 0.0616, 0.172, 0.172, 0.172]
        rep_de_lenv = [0.0496, 0.0708, 0.1731, 0.1893, 0.1893, 0.1752, 0.1527]
        rep_de_publ = [0.1957, 0.1379, 0.1379, 0.1379, 0.1379, 0.1645, 0.1645]


        equation_inputs = [0.178, 0.294, 0.0616, 0.0616, 0.0616, 0.1738, 0.1738]

        # Number of the weights we are looking to optimize.
        num_weights = 7

        """
        Genetic algorithm parameters:
            Mating pool size
            Population size
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
            fitness =  cal_pop_fitness(ECONOMISTE, new_population)

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
            print("Best result : ", numpy.min(numpy.sum(new_population* ECONOMISTE, axis=1)))

        # Getting the best solution after iterating finishing all generations.
        #At first, the fitness is calculated for each solution in the final generation.
        fitness =  cal_pop_fitness( ECONOMISTE, new_population)
        # Then return the index of that solution corresponding to the best fitness.
        best_match_idx = numpy.where(fitness == numpy.min(fitness))

        print("Best solution : ", new_population[best_match_idx, :])
        result = new_population[best_match_idx, :]
        print("Best solution fitness : ", fitness[best_match_idx])
        context['solution'] = result
        return HttpResponse(html_template.render(context, request))

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
