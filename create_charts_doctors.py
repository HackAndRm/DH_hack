# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 23:29:10 2017

@author: mskara
"""
import matplotlib.pyplot as plt
import numpy as np
from pandas import read_csv 
import pymongo

def data_subset_modifier(df_total, query_name, query_response):
    '''
    Takes as input the dataframe, it subsets for the query
    calculates mean of revenues and costs,
    analyze the cost/revenue splits
    give back the estimated values
    '''
    df = df_total[df_total[query_name] == query_response]

    rev_mean     = df['Revenues'].mean()/1000
    GKV_Umsatz   = df['GKV_Umsatz'].sum()   / (df['GKV_Umsatz'].sum() + df['Privatumsatz'].sum())
    Privatumsatz = df['Privatumsatz'].sum() / (df['GKV_Umsatz'].sum() + df['Privatumsatz'].sum())

    cost_mean       = df['Total_costs'].mean()/1000
    sum_costs       = df['Materialaufwand'].sum() + df['Personalkosten'].sum() + df['Raumkosten'].sum() + df['AfA'].sum() + df['Zinsen'].sum() + df['Sonstige_Kosten'].sum()
    Materialaufwand = df['Materialaufwand'].sum() / sum_costs
    Personalkosten  = df['Personalkosten'].sum()  / sum_costs
    AfA             = df['AfA'].sum()             / sum_costs
    Raumkosten      = df['Raumkosten'].sum()      / sum_costs
    Zinsen          = df['Zinsen'].sum()          / sum_costs
    Sonstige_Kosten = df['Sonstige_Kosten'].sum() / sum_costs
    
    return rev_mean, GKV_Umsatz, Privatumsatz, cost_mean, Materialaufwand, Personalkosten, AfA, Raumkosten, Zinsen, Sonstige_Kosten

    
def plot_bar (rev_mean, cost_mean, query_name, query_response):
    '''
    Takes as input the information about mean revenue and cost
    returns a bar-chart with the information
    '''
    objects = ('Einnahmen', 'Kosten')
    y_pos = np.arange(len(objects))
    performance = [rev_mean, cost_mean]
    color  = ['yellowgreen', 'red']

    plt.barh(y_pos, performance, color = color, align = 'center',linewidth = 0.4, alpha = 0.5)
    plt.yticks(y_pos, objects,fontsize = 12)
    plt.xlabel('Tausende Euro',fontsize = 12)
    title = 'Einnahmen und Kosten von '+ query_name +' in '+ query_response
    plt.suptitle(title, fontsize = 12, fontweight = 'bold') 
    plt.show()
    fig = plt.figure()
    return fig
    #return  it should return the image itself

    
def plot_costs(Materialaufwand, Personalkosten, AfA, Raumkosten, Zinsen, Sonstige_Kosten, query_name, query_response):
    '''
    Takes as input the information about costs
    returns a pie-chart with the splitted costs
    '''
    sizes   = [Materialaufwand, Personalkosten, AfA, Raumkosten, Zinsen, Sonstige_Kosten]
    labels  = ['Materialaufwand', 'Personalkosten', 'AfA','Raumkosten','Zinsen','Sonstige_Kosten']
    explode = (0, 0.1, 0, 0, 0, 0 )
    colors  = ['khaki', 'yellowgreen', 'lightcoral', 'bisque', 'powderblue', 'cornflowerblue']
    plt.pie (sizes, explode = explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
    plt.axis('equal')
    title = 'Kostenverteilung von '+ query_name +' in '+ query_response
    plt.suptitle(title, fontsize = 11, fontweight='bold')
    plt.show()
    fig = plt.figure()
    return fig
    #return  it should return the image itself


def plot_revenues(GKV_Umsatz, Privatumsatz, query_name, query_response):
    '''
    Takes as input the information about revenues
    returns a pie-chart with the splitted revenues
    '''
    sizes   = [GKV_Umsatz, Privatumsatz]
    labels  = ['GKV_Umsatz', 'Privatumsatz']
    explode = (0, 0.1)
    colors  = ['yellowgreen', 'cornflowerblue']
    plt.pie (sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
    plt.axis('equal')
    title = 'Umsatzverteilung von '+ query_name +' in '+ query_response
    plt.suptitle(title, fontsize = 11, fontweight = 'bold')
    plt.show()
    fig = plt.figure()
    return fig
    #return  it should return the image itself

### One graph with Revenue vs costs
### One graph for costs,
### One grapg for revenues,

##it should take data from mongo --> df_total
filename1 = '1_1_Production_data_doctors.csv' 
df_total  = read_csv(filename1, encoding="ISO-8859-1") ##dataframe_profits
df_total.keys()

#here you are an example, you can play around
query_name     = "Area" ##Sex  ##Area      ##Age             ## region ##typeOfCity
query_response = "Ost" ##f, m ##West, Ost ##<40, 40_50, >50 ## all 17 ##big, rural, small_medium


(rev_mean, GKV_Umsatz, Privatumsatz, cost_mean, 
 Materialaufwand, Personalkosten, AfA, Raumkosten, Zinsen, 
 Sonstige_Kosten) = data_subset_modifier( 
                                         df_total, query_name, query_response
                                         )
### 3 different graphs:cost-revenues, costs pie, revenues pie
profit_split = plot_bar (rev_mean, cost_mean, query_name, query_response)
profit_split.savefig('profit_split.png')

costs_split = plot_costs (Materialaufwand, Personalkosten, AfA, Raumkosten, Zinsen, 
            			  Sonstige_Kosten, query_name, query_response)
costs_split.savefig('costs_split.png')

revenues_split = plot_revenues(GKV_Umsatz, Privatumsatz, query_name, query_response)
revenues_split.savefig('revenues_split.png')
