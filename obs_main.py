import pandas as pd
from datetime import datetime as dt
import datetime 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import plot_roc_curve
import torch
import math
import matplotlib.pyplot as plt
import numpy as np
import time
import pickle
from scipy.spatial import distance
import networkx as nx
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta

%matplotlib inline

obstruccion = pd.read_excel("ANALISIS_ELEMENTOS.xlsx",sheet_name='OBSTRUCCION')
obstruccion['COOR_X'].fillna(0,inplace=True)
obstruccion['COOR_Y'].fillna(0,inplace=True)

lavados = pd.read_excel("Lavados redes.xlsx")

redes = pd.read_excel("ANALISIS_ELEMENTOS.xlsx",sheet_name='REDES')

df_arbol_urbano = pd.read_excel("Arbol_Urbano_Completo.xlsx", sheet_name='Hoja1')

df_densidad_clientes_1 = pd.read_excel("Densidad_Clientes_Todas_1.xlsx", sheet_name='Todos')
df_densidad_clientes_2 = pd.read_excel("Densidad_Clientes_Todas_2.xlsx",sheet_name='Todos')
df_densidad_clientes = pd.concat([df_densidad_clientes_1,df_densidad_clientes_2])

df_densidad_residuos = pd.read_excel("Puntos_Criticos.xlsx")

df_estaciones_meteoro = pd.read_excel("Estaciones_Metereologicas_SIATA.xlsx", sheet_name='Hoja1')
with open('PrecipitacionesDict.pkl', 'rb') as f:
    PrecipitacionesDict = pickle.load(f)

adyacencia = pd.read_excel("ANALISIS_ELEMENTOS.xlsx",sheet_name='ADYACENCIA')
adyacencia['IPID_INICIO'] = adyacencia['IPID_INICIO'].astype(int)
adyacencia['IPID_FIN'] = adyacencia['IPID_FIN'].astype(int)


def limpieza_redes():
    redes['COOR_X'].fillna(0,inplace=True)
    redes['COOR_Y'].fillna(0,inplace=True)
    redes['MATERIAL'][redes['MATERIAL']=='ACERO REVESTIDO EN CONCRETO'] = 'CONCRETO'
    redes['MATERIAL'][redes['MATERIAL']=='CIPP'] = 'OTRO'
    redes['MATERIAL'][redes['MATERIAL']=='CONCRETE CYLINDER PIPE'] = 'OTRO'
    redes['MATERIAL'][redes['MATERIAL']=='CONCRETO CLASE 1'] = 'CONCRETO'
    redes['MATERIAL'][redes['MATERIAL']=='CONCRETO CLASE 2'] = 'CONCRETO'
    redes['MATERIAL'][redes['MATERIAL']=='CONCRETO CLASE 3'] = 'CONCRETO'
    redes['MATERIAL'][redes['MATERIAL']=='CONCRETO CLASE I'] = 'CONCRETO'
    redes['MATERIAL'][redes['MATERIAL']=='CONCRETO CLASE II'] = 'CONCRETO'
    redes['MATERIAL'][redes['MATERIAL']=='CONCRETO CLASE III'] = 'CONCRETO'
    redes['MATERIAL'][redes['MATERIAL']=='CONCRETO CLASE III'] = 'CONCRETO'
    redes['MATERIAL'][redes['MATERIAL']=='CONCRETO CLASE III + POLIETILENO'] = 'CONCRETO'
    redes['MATERIAL'][redes['MATERIAL']=='CONCRETO CLASE IV'] = 'CONCRETO'
    redes['MATERIAL'][redes['MATERIAL']=='CONCRETO CLASE IV + POLIETILENO'] = 'CONCRETO'
    redes['MATERIAL'][redes['MATERIAL']=='CONCRETO CLASE V'] = 'CONCRETO'
    redes['MATERIAL'][redes['MATERIAL']=='CONCRETO PARA PIPE JACKING'] = 'OTRO'
    redes['MATERIAL'][redes['MATERIAL']=='CONCRETO REVESTIDO EN POLIETILENO'] = 'POLIETILENO'
    redes['MATERIAL'][redes['MATERIAL']=='GRES VITRIFICADO'] = 'OTRO'
    redes['MATERIAL'][redes['MATERIAL']=='HIERRO DUCTIL'] = 'OTRO'
    redes['MATERIAL'][redes['MATERIAL']=='HIERRO FUNDIDO'] = 'OTRO'
    redes['MATERIAL'][redes['MATERIAL']=='NOVALOC'] = 'NOVAFORT'
    redes['MATERIAL'][redes['MATERIAL']=='POLIESTER REFORZADO CON FIBRA DE VIDRIO'] = 'FIBRA DE VIDRIO'
    redes['MATERIAL'][redes['MATERIAL']=='NOVALOC'] = 'NOVAFORT'
    redes['MATERIAL'][redes['MATERIAL']=='POLIPROPILENO'] = 'OTRO'
    redes['MATERIAL'][redes['MATERIAL']=='RESINA TERMOESTABLE REFORZADA (FIBRA DE VIDRIO)'] = 'FIBRA DE VIDRIO'
    redes['MATERIAL'][redes['MATERIAL']=='SIN INFORMACION'] = 'OTRO'
    redes['MATERIAL'][redes['MATERIAL']=='POLIETILENO'] = 'POLIETILENO DE ALTA DENSIDAD'
    redes['TIPO_AGUA'][redes['TIPO_AGUA']=="DESCARGA DE CONDUCCION"] = "DESCARGA"
    redes['TIPO_AGUA'][redes['TIPO_AGUA']=="DESCARGA DE PLANTA"] = "DESCARGA"
    redes['TIPO_AGUA'][redes['TIPO_AGUA']=="DESCARGA DE TANQUE"] = "DESCARGA"
    redes['Año_instalacion'] = pd.DatetimeIndex(redes['FECHA_INSTALACION']).year
    redes['Año_instalacion'].fillna(1987,inplace=True)
    redes['Año_instalacion'] = redes['Año_instalacion'].astype(int)
    redes['Año_instalacion'][redes['Año_instalacion']>2022] = redes['Año_instalacion'] - 100
    redes.drop('FECHA_INSTALACION',axis=1,inplace=True)
    redes['IPID'] = redes['IPID'].astype(int)
    redes['IPID_FIN_ELEMENTO'] = redes['IPID_FIN_ELEMENTO'].astype(int)
    redes.drop('CUENCA',axis=1,inplace=True)
    redes['ARRANQUE'].fillna('NO',inplace=True)
    redes['PENDIENTE'].fillna(redes['PENDIENTE'].mean(),inplace=True)
    redes['COTA_BATEA_ABAJO'].fillna(redes['COTA_BATEA_ABAJO'].mean(),inplace=True)
    redes['COTA_BATEA_ARRIBA'].fillna(redes['COTA_BATEA_ARRIBA'].mean(),inplace=True)
    redes['PROF_BATEA_SALIDA'].fillna(redes['PROF_BATEA_SALIDA'].mean(),inplace=True)
    redes['PROF_BATEA_ENTRADA'].fillna(redes['PROF_BATEA_ENTRADA'].mean(),inplace=True)
    redes['MATERIAL'].fillna('CONCRETO',inplace=True)
    redes_dm = pd.get_dummies(redes)
    return(redes_dm)


def agregar_arista(G, u, v, w=1, di=True):
    G.add_edge(u, v, weight=w)

    if not di:
        G.add_edge(v, u, weight=w)

G = nx.DiGraph()
for index in range(adyacencia.shape[0]):
  inicio = adyacencia.iloc[index,0]
  fin = adyacencia.iloc[index,1]
  agregar_arista(G, inicio, fin)


def graph_grafica(cuenca):
    G_cuenca = nx.DiGraph()
    adyacencia_cuenca = adyacencia
    adyacencia_cuenca = adyacencia_cuenca[adyacencia_cuenca['CUENCA']==cuenca]
    for index in range(adyacencia_cuenca.shape[0]):
        inicio = adyacencia_cuenca.iloc[index,0]
        fin = adyacencia_cuenca.iloc[index,1]
        agregar_arista(G_cuenca, inicio, fin)
    plt.figure(figsize=(30,16))
    pos = nx.spring_layout(G_cuenca, k=0.8)
    nx.draw(G_cuenca, pos , with_labels = False, width=0.8, node_color='lightblue', node_size=800)
    plt.show()

redes_dm = limpieza_redes
redes_dm["Degree"] = dict(G.degree()).values()
redes_dm["PageRank"] = dict(nx.pagerank(G)).values()
redes_dm["Centrality"] = dict(nx.betweenness_centrality(G)).values()


def select_sensor_siata(año, semana, X_Activo, Y_Activo):
    if (X_Activo==0 or Y_Activo==0):
      return(0,0,0)

    node = np.array([[X_Activo, Y_Activo]])
    nodes = df_estaciones_meteoro[['X','Y']].to_numpy()
    loop = True
    precip_media = 0
    precip_max = 0
    precip_sum = 0
    while loop:
      closest_index = distance.cdist(node, nodes).argmin()
      Id_Sensor_Siata = df_estaciones_meteoro.iloc[closest_index]['Id_Sensor_Siata']
      año_filtro_precipitacion = PrecipitacionesDict[Id_Sensor_Siata]['year'] == año
      semana_filtro_precipitacion = PrecipitacionesDict[Id_Sensor_Siata]['week'] == semana
      filtro_año_semana = año_filtro_precipitacion & semana_filtro_precipitacion
      df_siata_filtrado = PrecipitacionesDict[Id_Sensor_Siata][filtro_año_semana]
      if len(df_siata_filtrado) > 0:
        precip_media = df_siata_filtrado['Media'].iloc[0]
        precip_max = df_siata_filtrado['Maxima'].iloc[0]
        precip_sum = df_siata_filtrado['Suma'].iloc[0]        
        loop = False
      if nodes.shape[0] > 0:        
        nodes = np.delete(nodes, closest_index, axis=0)
      else:
        loop = False
    return precip_media,precip_max,precip_sum

def iso_year_start(año):
    "The gregorian calendar date of the first day of the given ISO year"
    fourth_jan = datetime.date(año, 1, 4)
    delta = datetime.timedelta(fourth_jan.isoweekday()-1)
    return fourth_jan - delta 

def iso_to_gregorian(año, semana, iso_day):
    "Gregorian calendar date for the given ISO year, week and day"
    year_start = iso_year_start(año)
    return year_start + datetime.timedelta(days=iso_day-1, weeks=semana-1)   

def week_to_date(año, semana):      
    dt = iso_to_gregorian(año, semana, 1)    
    firstdate = dt
    lastdate = firstdate + relativedelta(days=7)
    return firstdate, lastdate

def arbol_urbano(año, semana, X_Activo, Y_Activo):
    firstdate, lastdate = week_to_date(año,semana)
    Rango_Geoespacial = 0.0009
    fecha_filtro = df_arbol_urbano['FECHA_INGR'] <= lastdate.strftime("%Y-%m-%d")
    Geo_X_Inf_filtro = df_arbol_urbano['X'] >= X_Activo - Rango_Geoespacial
    Geo_X_Sup_filtro = df_arbol_urbano['X'] <= X_Activo + Rango_Geoespacial
    Geo_Y_Inf_filtro = df_arbol_urbano['Y'] >= Y_Activo - Rango_Geoespacial
    Geo_Y_Sup_filtro = df_arbol_urbano['Y'] <= Y_Activo + Rango_Geoespacial      
    between = fecha_filtro & Geo_X_Inf_filtro & Geo_X_Sup_filtro & Geo_Y_Inf_filtro & Geo_Y_Sup_filtro  
    df_arbol_filtrado = df_arbol_urbano[between] 
    num_arboles = df_arbol_filtrado.shape[0]
    return(num_arboles)

def densidad_clientes(año, semana, X_Activo, Y_Activo):
    firstdate, lastdate = week_to_date(año,semana)
    Rango_Geoespacial = 0.0009
    fecha_filtro = df_densidad_clientes['FECHA_CREACION_PRODUCTO'] <= lastdate.strftime("%Y-%m-%d")
    Geo_X_Inf_filtro = df_densidad_clientes['LATITUD'] >= X_Activo - Rango_Geoespacial
    Geo_X_Sup_filtro = df_densidad_clientes['LATITUD'] <= X_Activo + Rango_Geoespacial
    Geo_Y_Inf_filtro = df_densidad_clientes['LONGITUD'] >= Y_Activo - Rango_Geoespacial
    Geo_Y_Sup_filtro = df_densidad_clientes['LONGITUD'] <= Y_Activo + Rango_Geoespacial      
    between = fecha_filtro & Geo_X_Inf_filtro & Geo_X_Sup_filtro & Geo_Y_Inf_filtro & Geo_Y_Sup_filtro  
    df_densidad_clientes_filtrado = df_densidad_clientes[between] 
    num_clientes = df_densidad_clientes_filtrado.shape[0]
    return(num_clientes)

def densidad_residuos(año, semana, X_Activo, Y_Activo):
    firstdate, lastdate = week_to_date(año,semana)
    Rango_Geoespacial = 0.0009
    fecha_filtro = df_densidad_residuos['FECHA_IDENTIFICACION'] <= lastdate.strftime("%Y-%m-%d")
    Geo_X_Inf_filtro = df_densidad_residuos['LONGITUD'] >= X_Activo - Rango_Geoespacial
    Geo_X_Sup_filtro = df_densidad_residuos['LONGITUD'] <= X_Activo + Rango_Geoespacial
    Geo_Y_Inf_filtro = df_densidad_residuos['LATITUD'] >= Y_Activo - Rango_Geoespacial
    Geo_Y_Sup_filtro = df_densidad_residuos['LATITUD'] <= Y_Activo + Rango_Geoespacial      
    between = fecha_filtro & Geo_X_Inf_filtro & Geo_X_Sup_filtro & Geo_Y_Inf_filtro & Geo_Y_Sup_filtro  
    df_densidad_residuos_filtrado = df_densidad_residuos[between] 
    num_residuos = df_densidad_residuos_filtrado.shape[0]
    return(num_residuos)

def principal(epoca):
    epoch = epoca
    anos = [2018,2019,2020,2021]
    for ep in range(epoch):
        for ano in anos:
            for semana in range(2,53):
                lista_elementos_t1 = obstruccion[(obstruccion['AÑO']==ano)&(obstruccion['SEMANA']==semana)]
                lista_elementos_t1_ipid = lista_elementos_t1['IPID']
                lista_elementos_t = obstruccion[(obstruccion['AÑO']==ano)&(obstruccion['SEMANA']==semana-1)]
                lista_elementos_t_ipid = lista_elementos_t['IPID']
                obstruccion_list_t1 = []
                obstruccion_list_t = []
                edad_obs = []
                control_lav = []
                cont_mto = []
                control_obs_list = []
                cont_obs_list = []
                precip_media_list = []
                precip_max_list = []
                precip_sum_list = []
                arboles_list = []
                clientes_list = []
                residuos_list = []
                redes_sub = redes_dm

                obsdate = str(ano)+" "+str(semana)
                dateobs = dt.strptime(obsdate + ' 1', "%Y %W %w")

                for elemento in range(redes.shape[0]):
                    if redes_sub.loc[elemento,'IPID'] in lista_elementos_t_ipid or redes_sub.loc[elemento,'IPID_FIN_ELEMENTO'] in lista_elementos_t_ipid:
                        obstruccion_list_t.append(1)
                    else:
                        obstruccion_list_t.append(0)

                    if redes_sub.loc[elemento,'IPID'] in lista_elementos_t1_ipid or redes_sub.loc[elemento,'IPID_FIN_ELEMENTO'] in lista_elementos_t1_ipid:
                        obstruccion_list_t1.append(1)
                    else:
                        obstruccion_list_t1.append(0)

                redes_sub['Obstruccion_actual'] = obstruccion_list_t
                redes_sub['Obstruccion_futura'] = obstruccion_list_t1

                redes_si = redes_sub[redes_sub['Obstruccion_futura']=='SI']
                redes_no_1 = redes_sub[(redes_sub['Obstruccion_futura']=='NO')&(redes_sub['Obstruccion_actual']=='SI')]
                if (redes_si.shape[0] + redes_no_1.shape[0]) < 300:
                    redes_no_2 = redes_sub[(redes_sub['Obstruccion_futura']=='NO')&(redes_sub['Obstruccion_actual']=='NO')]
                    redes_no_2 = redes_sub.sample(n=300-(redes_si.shape[0]+redes_no_1.shape[0]))
                    frames = [redes_si, redes_no_1,redes_no_2]
                    df = pd.concat(frames)
                else:
                    redes_no_1 = redes_no_1.sample(n=300-redes_si.shape[0]) 
                    frames = [redes_si, redes_no_1]
                    df = pd.concat(frames)
       
                #calculos sobre el df filtrado
                for indice in df.index:

                #Edad cuando se obstruyó
                    edad = ano - df.loc[indice,'Año_instalacion']
                    if edad < 0:
                        edad=0
                    edad_obs.append(edad)

                #Lavado preventivo cuando se obstruyó
                    lavados_filtro = lavados['FECHA'][lavados['IPID']==df.loc[indice,'IPID']]
                    list_lavados_dif = []
                    list_lavados_control = []
                    if lavados_filtro.shape[0] > 0:
                        for i in lavados_filtro.index:
                            datelav = lavados.loc[i,'FECHA']
                            datelav = dt.strptime(datelav, '%Y-%M-%d')
                            dif_mtto = dateobs - datelav
                            dif_mtto = dif_mtto.days
                            if dif_mtto < 0:
                                dif_mtto = 1600
                            list_lavados_dif.append(dif_mtto)
                            if dif_mtto < 180:
                                control_lavado = 1
                            else:
                                control_lavado = 0
                            list_lavados_control.append(control_lavado)
                        dif_mtto = min(list_lavados_dif)
                        control_lavado = max(list_lavados_control)
                    else:
                        control_lavado = 0
                        dif_mtto = 1600
                    control_lav.append(control_lavado)
                    cont_mto.append(dif_mtto)

                    #Correctivos
                    obstrucciones_filtro = obstruccion['DATE'][obstruccion['IPID']==df.loc[indice,'IPID']]
                    list_obs_dif = []
                    list_obs_control = []
                    if obstrucciones_filtro.shape[0] > 0:
                        for i in obstrucciones_filtro.index:
                            fecha_obs = obstruccion.loc[i,'DATE']
                            fecha_obs = dt.strptime(fecha_obs, '%Y-%M-%d')
                            dif_obs = dateobs - fecha_obs
                            dif_obs = dif_obs.days
                            if dif_obs < 0:
                                dif_obs = 1600
                            list_obs_dif.append(dif_obs)
                            if dif_obs < 365:
                                control_obs = 1
                            else:
                                control_obs = 0
                            list_obs_control.append(control_obs)
                        dif_obs = min(list_obs_dif)
                        control_obs = max(list_obs_control)
                    else:
                        control_obs = 0
                        dif_obs = 1600
                    control_obs_list.append(control_obs)
                    cont_obs_list.append(dif_obs)

        
                    #Sensores
                    semana_prima = semana - 1
                    semana_str = str(semana_prima).rjust(2,'0')

                    coor_x = df.loc[indice, 'COOR_X'].tolist()
                    coor_y = df.loc[indice, 'COOR_Y'].tolist()

                    precip_media, precip_max, precip_sum = select_sensor_siata(ano, semana_str, coor_x, coor_y)
                    precip_media_list.append(precip_media)
                    precip_max_list.append(precip_max)
                    precip_sum_list.append(precip_sum)

                    #Arboles urbanos
                    num_arboles = arbol_urbano(ano, semana, coor_x, coor_y)
                    arboles_list.append(num_arboles)

                    ##Densidad de clientes
                    clientes = densidad_clientes(ano, semana, coor_x, coor_y)
                    clientes_list.append(clientes)

                    ##Densidad residuos
                    residuos = densidad_residuos(ano, semana, coor_x, coor_y)
                    residuos_list.append(residuos)


                #Agrupar todo en el DataFrame
                df['Edad'] = edad_obs
                df['Tiempo lavado'] = cont_mto
                df['Control lavado'] = control_lav
                df['Tiempo obstruccion'] = cont_obs_list
                df['Control correctivo'] = control_obs_list
                df['precip_media'] = precip_media_list
                df['precip_max'] = precip_max_list
                df['precip_sum'] = precip_sum_list
                df['Numero_arboles'] = arboles_list
                df['Densidad clientes'] = clientes_list
                df['Cantidad residuos'] = residuos_list

                return(df)
