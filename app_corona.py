import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import scipy.optimize as optim
import math


def carga_datos():
	# CARGAMOS DATOS AUTOMÁTICAMENTE
	url_confirmed = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
	url_death = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv"
	url_recovered = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv"

	def to_dataframe(url):
	    df = pd.DataFrame([x.split(',') for x in requests.get(url).content.decode().split('\n')])
	    header = df.iloc[0]
	    df = df[1:]
	    df.columns = header
	    return df

	confirmed = to_dataframe(url_confirmed)
	death = to_dataframe(url_death)
	recovered = to_dataframe(url_recovered)

	confirmed_long = pd.melt(confirmed, id_vars=confirmed.columns[:4],value_vars=confirmed.columns[4:],
		                var_name="Date",value_name="Confirmed").astype({'Confirmed':'float32'})
	death_long = pd.melt(death, id_vars=death.columns[:4],value_vars=death.columns[4:],
		                var_name="Date",value_name="Death").astype({'Death':'float32'})
	recovered_long = pd.melt(recovered, id_vars=recovered.columns[:4],value_vars=recovered.columns[4:],
		                var_name="Date",value_name="Recovered").astype({'Recovered':'float32'})
	
	df_full = confirmed_long.merge(death_long,how='inner').merge(recovered_long,how='inner')
	df_full.Date = pd.to_datetime(df_full.Date)
	st.success("**DATOS CARGADOS**")
	return df_full

df_full = carga_datos()



################################################################################################3
################################################################################################3

# Texto/título
st.title("Información del COVID19")


# Cabecera/Subcabecera
st.header("Datos extraídos de: ")

if st.button('Mostrar tabla con todos los datos'):
	st.subheader("TABLA CON LOS DATOS")	
	st.write(df_full)




pais = st.selectbox("Pais: ",["Spain","Italy","China",'United Kingdom'])

if pais:
	st.write(df_full[df_full['Country/Region']==pais].groupby(['Country/Region','Date']).aggregate({"Confirmed":"sum","Death":"sum","Recovered":"sum"}))


################################################################################################3
################################################################################################3

if st.button('Mostrar código de carga'):
	with st.echo():
		# CÓDIGO PARA EXTRACCIÓN DE DATOS	# CARGAMOS DATOS AUTOMÁTICAMENTE
		url_confirmed = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
		url_death = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv"
		url_recovered = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv"

		def to_dataframe(url):
		    df = pd.DataFrame([x.split(',') for x in requests.get(url).content.decode().split('\n')])
		    header = df.iloc[0]
		    df = df[1:]
		    df.columns = header
		    return df

		confirmed = to_dataframe(url_confirmed)
		death = to_dataframe(url_death)
		recovered = to_dataframe(url_recovered)

		confirmed_long = pd.melt(confirmed, id_vars=confirmed.columns[:4],value_vars=confirmed.columns[4:],
				        var_name="Date",value_name="Confirmed").astype({'Confirmed':'float32'})
		death_long = pd.melt(death, id_vars=death.columns[:4],value_vars=death.columns[4:],
				        var_name="Date",value_name="Death").astype({'Death':'float32'})
		recovered_long = pd.melt(recovered, id_vars=recovered.columns[:4],value_vars=recovered.columns[4:],
				        var_name="Date",value_name="Recovered").astype({'Recovered':'float32'})
		
		df_full = confirmed_long.merge(death_long,how='inner').merge(recovered_long,how='inner')
		df_full.Date = pd.to_datetime(df_full.Date)



indice = st.slider('Cantidad',1,20)

plt.plot(df_full[df_full['Country/Region']=='Spain'].Confirmed)
plt.plot(df_full[df_full['Country/Region']=='Italy'].Confirmed)
plt.show()

st.pyplot()


###########################################################################
###########################################################################

st.header('AJUSTE CON MODELO LOGÍSTICO')

st.subheader('Los puntos son los valores reales, y la curva es el modelo ajustado. Se debe seleccionar el país, y los datos a ajustar que pueden ser *Confirmed*, *Death* o *Recovered*')

def logistica(t,a,b,c):
    return c/(1+a*np.exp(-b*t))

	
pais_modelo = st.selectbox("Pais para el modelo: ",["Spain","Italy","China",'United Kingdom'])
serie_modelo = st.selectbox("Confirmados, Muertes, Recuperados",["Confirmed","Death","Recovered"])

if st.button('Ajuste a una curva Logística:'):


	p0 = np.random.exponential(size=3)
	#bounds = (0, [100000000.,10.,100000000000.])

	y = np.array(df_full[df_full['Country/Region']==pais_modelo][serie_modelo].dropna())
	if pais_modelo == 'China' or pais_modelo == 'United Kingdom':
		y = np.array(df_full[df_full['Country/Region']==pais_modelo].groupby(['Country/Region','Date'])[serie_modelo].sum().dropna())
	x = np.array(range(len(y)))

	(a,b,c), cov = optim.curve_fit(logistica,x,y,p0=p0)#bounds=bounds,p0=p0)


	def curva_logist(t):
	    return c/(1+a*np.exp(-b*t))
	plt.scatter(x,y)
	plt.plot(x,curva_logist(x))
	plt.title("Ajuste del modelo logístico ("+serie_modelo+","+pais+")")
	plt.legend(['Modelo logístico', 'Datos reales'])
	plt.xlabel('Time')
	plt.ylabel('Infecciones')

	st.pyplot()

	plt.plot(y-curva_logist(x))
	plt.title("Errores del modelo")
	plt.xlabel('Time')
	plt.ylabel("Errores")
	plt.show()

	st.pyplot()

