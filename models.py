#!/usr/bin/env python
# coding: utf-8

# In[2]:



import pickle
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.metrics import mean_squared_error,r2_score
import plotly.graph_objects as go
from sklearn.svm import SVR
from datetime import timedelta
from statsmodels.tsa.api import Holt,SimpleExpSmoothing,ExponentialSmoothing
from pmdarima.arima import auto_arima


# In[3]:


from flask import Flask


# In[4]:


df=pd.read_csv('data.csv')
df.head()


# In[5]:


df.shape


# In[6]:


df.nunique() 


# In[7]:


df.drop(columns ='geoId', inplace = True)
df.drop(columns ='popData2020', inplace = True)
df.drop(columns ='continentExp', inplace = True)
df.drop(columns ='countryterritoryCode', inplace = True)
df


# In[8]:


df.isnull().sum()


# In[9]:


df=df.dropna()


# In[10]:


df.isnull().sum()


# In[11]:


df['dateRep'] = pd.to_datetime(df['dateRep'],dayfirst=True)


# In[12]:


df['cases'] = df['cases'].abs()
df['deaths'] = df['deaths'].abs()


# In[13]:




grouped = df.groupby('dateRep')['dateRep', 'cases', 'deaths'].sum().reset_index()

fig = px.line(grouped, x="dateRep", y="cases", 
              title="Worldwide Confirmed Cases Over Time")
fig.show()

fig = px.line(grouped, x="dateRep", y="cases", 
              title="Worldwide Confirmed Cases (Logarithmic Scale) Over Time", 
              log_y=True)
fig.show()


# In[14]:



grouped = df.groupby('dateRep')['dateRep', 'cases', 'deaths'].sum().reset_index()

fig = px.line(grouped, x="dateRep", y="deaths", 
              title="Worldwide Confirmed deaths Over Time")
fig.show()

fig = px.line(grouped, x="dateRep", y="deaths", 
              title="Worldwide Confirmed deaths (Logarithmic Scale) Over Time", 
              log_y=True)
fig.show()


# In[15]:


grouped_Greece = df[df['countriesAndTerritories'] == "Greece"].reset_index()
grouped_Sweden = df[df['countriesAndTerritories'] == "Sweden"].reset_index()
grouped_Spain = df[df['countriesAndTerritories'] == "Spain"].reset_index()
grouped_Portugal = df[df['countriesAndTerritories'] == "Portugal"].reset_index()
grouped_Bulgaria = df[df['countriesAndTerritories'] == "Bulgaria"].reset_index()
grouped_Norway = df[df['countriesAndTerritories'] == "Norway"].reset_index()
grouped_Greece_date = grouped_Greece.groupby('dateRep')['dateRep', 'cases', 'deaths'].sum().reset_index()
grouped_Sweden_date = grouped_Sweden.groupby('dateRep')['dateRep', 'cases', 'deaths'].sum().reset_index()
grouped_Spain_date = grouped_Spain.groupby('dateRep')['dateRep', 'cases', 'deaths'].sum().reset_index()
grouped_Portugal_date = grouped_Portugal.groupby('dateRep')['dateRep', 'cases', 'deaths'].sum().reset_index()
grouped_Bulgaria_date = grouped_Bulgaria.groupby('dateRep')['dateRep', 'cases', 'deaths'].sum().reset_index()
grouped_Norway_date = grouped_Norway.groupby('dateRep')['dateRep', 'cases', 'deaths'].sum().reset_index()


# In[16]:


plot_titles = ['Greece', 'Bulgaria', 'Sweden', 'Norway', 'Spain' , 'Portugal']

fig = px.line(grouped_Greece_date, x="dateRep", y="cases", 
              title=f"Confirmed Cases in {plot_titles[0].upper()} Over Time", 
              color_discrete_sequence=['#F61067'],
              height=500
             
             )
fig.show()

fig = px.line(grouped_Bulgaria_date, x="dateRep", y="cases", 
              title=f"Confirmed Cases in {plot_titles[1].upper()} Over Time", 
              color_discrete_sequence=['#F61067'],
              height=500
             )
fig.show()

fig = px.line(grouped_Sweden_date, x="dateRep", y="cases", 
              title=f"Confirmed Cases in {plot_titles[2].upper()} Over Time", 
              color_discrete_sequence=['#6F2DBD'],
              height=500
             )
fig.show()

fig = px.line(grouped_Norway_date, x="dateRep", y="cases", 
              title=f"Confirmed Cases in {plot_titles[3].upper()} Over Time", 
              color_discrete_sequence=['#6F2DBD'],
              height=500
             )
fig.show()
fig = px.line(grouped_Spain_date, x="dateRep", y="cases", 
              title=f"Confirmed Cases in {plot_titles[2].upper()} Over Time", 
              color_discrete_sequence=['#91C4F2'],
              height=500
             )
fig.show()

fig = px.line(grouped_Portugal_date, x="dateRep", y="cases", 
              title=f"Confirmed Cases in {plot_titles[3].upper()} Over Time", 
              color_discrete_sequence=['#91C4F2'],
              height=500
             )
fig.show()


# In[17]:


cols = ['cases', 'deaths']
date=grouped_Greece_date['dateRep']
gr=grouped_Greece_date[cols].cumsum(axis=0)
gr=gr.join(date)
date=grouped_Bulgaria_date['dateRep']
bu=grouped_Bulgaria_date[cols].cumsum(axis=0)
bu=bu.join(date)
date=grouped_Norway_date['dateRep']
no=grouped_Norway_date[cols].cumsum(axis=0)
no=no.join(date)
date=grouped_Sweden_date['dateRep']
sw=grouped_Sweden_date[cols].cumsum(axis=0)
sw=sw.join(date)
date=grouped_Spain_date['dateRep']
sp=grouped_Spain_date[cols].cumsum(axis=0)
sp=sp.join(date)
date=grouped_Portugal_date['dateRep']
po=grouped_Portugal_date[cols].cumsum(axis=0)
po=po.join(date)


# In[18]:


plot_titles = ['Greece', 'Bulgaria', 'Sweden', 'Norway', 'Spain' , 'Portugal']

fig = px.line(gr, x="dateRep", y="cases", 
              title=f"Confirmed Cases in {plot_titles[0].upper()} Over Time", 
              color_discrete_sequence=['#F61067'],
              height=500
             
             )
fig.show()

fig = px.line(bu, x="dateRep", y="cases", 
              title=f"Confirmed Cases in {plot_titles[1].upper()} Over Time", 
              color_discrete_sequence=['#F61067'],
              height=500
             )
fig.show()

fig = px.line(sw, x="dateRep", y="cases", 
              title=f"Confirmed Cases in {plot_titles[2].upper()} Over Time", 
              color_discrete_sequence=['#6F2DBD'],
              height=500
             )
fig.show()

fig = px.line(no, x="dateRep", y="cases", 
              title=f"Confirmed Cases in {plot_titles[3].upper()} Over Time", 
              color_discrete_sequence=['#6F2DBD'],
              height=500
             )
fig.show()
fig = px.line(sp, x="dateRep", y="cases", 
              title=f"Confirmed Cases in {plot_titles[2].upper()} Over Time", 
              color_discrete_sequence=['#91C4F2'],
              height=500
             )
fig.show()

fig = px.line(po, x="dateRep", y="cases", 
              title=f"Confirmed Cases in {plot_titles[3].upper()} Over Time", 
              color_discrete_sequence=['#91C4F2'],
              height=500
             )
fig.show()


# In[19]:



latest_grouped = df.groupby('countriesAndTerritories')['cases', 'deaths'].sum().reset_index()

fig = px.choropleth(latest_grouped, locations="countriesAndTerritories", 
                   locationmode='country names', color="cases", hover_name="countriesAndTerritories", scope="europe",
                   range_color=[1,10000000], 
                   color_continuous_scale="peach", 
                   title='Countries with Confirmed Cases')
fig.show()


# In[20]:


latest_grouped = df.groupby('countriesAndTerritories')['cases', 'deaths'].sum().reset_index()

fig = px.choropleth(latest_grouped, locations="countriesAndTerritories", 
                    locationmode='country names', color="deaths", hover_name="countriesAndTerritories", scope="europe",
                    range_color=[1,100000], 
                    color_continuous_scale="peach", 
                    title='Countries with Confirmed Cases')
fig.show()


# In[21]:




formated_gdf = df.groupby(['dateRep', 'countriesAndTerritories'])['cases', 'deaths'].max() 
formated_gdf = formated_gdf.reset_index()
formated_gdf['dateRep'] = pd.to_datetime(formated_gdf['dateRep'])
formated_gdf['dateRep'] = formated_gdf['dateRep'].dt.strftime('%d/%m/%Y')
formated_gdf['size'] = formated_gdf['cases'].pow(0.3)

fig = px.scatter_geo(formated_gdf, locations="countriesAndTerritories", locationmode='country names', 
                     color="cases", size='size', hover_name="countriesAndTerritories", 
                     range_color= [0, 10000], 
                     projection="natural earth", animation_frame="dateRep", scope="europe",
                     title='COVID-19: Spread Over Time', color_continuous_scale="portland")
# fig.update(layout_coloraxis_showscale=False)
fig.show()


# In[22]:


cols = ['cases', 'deaths']
gr=grouped_Greece_date[cols].sum(axis=0)/11000000
bu=grouped_Bulgaria_date[cols].sum(axis=0)/7000000
no=grouped_Norway_date[cols].sum(axis=0)/5500000
sw=grouped_Sweden_date[cols].sum(axis=0)/10500000
sp=grouped_Spain_date[cols].sum(axis=0)/48000000
po=grouped_Portugal_date[cols].sum(axis=0)/10500000


# In[23]:






plt.plot(gr['cases'],gr['deaths'], marker='.', markersize=10, color="blue")
plt.plot(bu['cases'],bu['deaths'], marker='o', markersize=10, color="red")
plt.plot(no['cases'],no['deaths'], marker='v', markersize=10, color="green")
plt.plot(sw['cases'],sw['deaths'], marker='x', markersize=10, color="grey")
plt.plot(sp['cases'],sp['deaths'], marker='8', markersize=10, color="black")
plt.plot(po['cases'],po['deaths'], marker='s', markersize=10, color="orange")


# In[24]:


grouped_country=df.groupby(["countriesAndTerritories","dateRep"]).agg({"cases":'sum',"deaths":'sum'})
grouped_country


# In[25]:


datewise=df.groupby(["dateRep"]).agg({"cases":'sum',"deaths":'sum'})
datewise["Days Since"]=datewise.index-datewise.index.min()
datewise


# In[26]:


countrywise=df.groupby(["countriesAndTerritories"]).agg({"cases":'sum',"deaths":'sum'}).sort_values(["cases"],ascending=False)
countrywise["Mortality"]=(countrywise["deaths"]/countrywise["cases"])*100
countrywise


# In[27]:


fig, (ax1, ax2) = plt.subplots(2, 1,figsize=(10,12))
countrywise_plot_mortal=countrywise[countrywise["cases"]>500].sort_values(["Mortality"],ascending=False).head(15)
sns.barplot(x=countrywise_plot_mortal["Mortality"],y=countrywise_plot_mortal.index,ax=ax1)
ax1.set_title("Top 15 Countries according High Mortatlity Rate")
ax1.set_xlabel("Mortality (in Percentage)")
top_15_deaths=countrywise.sort_values(["deaths"],ascending=False).head(15)
sns.barplot(x=top_15_deaths["deaths"],y=top_15_deaths.index,ax=ax2)
ax2.set_title("Top 15 countries as per Number of Death Cases")


# In[28]:


from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score,silhouette_samples
std=StandardScaler()
X=countrywise[["Mortality"]]
#Standard Scaling since K-Means Clustering is a distance based alogrithm
X=std.fit_transform(X) 
wcss=[]
sil=[]
for i in range(2,11):
    clf=KMeans(n_clusters=i,init='k-means++',random_state=42)
    clf.fit(X)
    labels=clf.labels_
    centroids=clf.cluster_centers_
    sil.append(silhouette_score(X, labels, metric='euclidean'))
    wcss.append(clf.inertia_)


# In[29]:


x=np.arange(2,11)
plt.figure(figsize=(10,5))
plt.plot(x,wcss,marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("Within Cluster Sum of Squares (WCSS)")
plt.title("Elbow Method")


# In[30]:


import scipy.cluster.hierarchy as sch
plt.figure(figsize=(20,15))
dendogram=sch.dendrogram(sch.linkage(X, method  = "ward"))


# In[31]:




clf_final=KMeans(n_clusters=3,init='k-means++',random_state=6)
clf_final.fit(X)


# In[32]:


countrywise["Clusters"]=clf_final.predict(X)


# In[33]:


cluster_summary=pd.concat([countrywise[countrywise["Clusters"]==1].head(15),countrywise[countrywise["Clusters"]==2].head(15),countrywise[countrywise["Clusters"]==0].head(15)])
cluster_summary.style.background_gradient(cmap='Reds').format("{:.2f}")


# In[34]:



datewise=df.groupby(["dateRep"]).agg({"cases":'sum',"deaths":'sum'})
datewise["Days Since"]=datewise.index-datewise.index.min()
datewise


# In[35]:


train_ml=datewise.iloc[:int(datewise.shape[0]*0.95)]
valid_ml=datewise.iloc[int(datewise.shape[0]*0.95):]


# In[36]:




poly = PolynomialFeatures(degree = 8) 


train_poly=poly.fit_transform(np.array(train_ml["Days Since"]).reshape(-1,1))
valid_poly=poly.fit_transform(np.array(valid_ml["Days Since"]).reshape(-1,1))
y=train_ml["cases"]



# In[37]:




linreg=LinearRegression(normalize=True)
linreg.fit(train_poly,y)


# In[38]:


LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=True)


# In[39]:


prediction_poly=linreg.predict(valid_poly)
rmse_poly=np.sqrt(mean_squared_error(valid_ml["cases"],prediction_poly))
print("Root Mean Squared Error for Polynomial Regression: ",rmse_poly)


# In[40]:


comp_data=poly.fit_transform(np.array(datewise["Days Since"]).reshape(-1,1))
plt.figure(figsize=(11,6))
predictions_poly=linreg.predict(comp_data)
fig=go.Figure()
fig.add_trace(go.Scatter(x=datewise.index, y=datewise["cases"],
                    mode='lines+markers',name="Train Data for Confirmed Cases"))
fig.add_trace(go.Scatter(x=datewise.index, y=predictions_poly,
                    mode='lines',name="Polynomial Regression Best Fit",
                    line=dict(color='black', dash='dot')))
fig.update_layout(title="Confirmed Cases Polynomial Regression Prediction",
                 xaxis_title="Date",yaxis_title="Confirmed Cases",
                 legend=dict(x=0,y=1,traceorder="normal"))
fig.show()


# In[41]:


train_ml=datewise.iloc[:int(datewise.shape[0]*0.95)]
valid_ml=datewise.iloc[int(datewise.shape[0]*0.95):]


# In[42]:


svm=SVR(C=1,degree=6,kernel='poly',epsilon=0.01)


# In[43]:


svm.fit(np.array(train_ml["Days Since"]).reshape(-1,1),np.array(train_ml["cases"]).reshape(-1,1))


# In[44]:


prediction_valid_svm=svm.predict(np.array(valid_ml["Days Since"]).reshape(-1,1))


# In[45]:




print("Root Mean Square Error for Support Vectore Machine: ",np.sqrt(mean_squared_error(valid_ml["cases"],prediction_valid_svm)))


# In[46]:




plt.figure(figsize=(11,6))
prediction_svm=svm.predict(np.array(datewise["Days Since"]).reshape(-1,1))
fig=go.Figure()
fig.add_trace(go.Scatter(x=datewise.index, y=datewise["cases"],
                    mode='lines+markers',name="Train Data for Confirmed Cases"))
fig.add_trace(go.Scatter(x=datewise.index, y=prediction_svm,
                    mode='lines',name="Support Vector Machine Best fit Kernel",
                    line=dict(color='black', dash='dot')))
fig.update_layout(title="Confirmed Cases Support Vectore Machine Regressor Prediction",
                 xaxis_title="Date",yaxis_title="Confirmed Cases",legend=dict(x=0,y=1,traceorder="normal"))
fig.show()


# In[47]:




model_train=datewise.iloc[:int(datewise.shape[0]*0.95)]
valid=datewise.iloc[int(datewise.shape[0]*0.95):]
y_pred2=valid.copy()


# In[48]:




holt=Holt(np.asarray(model_train["cases"])).fit(smoothing_level=0.2, smoothing_slope=0.1,optimized=False)     


# In[49]:




y_pred2["Holt"]=holt.forecast(len(valid))
print("Root Mean Square Error Holt's Linear Model: ",np.sqrt(mean_squared_error(y_pred2["cases"],y_pred2["Holt"])))


# In[50]:




fig=go.Figure()
fig.add_trace(go.Scatter(x=model_train.index, y=model_train["cases"],
                    mode='lines+markers',name="Train Data for Confirmed Cases"))
fig.add_trace(go.Scatter(x=valid.index, y=valid["cases"],
                    mode='lines+markers',name="Validation Data for Confirmed Cases",))
fig.add_trace(go.Scatter(x=valid.index, y=y_pred2["Holt"],
                    mode='lines+markers',name="Prediction of Confirmed Cases",))
fig.update_layout(title="Confirmed Cases Holt's Linear Model Prediction",
                 xaxis_title="Date",yaxis_title="Confirmed Cases",legend=dict(x=0,y=1,traceorder="normal"))
fig.show()


# In[51]:




model_train=datewise.iloc[:int(datewise.shape[0]*0.95)]
valid=datewise.iloc[int(datewise.shape[0]*0.95):]
y_pred3=valid.copy()
model_train= model_train[model_train['cases'] != 0]


# In[52]:


es=ExponentialSmoothing(np.asarray(model_train['cases']),seasonal_periods=7,trend='add', seasonal='add').fit()


# In[53]:


y_pred3["Holt's Winter Model"]=es.forecast(len(valid))
print("Root Mean Square Error for Holt's Winter Model: ",np.sqrt(mean_squared_error(y_pred3["cases"],y_pred3["Holt's Winter Model"])))


# 

# In[54]:




fig=go.Figure()
fig.add_trace(go.Scatter(x=model_train.index, y=model_train["cases"],
                    mode='lines+markers',name="Train Data for Confirmed Cases"))
fig.add_trace(go.Scatter(x=valid.index, y=valid["cases"],
                    mode='lines+markers',name="Validation Data for Confirmed Cases",))
fig.add_trace(go.Scatter(x=valid.index, y=y_pred3["Holt\'s Winter Model"],
                    mode='lines+markers',name="Prediction of Confirmed Cases",))
fig.update_layout(title="Confirmed Cases Holt's Winter Model Prediction",
                 xaxis_title="Date",yaxis_title="Confirmed Cases",legend=dict(x=0,y=1,traceorder="normal"))
fig.show()


# In[55]:




model_train=datewise.iloc[:int(datewise.shape[0]*0.95)]
valid4=datewise.iloc[int(datewise.shape[0]*0.95):]
y_pred4=valid4.copy()


# In[56]:




model_ar= auto_arima(model_train["cases"],trace=True, error_action='ignore', start_p=0,start_q=0,max_p=5,max_q=0,
                   suppress_warnings=True,stepwise=False,seasonal=False)
model_ar.fit(model_train["cases"])


# In[57]:


prediction_ar=model_ar.predict(len(valid4))
y_pred4["AR Model Prediction"]=prediction_ar


# In[58]:




print("Root Mean Square Error for AR Model: ",np.sqrt(mean_squared_error(y_pred4["cases"],y_pred4["AR Model Prediction"])))


# In[59]:




fig=go.Figure()
fig.add_trace(go.Scatter(x=model_train.index, y=model_train["cases"],
                    mode='lines+markers',name="Train Data for Confirmed Cases"))
fig.add_trace(go.Scatter(x=valid4.index, y=valid4["cases"],
                    mode='lines+markers',name="Validation Data for Confirmed Cases",))
fig.add_trace(go.Scatter(x=valid4.index, y=y_pred4["AR Model Prediction"],
                    mode='lines+markers',name="Prediction of Confirmed Cases",))
fig.update_layout(title="Confirmed Cases AR Model Prediction",
                 xaxis_title="Date",yaxis_title="Confirmed Cases",legend=dict(x=0,y=1,traceorder="normal"))
fig.show()


# In[60]:


model_train=datewise.iloc[:int(datewise.shape[0]*0.95)]
valid5=datewise.iloc[int(datewise.shape[0]*0.95):]
y_pred5=valid5.copy()


# In[61]:


model_ma= auto_arima(model_train["cases"],trace=True, error_action='ignore', start_p=0,start_q=0,max_p=0,max_q=5,
                   suppress_warnings=True,stepwise=False,seasonal=False)
model_ma.fit(model_train["cases"])


# In[62]:


prediction_ma=model_ma.predict(len(valid5))
y_pred5["MA Model Prediction"]=prediction_ma


# In[63]:




print("Root Mean Square Error for MA Model: ",np.sqrt(mean_squared_error(valid5["cases"],prediction_ma)))


# In[64]:


fig=go.Figure()
fig.add_trace(go.Scatter(x=model_train.index, y=model_train["cases"],
                    mode='lines+markers',name="Train Data for Confirmed Cases"))
fig.add_trace(go.Scatter(x=valid5.index, y=valid5["cases"],
                    mode='lines+markers',name="Validation Data for Confirmed Cases",))
fig.add_trace(go.Scatter(x=valid5.index, y=y_pred5["MA Model Prediction"],
                    mode='lines+markers',name="Prediction for Confirmed Cases",))
fig.update_layout(title="Confirmed Cases MA Model Prediction",
                 xaxis_title="Date",yaxis_title="Confirmed Cases",legend=dict(x=0,y=1,traceorder="normal"))
fig.show()


# In[65]:


model_train=datewise.iloc[:int(datewise.shape[0]*0.95)]
valid6=datewise.iloc[int(datewise.shape[0]*0.95):]
y_pred6=valid6.copy()


# In[66]:


model_arima= auto_arima(model_train["cases"],trace=True, error_action='ignore', start_p=0,start_q=0,max_p=5,max_q=5,
                   suppress_warnings=True,stepwise=False,seasonal=False)
model_arima.fit(model_train["cases"])


# In[67]:




prediction_arima=model_arima.predict(len(valid6))
y_pred6["ARIMA Model Prediction"]=prediction_arima


# In[68]:


print("Root Mean Square Error for ARIMA Model: ",np.sqrt(mean_squared_error(valid6["cases"],prediction_arima)))


# In[69]:


fig=go.Figure()
fig.add_trace(go.Scatter(x=model_train.index, y=model_train["cases"],
                    mode='lines+markers',name="Train Data for Confirmed Cases"))
fig.add_trace(go.Scatter(x=valid6.index, y=valid6["cases"],
                    mode='lines+markers',name="Validation Data for Confirmed Cases",))
fig.add_trace(go.Scatter(x=valid6.index, y=y_pred6["ARIMA Model Prediction"],
                    mode='lines+markers',name="Prediction for Confirmed Cases",))
fig.update_layout(title="Confirmed Cases ARIMA Model Prediction",
                 xaxis_title="Date",yaxis_title="Confirmed Cases",legend=dict(x=0,y=1,traceorder="normal"))
fig.show()


# In[70]:




model_sarima= auto_arima(model_train["cases"],trace=True, error_action='ignore', 
                         start_p=0,start_q=0,max_p=5,max_q=5,m=7,
                   suppress_warnings=True,stepwise=True,seasonal=True)
model_sarima.fit(model_train["cases"])


# In[71]:




prediction_sarima=model_sarima.predict(len(valid6))
y_pred6["SARIMA Model Prediction"]=prediction_sarima


# In[72]:


print("Root Mean Square Error for SARIMA Model: ",np.sqrt(mean_squared_error(y_pred6["cases"],y_pred6["SARIMA Model Prediction"])))


# In[73]:


fig=go.Figure()
fig.add_trace(go.Scatter(x=model_train.index, y=model_train["cases"],
                    mode='lines+markers',name="Train Data for Confirmed Cases"))
fig.add_trace(go.Scatter(x=valid6.index, y=valid6["cases"],
                    mode='lines+markers',name="Validation Data for Confirmed Cases",))
fig.add_trace(go.Scatter(x=valid6.index, y=y_pred6["SARIMA Model Prediction"],
                    mode='lines+markers',name="Prediction for Confirmed Cases",))
fig.update_layout(title="Confirmed Cases SARIMA Model Prediction",
                 xaxis_title="Date",yaxis_title="Confirmed Cases",legend=dict(x=0,y=1,traceorder="normal"))
fig.show()

pickle.dump(model_sarima, open('sarimad.pkl', 'wb'))
# In[74]:


from fbprophet import Prophet


# In[75]:




prophet_c=Prophet(interval_width=0.90,weekly_seasonality=True,)
prophet_confirmed=pd.DataFrame(zip(list(datewise.index),list(datewise["cases"])),columns=['ds','y'])
prophet_confirmed


# In[76]:


prophet_c.fit(prophet_confirmed)


# In[77]:
pickle.dump(prophet_c, open('prophetd.pkl', 'wb'))

forecast_c=prophet_c.make_future_dataframe(periods=36)
forecast_confirmed=forecast_c.copy()


# In[78]:




confirmed_forecast=prophet_c.predict(forecast_c)
print(confirmed_forecast[['ds','yhat', 'yhat_lower', 'yhat_upper']])


# In[79]:


print("Root Mean Squared Error for Prophet Model: ",np.sqrt(mean_squared_error(datewise["cases"],confirmed_forecast['yhat'].head(datewise.shape[0]))))


# In[80]:


prophet_c.plot(confirmed_forecast)


# In[81]:


print(prophet_c.plot_components(confirmed_forecast))


# In[82]:


from tabulate import tabulate
data = [['Polynomial Regression',rmse_poly ],
[' Support Vectore Machine ',np.sqrt(mean_squared_error(valid_ml["cases"],prediction_valid_svm))],
[" Holt's Linear Model ",np.sqrt(mean_squared_error(y_pred2["cases"],y_pred2["Holt"]))],
[" Holt's Winter Model ",np.sqrt(mean_squared_error(y_pred3["cases"],y_pred3["Holt's Winter Model"]))],
[" AR Model ",np.sqrt(mean_squared_error(y_pred4["cases"],y_pred4["AR Model Prediction"]))],
[" MA Model ",np.sqrt(mean_squared_error(valid5["cases"],prediction_ma))],
[" ARIMA Model  ",np.sqrt(mean_squared_error(valid6["cases"],prediction_arima))] ,
[" SARIMA Model:",np.sqrt(mean_squared_error(y_pred6["cases"],y_pred6["SARIMA Model Prediction"]))],
[" Prophet Model ",np.sqrt(mean_squared_error(datewise["cases"],confirmed_forecast['yhat'].head(datewise.shape[0])))]]
print (tabulate(data, headers=["model,mse"]))


# In[83]:


datewise=df.groupby(["dateRep"]).agg({"cases":'sum',"deaths":'sum'})
cols = ['cases', 'deaths']
datewise=datewise[cols].cumsum(axis=0)
datewise["Days Since"]=datewise.index-datewise.index.min()

datewise


# In[84]:


train_ml=datewise.iloc[:int(datewise.shape[0]*0.95)]
valid_ml=datewise.iloc[int(datewise.shape[0]*0.95):]


# In[85]:


poly = PolynomialFeatures(degree = 8) 


train_poly=poly.fit_transform(np.array(train_ml["Days Since"]).reshape(-1,1))
valid_poly=poly.fit_transform(np.array(valid_ml["Days Since"]).reshape(-1,1))
y=train_ml["cases"]


# In[86]:



linreg=LinearRegression(normalize=True)
linreg.fit(train_poly,y)


# In[87]:


LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=True)


# In[88]:


prediction_poly=linreg.predict(valid_poly)
rmse_poly=np.sqrt(mean_squared_error(valid_ml["cases"],prediction_poly))
print("Root Mean Squared Error for Polynomial Regression: ",rmse_poly)


# In[89]:


comp_data=poly.fit_transform(np.array(datewise["Days Since"]).reshape(-1,1))
plt.figure(figsize=(11,6))
predictions_poly=linreg.predict(comp_data)
fig=go.Figure()
fig.add_trace(go.Scatter(x=datewise.index, y=datewise["cases"],
                    mode='lines+markers',name="Train Data for Confirmed Cases"))
fig.add_trace(go.Scatter(x=datewise.index, y=predictions_poly,
                    mode='lines',name="Polynomial Regression Best Fit",
                    line=dict(color='black', dash='dot')))
fig.update_layout(title="Confirmed Cases Polynomial Regression Prediction",
                 xaxis_title="Date",yaxis_title="Confirmed Cases",
                 legend=dict(x=0,y=1,traceorder="normal"))
fig.show()


# In[90]:


train_ml=datewise.iloc[:int(datewise.shape[0]*0.95)]
valid_ml=datewise.iloc[int(datewise.shape[0]*0.95):]


# In[91]:


svm=SVR(C=1,degree=6,kernel='poly',epsilon=0.01)


# In[92]:


svm.fit(np.array(train_ml["Days Since"]).reshape(-1,1),np.array(train_ml["cases"]).reshape(-1,1))


# In[93]:


prediction_valid_svm=svm.predict(np.array(valid_ml["Days Since"]).reshape(-1,1))


# In[94]:


print("Root Mean Square Error for Support Vectore Machine: ",np.sqrt(mean_squared_error(valid_ml["cases"],prediction_valid_svm)))


# In[95]:



plt.figure(figsize=(11,6))
prediction_svm=svm.predict(np.array(datewise["Days Since"]).reshape(-1,1))
fig=go.Figure()
fig.add_trace(go.Scatter(x=datewise.index, y=datewise["cases"],
                    mode='lines+markers',name="Train Data for Confirmed Cases"))
fig.add_trace(go.Scatter(x=datewise.index, y=prediction_svm,
                    mode='lines',name="Support Vector Machine Best fit Kernel",
                    line=dict(color='black', dash='dot')))
fig.update_layout(title="Confirmed Cases Support Vectore Machine Regressor Prediction",
                 xaxis_title="Date",yaxis_title="Confirmed Cases",legend=dict(x=0,y=1,traceorder="normal"))
fig.show()


# In[96]:


model_train=datewise.iloc[:int(datewise.shape[0]*0.95)]
valid=datewise.iloc[int(datewise.shape[0]*0.95):]
y_pred2=valid.copy()


# In[97]:


holt=Holt(np.asarray(model_train["cases"])).fit(smoothing_level=0.3, smoothing_slope=0.3,optimized=False)     


# In[98]:


y_pred2["Holt"]=holt.forecast(len(valid))
print("Root Mean Square Error Holt's Linear Model: ",np.sqrt(mean_squared_error(y_pred2["cases"],y_pred2["Holt"])))


# In[99]:


fig=go.Figure()
fig.add_trace(go.Scatter(x=model_train.index, y=model_train["cases"],
                    mode='lines+markers',name="Train Data for Confirmed Cases"))
fig.add_trace(go.Scatter(x=valid.index, y=valid["cases"],
                    mode='lines+markers',name="Validation Data for Confirmed Cases",))
fig.add_trace(go.Scatter(x=valid.index, y=y_pred2["Holt"],
                    mode='lines+markers',name="Prediction of Confirmed Cases",))
fig.update_layout(title="Confirmed Cases Holt's Linear Model Prediction",
                 xaxis_title="Date",yaxis_title="Confirmed Cases",legend=dict(x=0,y=1,traceorder="normal"))
fig.show()


# In[100]:


model_train=datewise.iloc[:int(datewise.shape[0]*0.95)]
valid=datewise.iloc[int(datewise.shape[0]*0.95):]
y_pred3=valid.copy()
model_train= model_train[model_train['cases'] != 0]


# In[101]:


es=ExponentialSmoothing(np.asarray(model_train['cases']),seasonal_periods=14,trend='add', seasonal='add').fit()


# In[102]:


y_pred3["Holt's Winter Model"]=es.forecast(len(valid))
print("Root Mean Square Error for Holt's Winter Model: ",np.sqrt(mean_squared_error(y_pred3["cases"],y_pred3["Holt's Winter Model"])))


# In[103]:


fig=go.Figure()
fig.add_trace(go.Scatter(x=model_train.index, y=model_train["cases"],
                    mode='lines+markers',name="Train Data for Confirmed Cases"))
fig.add_trace(go.Scatter(x=valid.index, y=valid["cases"],
                    mode='lines+markers',name="Validation Data for Confirmed Cases",))
fig.add_trace(go.Scatter(x=valid.index, y=y_pred3["Holt\'s Winter Model"],
                    mode='lines+markers',name="Prediction of Confirmed Cases",))
fig.update_layout(title="Confirmed Cases Holt's Winter Model Prediction",
                 xaxis_title="Date",yaxis_title="Confirmed Cases",legend=dict(x=0,y=1,traceorder="normal"))
fig.show()


# In[104]:


model_train=datewise.iloc[:int(datewise.shape[0]*0.95)]
valid4=datewise.iloc[int(datewise.shape[0]*0.95):]
y_pred4=valid4.copy()


# In[105]:


model_ar= auto_arima(model_train["cases"],trace=True, error_action='ignore', start_p=0,start_q=0,max_p=4,max_q=0,
                   suppress_warnings=True,stepwise=False,seasonal=False)
model_ar.fit(model_train["cases"])


# In[106]:


prediction_ar=model_ar.predict(len(valid4))
y_pred4["AR Model Prediction"]=prediction_ar


# In[107]:


print("Root Mean Square Error for AR Model: ",np.sqrt(mean_squared_error(y_pred4["cases"],y_pred4["AR Model Prediction"])))


# In[108]:


fig=go.Figure()
fig.add_trace(go.Scatter(x=model_train.index, y=model_train["cases"],
                    mode='lines+markers',name="Train Data for Confirmed Cases"))
fig.add_trace(go.Scatter(x=valid4.index, y=valid4["cases"],
                    mode='lines+markers',name="Validation Data for Confirmed Cases",))
fig.add_trace(go.Scatter(x=valid4.index, y=y_pred4["AR Model Prediction"],
                    mode='lines+markers',name="Prediction of Confirmed Cases",))
fig.update_layout(title="Confirmed Cases AR Model Prediction",
                 xaxis_title="Date",yaxis_title="Confirmed Cases",legend=dict(x=0,y=1,traceorder="normal"))
fig.show()


# In[109]:


model_train=datewise.iloc[:int(datewise.shape[0]*0.95)]
valid5=datewise.iloc[int(datewise.shape[0]*0.95):]
y_pred5=valid5.copy()


# In[110]:


model_ma= auto_arima(model_train["cases"],trace=True, error_action='ignore', start_p=0,start_q=0,max_p=0,max_q=2,
                   suppress_warnings=True,stepwise=False,seasonal=False)
model_ma.fit(model_train["cases"])


# In[111]:


prediction_ma=model_ma.predict(len(valid5))
y_pred5["MA Model Prediction"]=prediction_ma


# In[112]:


print("Root Mean Square Error for MA Model: ",np.sqrt(mean_squared_error(valid5["cases"],prediction_ma)))


# In[113]:


fig=go.Figure()
fig.add_trace(go.Scatter(x=model_train.index, y=model_train["cases"],
                    mode='lines+markers',name="Train Data for Confirmed Cases"))
fig.add_trace(go.Scatter(x=valid5.index, y=valid5["cases"],
                    mode='lines+markers',name="Validation Data for Confirmed Cases",))
fig.add_trace(go.Scatter(x=valid5.index, y=y_pred5["MA Model Prediction"],
                    mode='lines+markers',name="Prediction for Confirmed Cases",))
fig.update_layout(title="Confirmed Cases MA Model Prediction",
                 xaxis_title="Date",yaxis_title="Confirmed Cases",legend=dict(x=0,y=1,traceorder="normal"))
fig.show()


# In[114]:


model_train=datewise.iloc[:int(datewise.shape[0]*0.95)]
valid6=datewise.iloc[int(datewise.shape[0]*0.95):]
y_pred6=valid6.copy()


# In[115]:


model_arima= auto_arima(model_train["cases"],trace=True, error_action='ignore', start_p=1,start_q=1,max_p=3,max_q=3,
                   suppress_warnings=True,stepwise=False,seasonal=False)
model_arima.fit(model_train["cases"])


# In[116]:


prediction_arima=model_arima.predict(len(valid6))
y_pred6["ARIMA Model Prediction"]=prediction_arima


# 

# In[117]:


print("Root Mean Square Error for ARIMA Model: ",np.sqrt(mean_squared_error(valid6["cases"],prediction_arima)))


# In[118]:


fig=go.Figure()
fig.add_trace(go.Scatter(x=model_train.index, y=model_train["cases"],
                    mode='lines+markers',name="Train Data for Confirmed Cases"))
fig.add_trace(go.Scatter(x=valid6.index, y=valid6["cases"],
                    mode='lines+markers',name="Validation Data for Confirmed Cases",))
fig.add_trace(go.Scatter(x=valid6.index, y=y_pred6["ARIMA Model Prediction"],
                    mode='lines+markers',name="Prediction for Confirmed Cases",))
fig.update_layout(title="Confirmed Cases ARIMA Model Prediction",
                 xaxis_title="Date",yaxis_title="Confirmed Cases",legend=dict(x=0,y=1,traceorder="normal"))
fig.show()


# In[119]:


model_sarima= auto_arima(model_train["cases"],trace=True, error_action='ignore', 
                         start_p=0,start_q=0,max_p=2,max_q=2,m=7,
                   suppress_warnings=True,stepwise=True,seasonal=True)
model_sarima.fit(model_train["cases"])


# In[120]:
pickle.dump(model_sarima, open('sarimat.pkl', 'wb'))

prediction_sarima=model_sarima.predict(len(valid6))
y_pred6["SARIMA Model Prediction"]=prediction_sarima


# In[121]:


print("Root Mean Square Error for SARIMA Model: ",np.sqrt(mean_squared_error(y_pred6["cases"],y_pred6["SARIMA Model Prediction"])))


# In[122]:


fig=go.Figure()
fig.add_trace(go.Scatter(x=model_train.index, y=model_train["cases"],
                    mode='lines+markers',name="Train Data for Confirmed Cases"))
fig.add_trace(go.Scatter(x=valid6.index, y=valid6["cases"],
                    mode='lines+markers',name="Validation Data for Confirmed Cases",))
fig.add_trace(go.Scatter(x=valid6.index, y=y_pred6["SARIMA Model Prediction"],
                    mode='lines+markers',name="Prediction for Confirmed Cases",))
fig.update_layout(title="Confirmed Cases SARIMA Model Prediction",
                 xaxis_title="Date",yaxis_title="Confirmed Cases",legend=dict(x=0,y=1,traceorder="normal"))
fig.show()


# In[123]:


prophet_c=Prophet(interval_width=0.95,weekly_seasonality=True,)
prophet_confirmed=pd.DataFrame(zip(list(datewise.index),list(datewise["cases"])),columns=['ds','y'])
prophet_confirmed


# In[124]:


prophet_c.fit(prophet_confirmed)

pickle.dump(prophet_c, open('prophett.pkl', 'wb'))

# In[125]:


forecast_c=prophet_c.make_future_dataframe(periods=36)
forecast_confirmed=forecast_c.copy()


# In[126]:


confirmed_forecast=prophet_c.predict(forecast_c)
print(confirmed_forecast[['ds','yhat', 'yhat_lower', 'yhat_upper']])


# In[127]:


print("Root Mean Squared Error for Prophet Model: ",np.sqrt(mean_squared_error(datewise["cases"],confirmed_forecast['yhat'].head(datewise.shape[0]))))


# In[128]:


prophet_c.plot(confirmed_forecast)


# In[129]:


print(prophet_c.plot_components(confirmed_forecast))


# In[130]:


from tabulate import tabulate
data = [['Polynomial Regression',rmse_poly ],
[' Support Vectore Machine ',np.sqrt(mean_squared_error(valid_ml["cases"],prediction_valid_svm))],
[" Holt's Linear Model ",np.sqrt(mean_squared_error(y_pred2["cases"],y_pred2["Holt"]))],
[" Holt's Winter Model ",np.sqrt(mean_squared_error(y_pred3["cases"],y_pred3["Holt's Winter Model"]))],
[" AR Model ",np.sqrt(mean_squared_error(y_pred4["cases"],y_pred4["AR Model Prediction"]))],
[" MA Model ",np.sqrt(mean_squared_error(valid5["cases"],prediction_ma))],
[" ARIMA Model  ",np.sqrt(mean_squared_error(valid6["cases"],prediction_arima))] ,
[" SARIMA Model:",np.sqrt(mean_squared_error(y_pred6["cases"],y_pred6["SARIMA Model Prediction"]))],
[" Prophet Model ",np.sqrt(mean_squared_error(datewise["cases"],confirmed_forecast['yhat'].head(datewise.shape[0])))]]
print (tabulate(data, headers=["model,mse"]))


# In[131]:



gr=grouped_Greece_date[cols].cumsum(axis=0)
gr=gr.join(date)
datewise=gr.groupby(["dateRep"]).agg({"cases":'sum',"deaths":'sum'})
datewise["Days Since"]=datewise.index-datewise.index.min()
datewise


# In[132]:


model_train=datewise.iloc[:int(datewise.shape[0]*0.95)]
valid6=datewise.iloc[int(datewise.shape[0]*0.95):]
y_pred6=valid6.copy()


# In[133]:


model_sarima= auto_arima(model_train["cases"],trace=True, error_action='ignore', 
                         start_p=0,start_q=0,max_p=2,max_q=2,m=7,
                   suppress_warnings=True,stepwise=True,seasonal=True)
model_sarima.fit(model_train["cases"])


# In[134]:


prediction_sarima=model_sarima.predict(len(valid6))
y_pred6["SARIMA Model Prediction"]=prediction_sarima


# In[135]:


print("Root Mean Square Error for SARIMA Model: ",np.sqrt(mean_squared_error(y_pred6["cases"],y_pred6["SARIMA Model Prediction"])))


# In[136]:


fig=go.Figure()
fig.add_trace(go.Scatter(x=model_train.index, y=model_train["cases"],
                    mode='lines+markers',name="Train Data for Confirmed Cases"))
fig.add_trace(go.Scatter(x=valid6.index, y=valid6["cases"],
                    mode='lines+markers',name="Validation Data for Confirmed Cases",))
fig.add_trace(go.Scatter(x=valid6.index, y=y_pred6["SARIMA Model Prediction"],
                    mode='lines+markers',name="Prediction for Confirmed Cases",))
fig.update_layout(title="Confirmed Cases SARIMA Model Prediction",
                 xaxis_title="Days since start",yaxis_title="Confirmed Cases",legend=dict(x=0,y=1,traceorder="normal"))
fig.show()


# In[137]:


prophet_c=Prophet(interval_width=0.95,weekly_seasonality=True,)
prophet_confirmed=pd.DataFrame(zip(list(datewise.index),list(datewise["cases"])),columns=['ds','y'])
prophet_confirmed


# In[138]:


prophet_c.fit(prophet_confirmed)


# In[139]:


forecast_c=prophet_c.make_future_dataframe(periods=36)
forecast_confirmed=forecast_c.copy()


# In[140]:


confirmed_forecast=prophet_c.predict(forecast_c)
print(confirmed_forecast[['ds','yhat', 'yhat_lower', 'yhat_upper']])


# In[141]:


print("Root Mean Squared Error for Prophet Model: ",np.sqrt(mean_squared_error(datewise["cases"],confirmed_forecast['yhat'].head(datewise.shape[0]))))


# In[142]:


prophet_c.plot(confirmed_forecast)


# In[143]:


print(prophet_c.plot_components(confirmed_forecast))


# In[144]:



datewise=grouped_Greece.groupby(["dateRep"]).agg({"cases":'sum',"deaths":'sum'})
datewise["Days Since"]=datewise.index-datewise.index.min()
datewise


# In[145]:


model_train=datewise.iloc[:int(datewise.shape[0]*0.95)]
valid6=datewise.iloc[int(datewise.shape[0]*0.95):]
y_pred6=valid6.copy()


# In[146]:


model_sarima= auto_arima(model_train["cases"],trace=True, error_action='ignore', 
                         start_p=0,start_q=0,max_p=2,max_q=2,m=7,
                   suppress_warnings=True,stepwise=True,seasonal=True)
model_sarima.fit(model_train["cases"])


# In[147]:


prediction_sarima=model_sarima.predict(len(valid6))
y_pred6["SARIMA Model Prediction"]=prediction_sarima


# In[148]:


print("Root Mean Square Error for SARIMA Model: ",np.sqrt(mean_squared_error(y_pred6["cases"],y_pred6["SARIMA Model Prediction"])))


# In[149]:


fig=go.Figure()
fig.add_trace(go.Scatter(x=model_train.index, y=model_train["cases"],
                    mode='lines+markers',name="Train Data for Confirmed Cases"))
fig.add_trace(go.Scatter(x=valid6.index, y=valid6["cases"],
                    mode='lines+markers',name="Validation Data for Confirmed Cases",))
fig.add_trace(go.Scatter(x=valid6.index, y=y_pred6["SARIMA Model Prediction"],
                    mode='lines+markers',name="Prediction for Confirmed Cases",))
fig.update_layout(title="Confirmed Cases SARIMA Model Prediction",
                 xaxis_title="Days since start",yaxis_title="Confirmed Cases",legend=dict(x=0,y=1,traceorder="normal"))
fig.show()


# In[150]:


prophet_c=Prophet(interval_width=0.95,weekly_seasonality=True,)
prophet_confirmed=pd.DataFrame(zip(list(datewise.index),list(datewise["cases"])),columns=['ds','y'])
prophet_confirmed


# In[151]:


prophet_c.fit(prophet_confirmed)


# In[152]:


forecast_c=prophet_c.make_future_dataframe(periods=17)
forecast_confirmed=forecast_c.copy()


# In[153]:


confirmed_forecast=prophet_c.predict(forecast_c)
print(confirmed_forecast[['ds','yhat', 'yhat_lower', 'yhat_upper']])


# In[154]:


print("Root Mean Squared Error for Prophet Model: ",np.sqrt(mean_squared_error(datewise["cases"],confirmed_forecast['yhat'].head(datewise.shape[0]))))


# In[155]:


prophet_c.plot(confirmed_forecast)
prophet_model=prophet_c


# In[156]:


print(prophet_c.plot_components(confirmed_forecast))


# In[3]:







