import streamlit as st

import numpy as np

import pandas as pd

import pandas_profiling





from streamlit_pandas_profiling import st_profile_report

from pycaret.regression import setup as setup_reg

from pycaret.regression import compare_models as compare_models_reg

from pycaret.regression import save_model as save_model_reg

from pycaret.regression import plot_model as plot_model_reg






from pycaret.classification import setup as setup_class

from pycaret.classification import compare_models as compare_models_class

from pycaret.classification import save_model as save_model_class

from pycaret.classification import plot_model as plot_model_class

import mlflow





import matplotlib.pyplot as plt

#from tkinter import filedialog

#from tkinter import *

import seaborn as sns

from sklearn.model_selection import train_test_split,KFold , cross_val_score,cross_validate

from sklearn.tree import DecisionTreeClassifier ,plot_tree,ExtraTreeClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.metrics import accuracy_score,precision_score ,classification_report,RocCurveDisplay , auc

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix

from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import StandardScaler , MinMaxScaler

from sklearn.metrics import recall_score,fbeta_score, make_scorer,roc_curve ,roc_auc_score

from sklearn.decomposition import PCA

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import BaggingClassifier,AdaBoostClassifier

from sklearn.svm import SVC

import time

from sklearn.pipeline import Pipeline

from sklearn.feature_selection import SelectFromModel

from sklearn.preprocessing import StandardScaler

import pickle

from sklearn.ensemble import RandomForestClassifier

import plotly.express as px

from sklearn.model_selection import cross_val_predict

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from pycaret.regression import setup, compare_models, blend_models, finalize_model, predict_model, plot_model

from pycaret.classification import *

from scipy.stats import kstest, expon

from scipy.stats import chisquare, poisson

from fitter import Fitter, get_common_distributions, get_distributions

from sklearn.decomposition import PCA

from scipy.stats import expon, poisson, gamma, lognorm, weibull_min, kstest,norm

import scipy

import scipy.stats

from mlxtend.plotting import plot_pca_correlation_graph

import scipy.stats as stats

from sklearn.ensemble import GradientBoostingClassifier

#import pickle ### on utilise ce bibliotheque pour sauvegarder notre modél , qui nous servira pour la partie deployement .

#url="https://www.linkedin.com/in/alioune-gaye-1a5161172/"

### faire un catching (garder en memoire tout ce qui est deja calculer)

@st.cache

def load_data(file):

    data=pd.read_csv(file)

    return data

### la fonction principale ...




def main():
    
    ## l'entéte de mon code
    st.title('Alioune Gaye : mon appplication AutoML')
    #st.sidebar.write("[Author : Gaye Alioune](%)" % url)
    st.sidebar.markdown(":green[This wep app is a No-code tool for Exploratory Data Analysis and building Machine Learning model for R. ] \n")
    #"**This wep app is a No-code tool for Exploratory Data Analysis and building Machine Learning model for R**"
        #"1.Load your dataset file (CSV file);\n"
        #"2.Click on *profile Dataset* button in order to generate the pandas profiling of the dataset;\n"
        #"3. Choose your target column ;\n"
        #"4.Choose the machine learning task (Regression or Classification);\n"
        #"5.Click on *Run Modeling * in order to start the training process.\n"
        #"When the model is built , you can view the results like the pipeline model , Residuals plot , Roc Curve, confusion Matrix ..."
        #"\n6. Download the Pipeline model in your local computer.")

    
    ## Charger le jeu de donnée
    st.write('faire un choix') 
    choix1=st.selectbox('Select  polluant_homogene or polluant_heterogene', ["polluant_homogene","polluant_heterogene"]) 
    file=st.file_uploader("Upload your dataset in csv format", type=["csv"])
    
    if file is not None: ## pour dire à l'utilisateur , si le fichier importer n'est pas nul alors fait ceci

        data=pd.read_csv(file)

        st.dataframe(data.head()) ## afficher les données importer

        #data=data.dropna(subset=target) ### suprimer les valeurs manquantes

        ## analyse exploiratoire du jeu de données

        ## creation d'un bouton de visualisation des données et graphe

        profile=st.button('profile dataset')

        if profile:

            profile_df=data.profile_report()

            st_profile_report(profile_df) ### afficher le profile

            ## Phase de modelisation

            ## Choix des targets

        target=st.selectbox('Select the target variable',data.columns)

        ## selection du type de model (classification ou regression)

        task=st.selectbox('Select a ML task', ["Classification","Regression"])

        ## Maintenant on peut commencer par ecrir le code

        ##Pour la regression et la classification

        
        if task=="Regression":
            if st.button("Run Modelling"):
                exo_reg= setup_reg(data,target=target)
                ## entrainer plusieurs models à la fois
                model_reg=compare_models_reg()
                ### sauvegarder le model
                save_model_reg(model_reg,"best_reg_model")
                ### Message de succé si tout ce passe bien
                st.success("Regression model built successfully")
                ## Results
                ### les residus
                st.write("Residuals")
                plot_model_reg(model_reg,plot='residuals',save=True)
                st.image("Residuals.png") ### Sauvegarder le resultat
                ### Les erreurs 
                #plot_model_reg(model_reg,plot='error',save=True)
                #st.image("erurs.png")  ### Sauvegarder le resultat
                ### Variables importantes
                st.write("Feature importance")
                plot_model_reg(model_reg,plot='feature',save=True)
                st.image("Feature Importance.png")
                ### Telecharger le pipeline
                with open('best_reg_model.pkl','rb') as f:
                    st.download_button('Download Pipeline Model',f,file_name="best_reg_model.pkl")
                    

        
        if task=="Classification":
            if st.button("Run Modelling"):
                if choix1=="polluant_homogene":
                    exo_class= setup_class(data,target=target,index=False,train_size =0.80,normalize = True,normalize_method = 'zscore',remove_multicollinearity = True,log_experiment=True, experiment_name="polluant-homogene"
                     ,pca =False, pca_method =None,pca_components =None)
                    
                if choix1=="polluant_heterogene": 
                    exo_class= setup_class(data,target=target,index=False,train_size =0.80,normalize = False,multicollinearity_threshold =0.8,normalize_method = 'zscore',remove_multicollinearity = True,log_experiment=True, experiment_name="polluant-heterogene"
                     ,pca =False, pca_method =None,pca_components =None)

                st.write('les caracteristiques de notre setup')
                ## entrainer plusieurs models à la fois
                model_class=compare_models_class()
                tuned_model_class = tune_model(model_class)
                st.write('Votre meilleur model de classification est ', model_class)
                ### sauvegarder le model une fois qu'on es satisfait du model
                final_model1 = finalize_model(model_class)  ### notre pipeline(entrainement du model sur tout les donnée)
                if choix1=="polluant_homogene":
                    save_model_class(final_model1,"best_class_model")
                    st.write("notre pipeline",save_model_class(model_class,"best_class_model"))
                    ### Message de succé si tout ce passe bien
                    st.write('Les metrics')
                    st.dataframe(pull(), height=200)
                    st.success("Classification model built successfully")
                if choix1=="polluant_heterogene":
                    save_model_class(final_model1,"best_class_model1")
                    st.write("notre pipeline",save_model_class(model_class,"best_class_model1"))
                    ### Message de succé si tout ce passe bien
                    st.write('Les metrics')
                    st.dataframe(pull(), height=200)
                    st.success("Classification model built successfully")     
                ## ResuLts
                col5, col6,col7,col8=st.columns(4)
                with col5:
                    st.write("ROC curve")
                    plot_model_class(model_class,save=True)
                    st.image("AUC.png")

 
                #with col6:

                    #plot_model_class(tuned_model_class,plot='class_report',display_format='streamlit')
                    #st.image("Class_repport.png")

 
                with col7:

                    st.write("Confusion Matrix")

                    plot_model_class(model_class,plot='confusion_matrix',save=True)
                    st.image("Confusion Matrix.png")


                with col8:

                    st.write("Feature Importance")

                    plot_model_class(model_class,plot='feature',save=True)

                    st.image("Feature Importance.png")

 

                col9,col10 =st.columns(2)

                #with col9:
                    #st.write("Boundary")
                    #plot_model_class(tuned_model_class,plot='boundary',display_format='streamlit',save=True)
                    #st.image("Boundary.png")


                
                ###prediction avec les données de test
                st.write("La prediction du model avec les données de test")    
                prediction=predict_model(final_model1)
                st.dataframe(prediction,height=200)

 

                ## Download the pipeline model
                if choix1=="polluant_homogene":
                    with open('best_class_model.pkl','rb') as f:
                        st.download_button('Download Pipeline Model',f,file_name="best_class_model.pkl")
                if choix1=="polluant_heterogene": 
                    with open('best_class_model1.pkl','rb') as f:
                        st.download_button('Download Pipeline Model',f,file_name="best_class_model1.pkl")
    else:

        st.image("https://cdn.futura-sciences.com/cdn-cgi/image/width=1280,quality=60,format=auto/sources/images/data_science_1.jpg")                    

    ### deploiement de notre model machine learning .

    # Prediction via pipeline      

    file_1=st.file_uploader("Upload your dataset à predir  in csv format", type=["csv"])    
    
    if file_1 is not None: ## pour dire à l'utilisateur , si le fichier importer n'est pas nul alors fait ceci

            data1=pd.read_csv(file_1)

            n=len(data1.columns)

            data2=data1

            st.write('les données que vous voulez predire est:',data2)  

            #logged_model = 'runs:/42ae053461bc4e4c9cd8faded887aeaa/model' ### chemin de mon meilleur model qui se trouve dans le dosier "mlruns"

            # Load model as a PyFuncModel.

            #loaded_model = mlflow.pyfunc.load_model(logged_model)

            #st.write('le model enregidtrer sur mlflow est',loaded_model)

 
            if choix1=="polluant_homogene":
                   loaded_model=load_model('best_class_model')
            if choix1=="polluant_heterogene":
                   loaded_model=load_model('best_class_model1')
                

            ### Affichage des resultats de la predition

            ### data2  est les données ----- à predir dans le future .............

            if st.button("Run prediction"):    

               prediction=predict_model(loaded_model,data=data2)

               #### On importe les données à predire dans le future , avec le codde deploié sur mlflow ........ Donc , on essaye de toujours  faire l'exploitationn des données aavent de les importer  dans le code

               st.write('la prediction de votre jeux de donner est:')

               st.dataframe(prediction.iloc[:,[len(prediction.columns)-2]],height=200)
               @st.experimental_memo
               def convert_df(df):
                   return df.to_csv(index=False).encode('utf-8')
               csv = convert_df(prediction.iloc[:,[len(prediction.columns)-2]])
               st.download_button(label="Download votre prediction",data=csv ,file_name='Votre prediction',key='download-csv')
                            
    else:

        st.image("https://cdn.futura-sciences.com/cdn-cgi/image/width=1280,quality=60,format=auto/sources/images/data_science_1.jpg")    
    #st.markdown(":blue[La partie Quatification. ] \n")    
    #st.markdown(La partie Quatification, unsafe_allow_html=False, *, help=None)     
    st.title(':blue[La partie Quatification. ] ')
    quant=st.selectbox('Selectionner la methode de quatification utilisée: ', ["quatif_un_pol","quatif_double_pol"] 
    #if quant=='quatif_un_pol':
         #### on met le code pour la partie quantification d'un polluant puis on affichedera les courbes obtenues , la valeurs de la concentration du polluant obtenue 
         ## la fonction d'ajustement , y_intercept , x_intercept , delta_x ,    
                       
                       
                       
    
if __name__=='__main__':
    main()
                                 

 

   

 

 
