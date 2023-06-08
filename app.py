import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pycaret.regression import *

st.set_page_config( page_title = 'Predição climática',
                    layout = 'wide',
                    initial_sidebar_state = 'expanded')

st.title('Agro - climatologia')

with st.expander('Objetivo', expanded = False):
    var_test = 5
    st.write('O objetivo principal deste app é a predição da precificação de vegetais e frutas por kg')

with st.sidebar:
    c1, c2 = st.columns(2)
    c2.subheader('Auto ML - GS')

    database = st.checkbox("CSV")

    if database is True:
        st.info('Upload do CSV')
        file = st.file_uploader('Selecione o arquivo CSV', type='csv')

#Tela principal
if database is True:
    if file:
        #carregamento do CSV
        Xtest = pd.read_csv(file)

        #carregamento / instanciamento do modelo pkl
        mdl_lgbm = load_model('./model')

        #predict do modelo
        ypred = predict_model(mdl_lgbm, data = Xtest)

        with st.expander('Visualizar CSV carregado:', expanded = False):
            c1, _ = st.columns([2,4])
            qtd_linhas = c1.slider('Visualizar quantas linhas do CSV:', 
                                    min_value = 5, 
                                    max_value = Xtest.shape[0], 
                                    step = 10,
                                    value = 5)
            st.dataframe(Xtest.head(qtd_linhas))

        with st.expander('Visualizar Predições:', expanded = True):
            c1, _, c2, c3 = st.columns([2,.5,1,1])
            print(c1, c2, c3)
            treshold = c1.slider('Treshold (ponto de corte para considerar predição como True)',
                                min_value = 0.0,
                                max_value = 1.0,
                                step = .1,
                                value = .5)
            qtd_true = ypred.loc[ypred['prediction_label'] > treshold].shape[0]

            def color_pred(val):
                color = 'olive' if val > treshold else 'orangered'
                return f'background-color: {color}'

            tipo_view = st.radio('', ('Completo', 'Apenas predições'))
            if tipo_view == 'Completo':
                df_view = ypred.copy()
            else:
                df_view = pd.DataFrame(ypred.iloc[:,-1].copy())

            st.dataframe(df_view)

            csv = df_view.to_csv(sep = ';', decimal = ',', index = True)
            st.markdown(f'Shape do CSV a ser baixado: {df_view.shape}')
            st.download_button(label = 'Download CSV',
                            data = csv,
                            file_name = 'Predicoes.csv',
                            mime = 'text/csv')

    else:
        st.warning('Arquivo CSV não foi carregado')
        
else:
    st.success('Selecione a opção disponível para a disponibilização do conjunto de dados')