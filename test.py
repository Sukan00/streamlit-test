import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go

### Page setup ###
st.set_page_config(page_title="Analysis Dashboard", page_icon=":bar_chart:", layout="wide")
st.markdown('<style>div.block-container{padding-top:1.5rem;}</style>',unsafe_allow_html=True) #ปรับ top padding
st.title("Data analysis Dashbord")
# End Page setup #


df = pd.read_csv('streamlit-test\E-Commerce data.csv', dtype={'CustomerID': str,'InvoiceID': str})
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
canceled_products = df[df['InvoiceNo'].str.contains('C', na=False)]

with st.expander("Data Preview"):
    st.markdown(f"Number of data: {len(df):,}")
    st.markdown(f"Number of products that were canceled: {len(canceled_products)}")
    st.dataframe(df)

st.markdown(':rainbow[The cleansed data that was retrieved.]')

variables = '''This dataframe contains 8 variables that correspond to:  
**InvoiceNo**: Invoice number. Nominal, a 6-digit integral number uniquely assigned to each transaction. If this code starts with letter 'c', it indicates a cancellation.  
**StockCode**: Product (item) code. Nominal, a 5-digit integral number uniquely assigned to each distinct product.  
**Description**: Product (item) name. Nominal.  
**Quantity**: The quantities of each product (item) per transaction. Numeric.  
**InvoiceDate**: Invice Date and time. Numeric, the day and time when each transaction was generated.  
**UnitPrice**: Unit price. Numeric, Product price per unit in sterling.  
**CustomerID**: Customer number. Nominal, a 5-digit integral number uniquely assigned to each customer.  
**Country**: Country name. Nominal, the name of the country where each customer resides.
'''
st.markdown(variables)

### Cleaning data ###
df = df.dropna()
df = df.drop_duplicates()
df = df[(df['Quantity'] >= 0) & (df['UnitPrice'] >= 0)]

a, b = st.columns(2)
with a:
    with st.expander("Cleaned data"):
        st.markdown(f"*Number of orders by members: {len(df):,}*")
        st.write(df)

    csv = df.to_csv().encode("utf-8")
    st.download_button(
        label="Download cleaned data as CSV",
        data=csv,
        file_name="Cleaned E-Commerce data.csv",
        mime="text/csv",
    )
    df_summary = pd.DataFrame([{'products': len(df['StockCode'].value_counts()),    
               'transactions': len(df['InvoiceNo'].value_counts()),
               'customers': len(df['CustomerID'].value_counts()),
               'countries': len(df['Country'].value_counts()) 
              }], columns = ['products', 'transactions', 'customers', 'countries'], index = ['quantity'])
    st.dataframe(df_summary)
with b:
    temp = df[['CustomerID', 'InvoiceNo', 'Country']].groupby(['CustomerID', 'InvoiceNo', 'Country']).count()
    temp = temp.reset_index(drop = False)
    countries = temp['Country'].value_counts()

    # Define the data for the choropleth map
    data = dict(
        type='choropleth',
        locations=countries.index,
        locationmode='country names',
        z=countries,
        text=countries.index,
        colorbar={'title': 'Order nb.'},
        colorscale=[
            [0, 'rgb(224,255,255)'],
            [0.01, 'rgb(166,206,227)'],
            [0.02, 'rgb(31,120,180)'],
            [0.03, 'rgb(178,223,138)'],
            [0.05, 'rgb(51,160,44)'],
            [0.10, 'rgb(251,154,153)'],
            [0.20, 'rgb(255,255,0)'],
            [1, 'rgb(227,26,28)']
        ],
        reversescale=False
    )

    # Define the layout for the map
    layout = dict(
        title='Number of Orders per Country',
        geo=dict(showframe=True, projection={'type': 'mercator'})
    )

    # Create the figure using plotly.graph_objects
    choromap = go.Figure(data=[data], layout=layout)

    # Streamlit rendering
    st.title("Choropleth Map Example")
    st.plotly_chart(choromap)

# End Main page #