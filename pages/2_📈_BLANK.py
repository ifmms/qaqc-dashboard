import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyodbc as db
import streamlit as st


# Set page title
st.set_page_config(
    page_title="QAQC-BLANK",
    page_icon="📈",
    layout='wide',
    initial_sidebar_state="expanded"
)

# Initialize connection
# Use st.cache_resource to only run once
@st.cache_resource
def init_connection():
    return db.connect(
        "DRIVER={ODBC Driver 17 for SQL Server};SERVER="
        + st.secrets["server"]
        + ";DATABASE="
        + st.secrets["database"]
        + ";UID="
        + st.secrets["username"]
        + ";PWD="
        + st.secrets["password"]
    )
conn = init_connection()

# Use st.cache_data to only rerun when the query changes or after 30 min
@st.cache_data
def run_query(query):
    with conn.cursor() as cur:
        cur.execute(query)
        return cur.fetchall()
    
    # Get data    
data_list = run_query("""SELECT  [IUP_COMPANY]
            ,[STANDARD_ID]
            ,[SAMPLE_TAG]
            ,[QC_SOURCE]
            ,[DESPATCH_ID]
            ,[SEND_DATE]
            ,[LAB_ID]
            ,[LAB_JOB_NO]
            ,[LAB_DATE]
            ,[RECEIPT_DATE]
            ,[ELEMENT]
            ,[LAB_ELEMENT]
            ,[LAB_METHOD]
            ,[GENERIC_METHOD]
            ,[RESULT]
            ,[DETECTION_LIMIT]
            ,[REMARKS_DL]
            ,[DETECTION_LIMIT_3]
            ,[REMARKS_3DL]
            FROM [DWH_WETAR].[dbo].[LINK_QAQC_BLANK_DASH_SAK]
              WHERE ELEMENT IN ('Cu', 'Au', 'Ag', 'Pb', 'Zn', 'S', 'Fe', 'SCIS')
              ORDER BY SEND_DATE
              """)

# Access the data inside the list of tuple
row_data = []
for row in data_list:
    values = list(row)  # Convert the tuple to a list of values
    row_data.append(values)  # Append the list of values for the row to the row_data list


# Define column names
col_name = ['IUP_COMPANY', 'STANDARD_ID', 'SAMPLE_TAG', 'QC_SOURCE', 'DESPATCH_ID',
            'SEND_DATE', 'LAB_ID', 'LAB_JOB_NO', 'LAB_DATE', 'RECEIPT_DATE',
            'ELEMENT', 'LAB_ELEMENT', 'LAB_METHOD', 'GENERIC_METHOD', 'RESULT',
            'DETECTION_LIMIT', 'REMARKS_DL','DETECTION_LIMIT_3', 'REMARKS_3DL' 
            ]

# Convert to dataframe
df=pd.DataFrame(row_data, columns=col_name)

# Convert datetime column to python datetime type
date_columns = ['SEND_DATE', 'LAB_DATE', 'RECEIPT_DATE']
df[date_columns] = df[date_columns].apply(pd.to_datetime, format='%Y-%m-%d %H:%M:%S')

# Extract year and month from lab date
df['YEAR'] = df['LAB_DATE'].dt.year
df['MONTH'] = df['LAB_DATE'].dt.month

# Streamlit Title
st.title("QAQC Blank Dashboard")

# Sidebar Filters
with st.sidebar:
    st.sidebar.subheader('Data Filter')
    select_iup = st.sidebar.multiselect("Select IUP Company", df['IUP_COMPANY'].unique())
    select_lab = st.sidebar.multiselect("Select LAB ID", df['LAB_ID'].unique())
    select_elm = st.sidebar.multiselect("Select Element", df['ELEMENT'].unique())
    select_mtd = st.sidebar.multiselect("Select Generic Method", df['GENERIC_METHOD'].unique())
    select_lbm = st.sidebar.multiselect("Select Lab Method", df['LAB_METHOD'].unique())
    Smin_date, Smax_date = st.sidebar.date_input("Select Lab Date", (df['LAB_DATE'].min(), df['LAB_DATE'].max()))

# Convert min and max dates to datetime objects
Smin_date = pd.to_datetime(Smin_date)
Smax_date = pd.to_datetime(Smax_date)

# Apply Filters
filt = (df['IUP_COMPANY'].isin(select_iup)) &\
        (df['LAB_ID'].isin(select_lab)) & \
       (df['ELEMENT'].isin(select_elm)) & \
       (df['GENERIC_METHOD'].isin(select_mtd)) & \
       (df['LAB_METHOD'].isin(select_lbm)) & \
       (df['LAB_DATE'] >= Smin_date) & \
       (df['LAB_DATE'] <= Smax_date)
fil_df = df[filt].reset_index(drop=True)

# Display the filtered DataFrames as needed
st.write("Filtered LAB_DATE Dataset:")
st.write(fil_df)

# Plotting
grouped = fil_df.groupby(['IUP_COMPANY', 'ELEMENT', 'GENERIC_METHOD', 'LAB_METHOD'], as_index=False)
for group_name, group_data in grouped:
    plt.figure(figsize=(30, 14.5))
    plt.gca().set_facecolor('white')

    # Plot result
    plt.plot(group_data['SAMPLE_TAG'], group_data['RESULT'], marker='o', linestyle='--', color='blue', label='RESULT')


    # Plot horizontal line for specified values
    lines = ['DETECTION_LIMIT', 'DETECTION_LIMIT_3']
    colors = ['orange', 'red']
    for line, color in zip(lines, colors):
        plt.axhline(y=group_data[line].iloc[0], color=color, linestyle='-', label=line)
    
    # Annotate for sample that out of the 2SD
    for i, remarks in enumerate(group_data['REMARKS_DL']):
        if remarks == 'out':
            plt.text(group_data['SAMPLE_TAG'].iloc[i], group_data['RESULT'].iloc[i], group_data['SAMPLE_TAG'].iloc[i], color='purple', ha='center', fontsize=12)

    # Calculate 5-period moving average
    group_data['MA5'] = group_data['RESULT'].rolling(window=5, min_periods=1).mean()
    # Plot 5-period moving average
    plt.plot(group_data['SAMPLE_TAG'], group_data['MA5'], linestyle='-', color='green', label='5-Period Moving Average')


    # Set graph borderline
    plt.gca().spines['top'].set_linewidth(0.8)
    plt.gca().spines['top'].set_color('gray')
    plt.gca().spines['right'].set_linewidth(0.8)
    plt.gca().spines['right'].set_color('gray')
    plt.gca().spines['bottom'].set_linewidth(0.8)
    plt.gca().spines['bottom'].set_color('gray')
    plt.gca().spines['left'].set_linewidth(0.8)
    plt.gca().spines['left'].set_color('gray')


    # Sub header for data that out of 2SD
    out_samples = group_data[group_data['REMARKS_DL'].str.lower() == 'out'] 
    title_text = f'{group_name[2]}-{group_name[3]}\n'
    title_text += f'Total Samples: {len(group_data)}\n'
    title_text += f'Samples OUT DL: {len(out_samples)}\n'

    # Customize the axis
    plt.suptitle(f'BLANK-{group_name[1]}', fontweight='bold', fontsize='24', x=0.51)
    plt.title(title_text, y=1)
    plt.xlabel('SAMPLE_TAG', fontsize=20, fontweight='bold')
    plt.ylabel('VALUE (ppm)', fontsize=20, fontweight='bold')
    plt.xticks(rotation='vertical', fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid(axis='y', linestyle='--', color='gray', linewidth=1.2)
    plt.legend(bbox_to_anchor=(0.5, -0.16), loc='lower center', ncol=9, facecolor='white', prop={'size': 15, 'weight': 'bold'})
    
    # Display in streamlit
    st.markdown(f'<h3 style="font-size: 14px;">{group_name[0]} - Element: {group_name[1]}, Method: {group_name[2]} - {group_name[3]}</h3>', unsafe_allow_html=True)
    st.pyplot(plt)
    plt.clf()  # Clear plot to avoid overlapping legends
