import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyodbc as db
import streamlit as st

st.set_page_config(page_title="QAQC-DUPLICATE", 
                   page_icon="📈",
                   layout='wide',
                   initial_sidebar_state="expanded"
)

# Set Streamlit app to full screen
st.write("""
    <style>
        body {
            margin: 0;
            padding: 0;
        }
        .fullScreenFrame {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            overflow: hidden;
            z-index: -1;
        }
    </style>
    <div class="fullScreenFrame"></div>
""", unsafe_allow_html=True)

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
data_list = run_query("""SELECT  [IUP_COMPANY],[PROJECT],[SITE_ID],[ORI_SAMPLE_ID]
        ,[DUP_SAMPLE_ID],[QC_TYPE],[ORI_SAMPLE_TYPE],[ORI_SAMPLE_METHOD],[LAB_ID],[DESPATCH_ID]
        ,[SEND_DATE],[LAB_JOB_NO],[RECEIPT_DATE],[LAB_DATE],[ELEMENT]
        ,[ORI],[DUP],[REMARKS]
        FROM [DWH_WETAR].[dbo].[LINK_QAQC_FIELD_CHECK_DASH_SAK]
        WHERE ORI IS NOT NULL AND DUP IS NOT NULL AND ELEMENT IN ('Fe_PCT','Pb_PPM','AG_PPM','Au_PPM','CU_PPM','S_PCT','SCIS_PCT','Zn_PPM')
        ORDER BY SEND_DATE""")

# Access the data inside the list of tuple
row_data = []
for row in data_list:
    values = list(row)  # Convert the tuple to a list of values
    row_data.append(values)  # Append the list of values for the row to the row_data list

# Define column names
col_name = [ 'IUP_COMPANY','PROJECT','SITE_ID', 'ORI_SAMPLE_ID'
        ,'DUP_SAMPLE_ID','QC_TYPE','ORI_SAMPLE_TYPE','ORI_SAMPLE_METHOD','LAB_ID','DESPATCH_ID'
        ,'SEND_DATE','LAB_JOB_NO','RECEIPT_DATE','LAB_DATE','ELEMENT'
        ,'ORI','DUP','REMARKS']

# Convert to dataframe
df=pd.DataFrame(row_data, columns=col_name)

# Convert datetime column to python datetime type
date_columns = ['SEND_DATE', 'LAB_DATE', 'RECEIPT_DATE']
df[date_columns] = df[date_columns].apply(pd.to_datetime, format='%Y-%m-%d %H:%M:%S')

# Convert ORI and DUP to float
float_col = ['ORI','DUP']
df[float_col] = df[float_col].astype(float)

# Extract year and month from lab date
df['YEAR'] = df['LAB_DATE'].dt.year
df['MONTH'] = df['LAB_DATE'].dt.month

# Calculate Coefficient of Variance (CV) from ORI and DUP
df['MEAN'] = df[['ORI','DUP']].mean(axis=1)
df['SD'] = df[['ORI','DUP']].std(axis=1)
df['CV'] = df['SD']/df['MEAN']*100

# Calculate RPD
df['RPD'] = (df['DUP']-df['ORI'])/df['MEAN']
df['ABS_RPD'] = df['RPD'].abs()
df['ABS_RPD_PCT'] = df['ABS_RPD']*100

# Sort the RPD percent ascending
df_abs_rpd = df.copy()
df_abs_rpd.sort_values(by='ABS_RPD',inplace=True)
# Calculate Rank of the ABS RPD
total_data = df_abs_rpd['ABS_RPD'].count()
df_abs_rpd['smpl_pct'] = (1/total_data)*100
df_abs_rpd['RANK'] = df_abs_rpd['smpl_pct'].cumsum()
df_abs_rpd_r = df_abs_rpd.reset_index(drop=True)

# Streamlit title
st.title("QAQC Duplicate Dashboard")

# Sidebar Filters
with st.sidebar:
    st.sidebar.subheader('Data FIlter')
    select_iup = st.sidebar.multiselect("Select IUP Company", df['IUP_COMPANY'].unique())
    select_qc = st.sidebar.multiselect("Select QC Type", df['QC_TYPE'].unique())
    select_ori = st.sidebar.multiselect("Select Original Sample Type", df['ORI_SAMPLE_TYPE'].unique())
    select_elm = st.sidebar.multiselect("Select Element", df['ELEMENT'].unique())
    Smin_date, Smax_date = st.sidebar.date_input("Select Lab Date", (df['LAB_DATE'].min(), df['LAB_DATE'].max()))

# Convert min and max dates to datetime objects
Smin_date = pd.to_datetime(Smin_date)
Smax_date = pd.to_datetime(Smax_date)

# Apply filters
filt =  (df['IUP_COMPANY'].isin(select_iup)) &\
        (df['QC_TYPE'].isin(select_qc)) &\
        (df['ORI_SAMPLE_TYPE'].isin(select_ori)) &\
        (df['ELEMENT'].isin(select_elm)) &\
        (df['LAB_DATE'] >= Smin_date) & \
        (df['LAB_DATE'] <= Smax_date)
fil_df = df[filt].reset_index(drop=True)  

# Sort the RPD percent ascending
df_abs_rpd = fil_df.copy()
df_abs_rpd.sort_values(by='ABS_RPD',inplace=True)
# Calculate Rank of the ABS RPD
total_data = df_abs_rpd['ABS_RPD'].count()
df_abs_rpd['smpl_pct'] = (1/total_data)*100
df_abs_rpd['RANK'] = df_abs_rpd['smpl_pct'].cumsum()
df_abs_rpd_r = df_abs_rpd.reset_index(drop=True)

# Display datasets
st.write("Filtered Datasets:")
st.write(fil_df)

col1, col2, col3 = st.columns(3)

# RPD% Plot
with col1:
    group_1 = df_abs_rpd_r.groupby(['QC_TYPE','ORI_SAMPLE_TYPE','ELEMENT'])
    for group_name, group_data in group_1:
        plt.figure(figsize=(30, 14))
        plt.gca().set_facecolor('white')

        plt.scatter(group_data['RANK'], group_data['ABS_RPD_PCT'])
        # Set font size of tick labels
        plt.xticks(fontsize=24)  
        plt.yticks(fontsize=24)  


        plt.axhline(y=20, color='red', linestyle='-', linewidth=2)
        plt.axvline(x=80, color='red', linestyle='-', linewidth=2)

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
        title_text = f'QC TYPE: {group_name[0]}\n'
        title_text += f'ORI SAMPLE TYPE: {group_name[1]}\n'
        title_text += f'ELEMENT: {group_name[2]}\n'

        plt.suptitle(f'RPD PLOT', fontweight='bold', fontsize='29', x=0.515)
        plt.title(title_text, y=0.97, fontsize=22)

        plt.xlabel('RANK', fontsize=25, fontweight='bold')
        plt.ylabel('RPD%', fontsize=25, fontweight='bold')
        plt.grid(axis='y', linestyle='--', color='gray', linewidth=1.2)

        # Display in streamlit
        st.markdown(f'<h3 style="font-size: 14px;">RELATIVE % DIFFERENCE PLOT - {group_name[0]} - Ori Sample Type: {group_name[1]}, {group_name[2]}</h3>', unsafe_allow_html=True)
        st.pyplot(plt)
        plt.clf()  # Clear plot to avoid overlapping legends

# RPD% Plot
with col2:
    group_2 = fil_df.groupby(['QC_TYPE','ORI_SAMPLE_TYPE','ELEMENT'])
    for group_name, group_data in group_2:

        plt.figure(figsize=(30, 14))
        plt.gca().set_facecolor('white')

        plt.scatter(group_data['MEAN'], group_data['RPD'])
        # Set font size of tick labels
        plt.xticks(fontsize=24)  
        plt.yticks(fontsize=24) 

        plt.axhline(y=0, color='red', linestyle='-', linewidth=2)

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
        title_text = f'QC TYPE: {group_name[0]}\n'
        title_text += f'ORI SAMPLE TYPE: {group_name[1]}\n'
        title_text += f'ELEMENT: {group_name[2]}\n'

        plt.suptitle(f'RPD %DIFFERENCE PLOT', fontweight='bold', fontsize='29', x=0.515)
        plt.title(title_text, y=0.97, fontsize=22)
        plt.xlabel(f'{group_name[2]}', fontsize=25, fontweight='bold')
        plt.ylabel('% Difference', fontsize=25, fontweight='bold')
        plt.grid(axis='y', linestyle='--', color='gray', linewidth=1.2)

        # Display in streamlit
        st.markdown(f'<h3 style="font-size: 14px;">RPD %DIFFERENCE PLOT - {group_name[0]} - Ori Sample Type: {group_name[1]}, Element: {group_name[2]}</h3>', unsafe_allow_html=True)
        st.pyplot(plt)
        plt.clf()  # Clear plot to avoid overlapping legends

# CoV plot
with col3:
    group_3 = fil_df.groupby(['QC_TYPE','ORI_SAMPLE_TYPE','ELEMENT'])
    for group_name, group_data in group_3:

        plt.figure(figsize=(30, 14))
        plt.gca().set_facecolor('white')

        # Calculate moving average
        group_data['MA5'] = group_data['MEAN'].rolling(window=5, min_periods=1).mean()
        group_data.sort_values(by='MA5', inplace=True) # Sort MA data
    
        plt.plot(group_data['MA5'], group_data['CV'])
        # Set font size of tick labels
        plt.xticks(fontsize=24)  
        plt.yticks(fontsize=24) 

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
        title_text = f'QC TYPE: {group_name[0]}\n'
        title_text += f'ORI SAMPLE TYPE: {group_name[1]}\n'
        title_text += f'ELEMENT: {group_name[2]}\n'
        
        plt.suptitle(f'COEFFICIENT OF VARIANCE PLOT', fontweight='bold', fontsize='29', x=0.515)
        plt.title(title_text, y=0.97, fontsize=22)
        plt.xlabel(f'{group_name[2]}', fontsize=25, fontweight='bold')
        plt.ylabel('Coefficient of Variance', fontsize=25, fontweight='bold')
        plt.grid(axis='y', linestyle='--', color='gray', linewidth=1.2)

        # Display in streamlit
        st.markdown(f'<h3 style="font-size: 14px;">COEFFICIENT OF VARIANCE PLOT - {group_name[0]} - Ori Sample Type: {group_name[1]}, Element: {group_name[2]}</h3>', unsafe_allow_html=True)
        st.pyplot(plt)
        plt.clf()  # Clear plot to avoid overlapping legends
