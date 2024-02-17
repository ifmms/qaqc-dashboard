import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyodbc as db
import streamlit as st
import subprocess


# Set page title
st.set_page_config(
    page_title="QAQC-STANDARD",
    page_icon="📈",
    layout='wide',
    initial_sidebar_state="expanded"
)

# Initialize connection
# Use st.cache_resource to only run once
@st.cache_resource
#def connect_to_vpn():
   # subprocess.run(["openvpn"]_
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
data_list = run_query("""SELECT [PROSPECT_GROUP],[STANDARD_ID],[SAMPLE_TAG],[QC_SOURCE],[DESPATCH_ID]
                      ,[SEND_DATE],[LAB_ID],[LAB_JOB_NO],[LAB_DATE],[RECEIPT_DATE]
                      ,[ELEMENT],[LAB_ELEMENT],[LAB_METHOD],[GENERIC_METHOD],[RESULT]
                      ,[NOMINATED_UNITS] ,[NOMINAL_VALUE] AS EXPECTED_VALUE,[NOMINAL_VALUE_UNITS]
                      ,[STD_DEVIATION],[SD_UPPER_1],[SD_UPPER_2],[SD_UPPER_3],[SD_LOWER_1],[SD_LOWER_2]
                      ,[SD_LOWER_3],[REMARKS_2SD],[SD_UPPER_5],[SD_LOWER_5]
                      ,[IUP_COMPANY],[REMARKS_5]
              FROM [DWH_WETAR].[dbo].[LINK_QAQC_CRM_BLANK_SHEWHART_DASH_SAK]
              WHERE ELEMENT IN ('Cu', 'Au', 'Ag', 'Pb', 'Zn', 'S', 'Fe', 'SCIS')
              ORDER BY SEND_DATE
              """)

# Access the data inside the list of tuple
row_data = []
for row in data_list:
    values = list(row)  # Convert the tuple to a list of values
    row_data.append(values)  # Append the list of values for the row to the row_data list


# Define column names
col_name = ['PROSPECT_GROUP', 'STANDARD_ID', 'SAMPLE_TAG', 'QC_SOURCE', 'DESPATCH_ID',
            'SEND_DATE', 'LAB_ID', 'LAB_JOB_NO', 'LAB_DATE', 'RECEIPT_DATE',
            'ELEMENT', 'LAB_ELEMENT', 'LAB_METHOD', 'GENERIC_METHOD', 'RESULT',
            'NOMINATED_UNITS', 'EXPECTED_VALUE', 'NOMINAL_VALUE_UNITS',
            'STD_DEVIATION', 'SD_UPPER_1', 'SD_UPPER_2', 'SD_UPPER_3', 'SD_LOWER_1', 'SD_LOWER_2',
            'SD_LOWER_3', 'REMARKS_2SD', 'SD_UPPER_5', 'SD_LOWER_5',
            'IUP_COMPANY', 'REMARKS_5']

# Convert to dataframe
df=pd.DataFrame(row_data, columns=col_name)

# Convert to uppercase in STANDARD ID Column
df['STANDARD_ID'] = df['STANDARD_ID'].str.upper()

# Convert datetime column
date_columns = ['SEND_DATE', 'LAB_DATE', 'RECEIPT_DATE']
df[date_columns] = df[date_columns].apply(pd.to_datetime, format='%Y-%m-%d %H:%M:%S')

# Streamlit Title
st.title("QAQC CRM Dashboard")

# Sidebar Filters
with st.sidebar:
    st.sidebar.subheader('Data Filter')
    select_pgr = st.sidebar.multiselect("Select Prospect Group", df['PROSPECT_GROUP'].unique())
    select_std = st.sidebar.multiselect("Select STANDARD ID", df['STANDARD_ID'].unique())
    select_elm = st.sidebar.multiselect("Select Element", df['ELEMENT'].unique())
    select_mtd = st.sidebar.multiselect("Select Generic Method", df['GENERIC_METHOD'].unique())
    select_lbm = st.sidebar.multiselect("Select Lab Method", df['LAB_METHOD'].unique())
    Smin_date, Smax_date = st.sidebar.date_input("Select Lab Date", (df['LAB_DATE'].min(), df['LAB_DATE'].max()))

# Convert min and max dates to datetime objects
Smin_date = pd.to_datetime(Smin_date)
Smax_date = pd.to_datetime(Smax_date)

# Apply Filters
filt = (df['PROSPECT_GROUP'].isin(select_pgr)) &\
        (df['STANDARD_ID'].isin(select_std)) & \
       (df['ELEMENT'].isin(select_elm)) & \
       (df['GENERIC_METHOD'].isin(select_mtd)) & \
       (df['LAB_METHOD'].isin(select_lbm)) & \
       (df['LAB_DATE'] >= Smin_date) & \
       (df['LAB_DATE'] <= Smax_date)
send_df = df[filt].reset_index(drop=True)

# Display the filtered DataFrames as needed
st.write("Filtered LAB_DATE Dataset:")
st.write(send_df)

# Plotting
grouped = send_df.groupby(['STANDARD_ID', 'ELEMENT', 'GENERIC_METHOD', 'LAB_METHOD'], as_index=False)
for group_name, group_data in grouped:
    plt.figure(figsize=(30, 14.5))
    plt.gca().set_facecolor('white')

    # Plot result
    plt.plot(group_data['SAMPLE_TAG'], group_data['RESULT'], marker='o', linestyle='--', color='blue', label='RESULT')


    # Plot horizontal line for specified values
    lines = ['EXPECTED_VALUE', 'SD_UPPER_2', 'SD_LOWER_2', 'SD_UPPER_3', 'SD_LOWER_3', 'SD_UPPER_5', 'SD_LOWER_5']
    colors = ['green', 'red', 'red', 'purple', 'purple', 'orange', 'orange']
    for line, color in zip(lines, colors):
        plt.axhline(y=group_data[line].iloc[0], color=color, linestyle='-', label=line)
    
    # Annotate for sample that out of the 2SD
    for i, remarks in enumerate(group_data['REMARKS_2SD']):
        if remarks == 'out':
            plt.text(group_data['SAMPLE_TAG'].iloc[i], group_data['RESULT'].iloc[i], group_data['SAMPLE_TAG'].iloc[i], color='purple', ha='center', fontsize=12)

    # Calculate 5-period moving average
    group_data['MA5'] = group_data['RESULT'].rolling(window=5, min_periods=1).mean()
    # Plot 5-period moving average
    plt.plot(group_data['SAMPLE_TAG'], group_data['MA5'], linestyle='-', color='green', label='5-Period Moving Average')

    # Calculate Z-Score
    z_score = np.mean((group_data['RESULT'] - group_data['EXPECTED_VALUE']) / group_data['STD_DEVIATION'])
    # Classify performance category based on Z-score
    if -0.2 <= z_score <= 0.2:
        performance_category = 'Excellent Performance'
    elif -0.4 <= z_score <= 0.4:
        performance_category = 'Good Performance'
    elif -0.8 <= z_score <= 0.8:
        performance_category = 'Acceptable Performance'
    elif -1.2 <= z_score <= 1.2:
        performance_category = 'Marginal Performance'
    else:
        performance_category = 'Not Acceptable'


    # Set graph borderline
    plt.gca().spines['top'].set_linewidth(0.8)
    plt.gca().spines['top'].set_color('gray')
    plt.gca().spines['right'].set_linewidth(0.8)
    plt.gca().spines['right'].set_color('gray')
    plt.gca().spines['bottom'].set_linewidth(0.8)
    plt.gca().spines['bottom'].set_color('gray')
    plt.gca().spines['left'].set_linewidth(0.8)
    plt.gca().spines['left'].set_color('gray')

    # Add Z Score and Performance to the title
    # Sub header for data that out of 2SD
    out_samples = group_data[group_data['REMARKS_2SD'].str.lower() == 'out'] 
    title_text = f'{group_name[2]}-{group_name[3]}\n'
    title_text += f'Total Samples: {len(group_data)}\n'
    title_text += f'Samples OUT 2SD: {len(out_samples)}\n'
    title_text += f'Z Score: {z_score:.2f}\n'
    title_text += f'Performance: {performance_category}\n'

    # Customize the axis
    plt.suptitle(f'{group_name[0]}-{group_name[1]}', fontweight='bold', fontsize='24', x=0.51)
    plt.title(title_text, y=0.99)
    plt.xlabel('SAMPLE_TAG', fontsize=20, fontweight='bold')
    plt.ylabel('VALUE (ppm)', fontsize=20, fontweight='bold')
    plt.xticks(rotation='vertical', fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid(axis='y', linestyle='--', color='gray', linewidth=1.2)
    plt.legend(bbox_to_anchor=(0.5, -0.16), loc='lower center', ncol=9, facecolor='white', prop={'size': 15, 'weight': 'bold'})
    mplcursors.cursor(hover=True).connect("add", lambda sel: sel.annotation.set_text(f'Sample Tag: {group_data["SAMPLE_TAG"].iloc[sel.target.index]}\nResult: {group_data["RESULT"].iloc[sel.target.index]}'))
    
    # Display in streamlit
    st.markdown(f'<h3 style="font-size: 14px;">{group_name[0]} - Element: {group_name[1]}, Method: {group_name[2]} - {group_name[3]}</h3>', unsafe_allow_html=True)
    st.pyplot(plt)
    plt.clf()  # Clear plot to avoid overlapping legends
