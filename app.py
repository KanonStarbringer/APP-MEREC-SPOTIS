import streamlit as st
import pandas as pd
import xlsxwriter
import openpyxl
import numpy as np
from st_aggrid import AgGrid
import io
import math
from scipy.integrate import quad
from reportlab.lib.pagesizes import letter, A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle
import plotly.express as px
import plotly.graph_objs as go

# Set the app title and description
st.set_page_config(
    page_title="MEREC-SPOTIS Calculator",
    initial_sidebar_state="auto"
)
# Function to download the Excel template
def download_template():
    # Adjust based on the number of alternatives and criteria
    num_alternatives = 9  # You can set a default number or ask the user for input
    num_criteria = 17  # Same for criteria

    # Generate a list of alternative names
    alternatives = [f'A{i+1}' for i in range(num_alternatives)]

    # Create data for the template: "C1", "C2", ..., in the first row, and "Max/Min" in the second row
    criteria_labels = [f'C{i+1}' for i in range(num_criteria)]
    benefit_cost_row = ['Max' if i < 10 else 'Min' for i in range(num_criteria)]  # First 10 are Max, rest are Min

    # Prepare data for the DataFrame
    data = {f'C{i+1}': [''] * num_alternatives for i in range(num_criteria)}
    df = pd.DataFrame(data)

    # Set the first row for the "C1", "C2", ..., and second row for the "Max/Min"
    df.loc[-2] = criteria_labels
    df.loc[-1] = benefit_cost_row
    df.index = df.index + 2  # Shifting the index to make space for the new rows
    df = df.sort_index()

    # Add the "A/C" column for alternatives
    df.insert(0, 'A/C', ['A/C'] + [''] + alternatives)

    # Convert the DataFrame to an Excel file
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, header=False)

    # Provide a download link for the template
    st.download_button(
        label="Download Excel template",
        data=excel_buffer,
        file_name="MEREC_SPOTIS_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# Function to read Excel file
def read_excel(uploaded_file):
    df = pd.read_excel(uploaded_file)

    # Show the first two rows for debugging purposes
    #st.write("First Row (Headers):")
    #st.write(df.columns.tolist())
    
    #st.write("Second Row (Criterion Types):")
    #st.write(df.iloc[0, 1:].tolist())

    # Extract the "Max" or "Min" labels from the second row (the row defining if it's Benefit or Cost)
    criterion_types = df.iloc[0, 1:].apply(lambda x: 'Benefit' if str(x).strip().lower() == 'max' else 'Cost').tolist()

    # Remove the second row (the row with "Max" or "Min" labels) from the DataFrame
    df = df.drop(0).reset_index(drop=True)

    # Rename columns to C1, C2, etc. and keep the first column as 'A/C'
    num_criteria = len(df.columns) - 1
    columns = ['A/C'] + [f'C{i+1}' for i in range(num_criteria)]
    df.columns = columns

    # Convert all the criteria columns (except the 'A/C' column) to numeric values
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df, criterion_types, df.shape[0], num_criteria

# Function to manually create the payoff matrix
def get_payoff_matrix():
    num_alternatives = st.number_input("Enter the number of alternatives:", min_value=2, value=2, step=1, key="num_alternatives")
    num_criteria = st.number_input("Enter the number of criteria:", min_value=1, value=1, step=1, key="num_criteria")

    # Create a DataFrame to hold the payoff matrix
    columns = ['A/C'] + [f'C{i+1}' for i in range(num_criteria)]
    data = [[f'A{j+1}'] + [0 for _ in range(num_criteria)] for j in range(num_alternatives)]
    payoff_matrix = pd.DataFrame(data, columns=columns)

    # Create an ag-Grid component
    grid_response = AgGrid(payoff_matrix, editable=True, index=False, fit_columns_on_grid_load=True)

    # Get the edited DataFrame from the AgGrid response
    edited_matrix = grid_response['data']

    # Convert the edited matrix values to numeric
    for col in edited_matrix.columns:
        if col != 'A/C':  # Skip the 'A/C' column which contains string labels
            edited_matrix[col] = pd.to_numeric(edited_matrix[col])

    # Get the type of each criterion (Benefit or Cost)
    criterion_types = []
    for i in range(num_criteria):
        criterion_label = f"C{i+1}"
        criterion_type = st.selectbox(f"{criterion_label} - Benefit or Cost?", ["Benefit", "Cost"])
        criterion_types.append(criterion_type)
    
    # Print the criterion types for debugging
    st.write("Criterion Types (Manual Input):", criterion_types)

    return edited_matrix, criterion_types, num_alternatives, num_criteria

def normalize_matrix_manual(matrix, criterion_types):
    normalized_matrix = matrix.copy()

    for j, criterion_type in enumerate(criterion_types):
        if criterion_type == "Benefit":
            col_min = matrix.iloc[:, j+1].min()
            normalized_matrix.iloc[:, j+1] = col_min / matrix.iloc[:, j+1]  # Benefit normalization
        else:  # Cost criterion
            col_max = matrix.iloc[:, j+1].max()
            normalized_matrix.iloc[:, j+1] = matrix.iloc[:, j+1] / col_max  # Cost normalization
    
    return normalized_matrix

def normalize_matrix_excel(matrix, criterion_types_from_excel):
    normalized_matrix = matrix.copy()

    for j, criterion_type in enumerate(criterion_types_from_excel):
        if criterion_type == "Benefit":
            col_min = matrix.iloc[:, j+1].min()
            normalized_matrix.iloc[:, j+1] = col_min / matrix.iloc[:, j+1]  # Benefit normalization
        else:  # Cost criterion
            col_max = matrix.iloc[:, j+1].max()
            normalized_matrix.iloc[:, j+1] = matrix.iloc[:, j+1] / col_max  # Cost normalization
    
    return normalized_matrix

def calculate_performance(normalized_matrix):
    """
    Calculate the overall performance of the alternatives (Si) based on the MEREC method.

    :param normalized_matrix: A 2D list representing the normalized decision matrix.
    :return: A list representing the performance scores of the alternatives.
    """

    m = normalized_matrix.shape[1] - 1 # Subtracting 1 to exclude the 'A/C' column
    
    S_i = []

    for _, row in normalized_matrix.drop(columns='A/C').iterrows():
        # Sum of the natural logarithms of the normalized values for each alternative
        sum_ln = sum([math.fabs(math.log(nij)) for nij in row])

        # Calculate Si using the provided formula
        Si = math.log(1 + (1/m) * sum_ln)
        S_i.append(Si)

    return S_i

def calculate_performance_G(normalized_matrix):
    """
    Calculate the overall performance of the alternatives (Si) based on the MEREC method.

    :param normalized_matrix: A 2D list representing the normalized decision matrix.
    :return: A list representing the performance scores of the alternatives.
    """

    m = normalized_matrix.shape[1] - 1 # Subtracting 1 to exclude the 'A/C' column
    
    GM = []

    for _, row in normalized_matrix.drop(columns='A/C').iterrows():
        prod = np.prod(row)
        prod_gm = np.power(prod, 1/m)
        GM.append(prod_gm)

    return GM

def calculate_performance_H(normalized_matrix):
    m = normalized_matrix.shape[1] - 1 
    HI = []

    for _, row in normalized_matrix.drop(columns='A/C').iterrows():
        # Sum of the inverses of the normalized values for each alternative
        sum_of_inverses = sum([1/nij if nij != 0 else 0 for nij in row])  # Check to avoid division by zero
        Hi = m / sum_of_inverses if sum_of_inverses != 0 else 0
        HI.append(Hi)

    return HI

def calculate_performance_without_criterion(normalized_matrix):
    """
    Calculate the performance of the alternatives by removing each criterion.

    :param normalized_matrix: A 2D list representing the normalized decision matrix.
    :return: A 2D list where each inner list represents the performance scores of the alternatives 
             without considering a particular criterion.
    """
    
    m = normalized_matrix.shape[1] - 1 # Subtracting 1 to exclude the 'A/C' column
    
    performances_without_criterion = []

    for j in range(m):
        performance_scores = []

        for _, row in normalized_matrix.drop(columns='A/C').iterrows():
            # Excluding the jth criterion
            values_without_j = [value for idx, value in enumerate(row) if idx != j]
            # Inside your loop where you exclude jth criterion
            # print(f"Original row: {row}")
            # print(f"Row without {j}th criterion: {values_without_j}")

            
            # Sum of the natural logarithms of the normalized values for each alternative
            sum_ln = sum([math.fabs(math.log(nik)) for nik in values_without_j])

            # Calculate Si using the provided formula but without jth criterion
            Si_j = math.log(1 + (1/(m)) * sum_ln)  # (m-1) since we're excluding one criterion
            performance_scores.append(Si_j)

        performances_without_criterion.append(performance_scores)

    # Transpose the results to match the format shown in the image
    return pd.DataFrame(performances_without_criterion).transpose().values.tolist()

def calculate_performance_without_criterion_geometric(normalized_matrix):
    """
    Calculate the performance of the alternatives by removing each criterion using the
    geometric mean (MEREC-G method).

    :param normalized_matrix: A DataFrame representing the normalized decision matrix.
    :return: A 2D list where each inner list represents the performance scores of the alternatives
             without considering a particular criterion.
    """
    
    m = normalized_matrix.shape[1] - 1  # Subtracting 1 to exclude the 'A/C' column
    
    performances_without_criterion_geometric = []

    for j in range(1, m+1):  # Starting from 1 to align with the actual data columns
        performance_scores = []

        for _, row in normalized_matrix.drop(columns='A/C').iterrows():
            # Excluding the jth criterion
            values_without_j = [value for idx, value in enumerate(row, 1) if idx != j]

            # Calculate the geometric mean without the jth criterion
            product_of_values = np.prod(values_without_j)
            Si_j = np.power(product_of_values, 1/(m-1))
            performance_scores.append(Si_j)

        performances_without_criterion_geometric.append(performance_scores)

    return pd.DataFrame(performances_without_criterion_geometric).transpose().values.tolist()

def calculate_performance_without_criterion_harmonic(normalized_matrix):
    """
    Calculate the performance of the alternatives by removing each criterion using the
    harmonic mean (MEREC-H method).

    :param normalized_matrix: A DataFrame representing the normalized decision matrix.
    :return: A 2D list where each inner list represents the performance scores of the alternatives
             without considering a particular criterion.
    """
    
    m = normalized_matrix.shape[1] - 1  # Subtracting 1 to exclude the 'A/C' column
    
    performances_without_criterion_harmonic = []

    for j in range(1, m+1):  # Starting from 1 to align with the actual data columns
        performance_scores = []

        for _, row in normalized_matrix.drop(columns='A/C').iterrows():
            # Excluding the jth criterion
            values_without_j = [value for idx, value in enumerate(row, 1) if idx != j]

            # Calculate the harmonic mean without the jth criterion
            sum_of_inverses = sum([1/nik if nik != 0 else 0 for nik in values_without_j])
            Si_j = (m-1) / sum_of_inverses if sum_of_inverses != 0 else 0
            performance_scores.append(Si_j)

        performances_without_criterion_harmonic.append(performance_scores)

    return pd.DataFrame(performances_without_criterion_harmonic).transpose().values.tolist()

def compute_removal_effect_for_criteria(Si, performances_without_criterion):
    """
    Compute the removal effect of each criterion.

    :param Si: A list representing the overall performance scores of the alternatives.
    :param performances_without_criterion: A 2D list where each inner list represents the performance scores of the alternatives without considering a particular criterion.
    :return: A list representing the removal effect of each criterion.
    """
    
    E_j = []
    
    # Number of criteria
    num_criteria = len(performances_without_criterion[0])
    
    # For each criterion j
    for j in range(num_criteria):
        summation = 0
        # For each alternative i
        for i in range(len(Si)):
            summation += abs(performances_without_criterion[i][j] - Si[i])
        E_j.append(summation)
    
    return E_j

def compute_removal_effect_for_criteria_G(GM, performances_without_criterion_geometric):
    """
    Compute the removal effect of each criterion.

    :param Si: A list representing the overall performance scores of the alternatives.
    :param performances_without_criterion: A 2D list where each inner list represents the performance scores of the alternatives without considering a particular criterion.
    :return: A list representing the removal effect of each criterion.
    """
    
    E_jG = []
    
    # Number of criteria
    num_criteria_geometric = len(performances_without_criterion_geometric[0])
    
    # For each criterion j
    for j in range(num_criteria_geometric):
        summation_geometric = 0
        # For each alternative i
        for i in range(len(GM)):
            summation_geometric += abs(performances_without_criterion_geometric[i][j] - GM[i])
        E_jG.append(summation_geometric)
    
    return E_jG

def compute_removal_effect_for_criteria_H(HI, performances_without_criterion_harmonic):
    """
    Compute the removal effect of each criterion.

    :param Si: A list representing the overall performance scores of the alternatives.
    :param performances_without_criterion: A 2D list where each inner list represents the performance scores of the alternatives without considering a particular criterion.
    :return: A list representing the removal effect of each criterion.
    """
    
    E_jH = []
    
    # Number of criteria
    num_criteria_harmonic = len(performances_without_criterion_harmonic[0])
    
    # For each criterion j
    for j in range(num_criteria_harmonic):
        summation_harmonic = 0
        # For each alternative i
        for i in range(len(HI)):
            summation_harmonic += abs(performances_without_criterion_harmonic[i][j] - HI[i])
        E_jH.append(summation_harmonic)
    
    return E_jH

def compute_criteria_weights(Ej_list):
    """
    Compute the final weights of the criteria based on their removal effects.

    :param Ej_list: A list representing the removal effects of each criterion.
    :return: A list representing the final weight of each criterion.
    """
    
    total_removal_effect = sum(Ej_list)
    weights = [Ej / total_removal_effect for Ej in Ej_list]
    
    
    return weights

def compute_criteria_weights_G(EjG_list):
    total_removal_effect_G = sum(EjG_list)
    weights_G = [EjG / total_removal_effect_G for EjG in EjG_list]
    
    
    return weights_G

def compute_criteria_weights_H(EjH_list):
    total_removal_effect_H = sum(EjH_list)
    weights_H = [EjH / total_removal_effect_H for EjH in EjH_list]
    
    
    return weights_H

# def calculate_variables(weights):
#     w_values = weights.values
#     variables_df = pd.DataFrame({'w': w_values})

#     return variables_df

def get_criteria_ranges(num_criteria):
    """
    Function to get the min and max values (ranges) for each criterion.

    Parameters:
    - num_criteria: number of criteria

    Returns:
    - criteria_ranges: a dictionary with the min and max values for each criterion
    """
    # Initialize a dictionary to store the min and max values for each criterion
    criteria_ranges = {}

    # Prompt the user to input the min and max values for each criterion
    for i in range(num_criteria):
        st.write(f"Enter the range for Criterion {i+1}:")

        # Streamlit sliders to input min and max values
        min_val = st.number_input(f'Min value for Criterion {i+1}', 0, 50000, 0)  # You can adjust the slider ranges as needed
        max_val = st.number_input(f'Max value for Criterion {i+1}', 0, 50000, 1000)  # Initial value is just an example

        criteria_ranges[f"C{i+1}"] = {"min": min_val, "max": max_val}

    return criteria_ranges

def plot_criteria_weights(weights):
    # Dynamically generate the criteria names based on the number of weights
    criteria_names = [f"Criterion {i+1}" for i in range(len(weights))]

    # Convert the criteria weights and names to a DataFrame
    criteria_df = pd.DataFrame({
        'Criteria': criteria_names,
        'Weight': weights
    })

    # Plot the weights using Plotly bar chart
    fig = px.bar(
        criteria_df,
        x='Criteria',
        y='Weight',
        labels={'Criteria': 'Criteria', 'Weight': 'Weight'},
        title='Weights for Criteria',
    )

    # Customize the chart layout
    fig.update_layout(
        xaxis_title_text='Criteria',
        yaxis_title_text='Weight',
        xaxis_tickangle=-45,
    )

    # Show the Plotly chart in Streamlit
    st.plotly_chart(fig)

def plot_criteria_weights_all(weights, weights_G, weights_H):
    """
    Plot the criteria weights calculated by MEREC, MEREC-G, and MEREC-H methods.

    :param criteria_labels: A list of criterion labels (e.g., ['C1', 'C2', 'C3']).
    :param weights_merec: A list of weights from the MEREC method.
    :param weights_merec_g: A list of weights from the MEREC-G method.
    :param weights_merec_h: A list of weights from the MEREC-H method.
    """
    # Generate criteria labels based on the length of the weights list
    num_criteria = len(weights)
    criteria_labels = [f'C{i+1}' for i in range(num_criteria)]

    # Create bar chart
    fig = go.Figure(data=[
        go.Bar(name='MEREC', x=criteria_labels, y=weights),
        go.Bar(name='MEREC-G', x=criteria_labels, y=weights_G),
        go.Bar(name='MEREC-H', x=criteria_labels, y=weights_H)
    ])
    
    # Change the bar mode
    fig.update_layout(barmode='group')
    
    # Update layout with title and axis labels
    fig.update_layout(
        title='Comparison of Criteria Weights by MEREC Methods',
        xaxis_title='Criteria',
        yaxis_title='Weights',
        legend_title='Methods'
    )
    
    st.plotly_chart(fig)


def compute_ideal_solution(criterion_types, criteria_ranges):
    """
    Compute the ideal solution based on criterion types and criteria ranges.
    
    Parameters:
    - criterion_types: A list of strings, where each string is either "Benefit" or "Cost".
    - criteria_ranges: A dictionary with the min and max values for each criterion.
    
    Returns:
    - A list representing the ideal solution for each criterion.
    """
    ideal_solution = []

    # For each criterion
    for j in range(len(criterion_types)):
        # If it's a Benefit criterion (more is better)
        if criterion_types[j] == "Benefit":
            ideal_score = criteria_ranges[f"C{j+1}"]["max"]
        # If it's a Cost criterion (less is better)
        elif criterion_types[j] == "Cost":
            ideal_score = criteria_ranges[f"C{j+1}"]["min"]
        else:
            raise ValueError(f"Unknown criterion type: {criterion_types[j]}")
        
        ideal_solution.append(ideal_score)

    return ideal_solution

def normalize_spotis(matrix, ideal_solution, criteria_ranges, criterion_types):
    
    normalized_spotis = matrix.copy()
    
    for j, criterion_type in enumerate(criterion_types):
        for i in range(len(matrix)):
            s_ij = float(matrix.iloc[i, j+1])
            s_j_star = float(ideal_solution[j])
            s_j_max = float(criteria_ranges[f"C{j+1}"]["max"])
            s_j_min = float(criteria_ranges[f"C{j+1}"]["min"])
            
            normalized_value = (s_ij - s_j_min) / (s_j_max - s_j_min)
            
            normalized_spotis.iloc[i, j+1] = normalized_value
    
    return normalized_spotis

def apply_weights(normalized_spotis, weights):
    weighted_matrix = normalized_spotis.copy()
    
    # Start from 1 because the first column is alternatives' names
    for j, weight in enumerate(weights):
        weighted_matrix.iloc[:, j+1] = normalized_spotis.iloc[:, j+1] * weight
    
    return weighted_matrix

def sum_and_sort_rows(weighted_matrix):
    # Sum the rows excluding the first column with alternatives' names
    weighted_matrix['Total'] = weighted_matrix.iloc[:, 1:].sum(axis=1)

    # Sort the matrix based on the 'Total' column in descending order
    sorted_matrix = weighted_matrix.sort_values(by='Total', ascending=True)
    
    return sorted_matrix

# Main function
def main():
    menu = ["Home", "MEREC Methods Comparison (MEREC, MEREC-G, MEREC-H)", "MEREC-SPOTIS", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.header("Home")
        st.subheader("Welcome to the MEREC-SPOTIS Calculator!")
        st.write("This is an MCDA Calculator for the MEREC-SPOTIS method and related methods.")
        st.write("Please use the side menu to select a method.")
        st.write("1. MEREC Methods Comparison: Calculate and compare the weights using MEREC, MEREC-G, and MEREC-H.")
        st.write("2. MEREC-SPOTIS: Apply the SPOTIS method with MEREC-derived weights.")

    elif choice == "MEREC Methods Comparison (MEREC, MEREC-G, MEREC-H)":
        st.title("MEREC Methods Comparison (MEREC, MEREC-G, MEREC-H)")

        data_input_method = st.selectbox("Select Data Input Method:", ["Manual Input", "Upload Excel"])

        # Handle data input: either manually or via Excel
        if data_input_method == "Upload Excel":
            st.write("Download the template to fill out the data:")
            download_template()
            uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])

            if uploaded_file:
                payoff_matrix, criterion_types, num_alternatives, num_criteria = read_excel(uploaded_file)
                st.subheader("Payoff Matrix from Uploaded Excel:")
                st.dataframe(payoff_matrix)

        elif data_input_method == "Manual Input":
            payoff_matrix, criterion_types, num_alternatives, num_criteria = get_payoff_matrix()
            st.subheader("Payoff Matrix:")
            st.dataframe(payoff_matrix)

        if 'payoff_matrix' in locals():
            normalized_matrix = normalize_matrix_excel(payoff_matrix, criterion_types) if data_input_method == "Upload Excel" else normalize_matrix_manual(payoff_matrix, criterion_types)
            st.subheader("Normalized Matrix:")
            st.dataframe(normalized_matrix)

            # MEREC calculations
            try:
                Si_df = calculate_performance(normalized_matrix)
                st.subheader("Calculated Performance (MEREC):")
                st.dataframe(pd.DataFrame(Si_df).T)

                performance_without_criterion = calculate_performance_without_criterion(normalized_matrix)
                st.subheader("Performance without Criterion (MEREC):")
                st.dataframe(pd.DataFrame(performance_without_criterion).T)

                Ej_for_criteria = compute_removal_effect_for_criteria(Si_df, performance_without_criterion)
                st.subheader("Removal Effect of Each Criterion (MEREC):")
                st.dataframe(pd.DataFrame(Ej_for_criteria).T)

                criteria_weights_merec = compute_criteria_weights(Ej_for_criteria)
                st.subheader("Final Weights of the Criteria (MEREC):")
                st.dataframe(pd.DataFrame(criteria_weights_merec).T)

            except Exception as e:
                st.error(f"Error in MEREC calculations: {e}")

            # MEREC-G calculations
            try:
                GM_df = calculate_performance_G(normalized_matrix)
                st.subheader("Calculated Performance (MEREC-G):")
                st.dataframe(pd.DataFrame(GM_df).T)

                performance_without_criterion_G = calculate_performance_without_criterion_geometric(normalized_matrix)
                st.subheader("Performance without Criterion (MEREC-G):")
                st.dataframe(pd.DataFrame(performance_without_criterion_G).T)

                Ej_for_criteria_G = compute_removal_effect_for_criteria_G(GM_df, performance_without_criterion_G)
                st.subheader("Removal Effect of Each Criterion (MEREC-G):")
                st.dataframe(pd.DataFrame(Ej_for_criteria_G).T)

                criteria_weights_g = compute_criteria_weights_G(Ej_for_criteria_G)
                st.subheader("Final Weights of the Criteria (MEREC-G):")
                st.dataframe(pd.DataFrame(criteria_weights_g).T)

            except Exception as e:
                st.error(f"Error in MEREC-G calculations: {e}")

            # MEREC-H calculations
            try:
                HI_df = calculate_performance_H(normalized_matrix)
                st.subheader("Calculated Performance (MEREC-H):")
                st.dataframe(pd.DataFrame(HI_df).T)

                performance_without_criterion_H = calculate_performance_without_criterion_harmonic(normalized_matrix)
                st.subheader("Performance without Criterion (MEREC-H):")
                st.dataframe(pd.DataFrame(performance_without_criterion_H).T)

                Ej_for_criteria_H = compute_removal_effect_for_criteria_H(HI_df, performance_without_criterion_H)
                st.subheader("Removal Effect of Each Criterion (MEREC-H):")
                st.dataframe(pd.DataFrame(Ej_for_criteria_H).T)

                criteria_weights_h = compute_criteria_weights_H(Ej_for_criteria_H)
                st.subheader("Final Weights of the Criteria (MEREC-H):")
                st.dataframe(pd.DataFrame(criteria_weights_h).T)

            except Exception as e:
                st.error(f"Error in MEREC-H calculations: {e}")

            # Plot comparison of weights for MEREC, MEREC-G, and MEREC-H
            plot_criteria_weights_all(criteria_weights_merec, criteria_weights_g, criteria_weights_h)

    elif choice == "MEREC-SPOTIS":
        st.title("MEREC-SPOTIS Method MCDA Calculator")

        # Input data via Excel or Manual
        data_input_method = st.selectbox("Select Data Input Method:", ["Manual Input", "Upload Excel"])

        if data_input_method == "Upload Excel":
            st.write("Download the template to fill out the data:")
            download_template()
            uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])

            if uploaded_file:
                payoff_matrix, criterion_types, num_alternatives, num_criteria = read_excel(uploaded_file)
                st.subheader("Payoff Matrix from Uploaded Excel:")
                st.dataframe(payoff_matrix)

        elif data_input_method == "Manual Input":
            payoff_matrix, criterion_types, num_alternatives, num_criteria = get_payoff_matrix()
            st.subheader("Payoff Matrix:")
            st.dataframe(payoff_matrix)

        if 'payoff_matrix' in locals():
            criteria_ranges = get_criteria_ranges(num_criteria)
            st.subheader("Criteria Ranges:")
            st.dataframe(criteria_ranges)

            ideal_solution = compute_ideal_solution(criterion_types, criteria_ranges)
            st.subheader("Ideal Solution:")
            st.dataframe(pd.DataFrame(ideal_solution).T)

            weight_method = st.selectbox("Select Criteria Weights Method:", ["Manual", "MEREC", "MEREC-G", "MEREC-H"])

            # Initialize criteria_weights to None
            criteria_weights = None

            if weight_method == "Manual":
                criteria_weights = []
                for i in range(num_criteria):
                    weight = st.number_input(f'Weight for Criterion {i+1}', min_value=0.0, max_value=1.0, value=0.0)
                    criteria_weights.append(weight)
                criteria_weights = [weight / sum(criteria_weights) for weight in criteria_weights]

            elif weight_method == "MEREC":
                try:
                    normalized_matrix = normalize_matrix_excel(payoff_matrix, criterion_types) if data_input_method == "Upload Excel" else normalize_matrix_manual(payoff_matrix, criterion_types)
                    st.subheader("Normalized Matrix:")
                    st.dataframe(normalized_matrix)

                    Si_df = calculate_performance(normalized_matrix)
                    st.subheader("Calculated Performance (MEREC):")
                    st.dataframe(pd.DataFrame(Si_df).T)

                    performance_without_criterion = calculate_performance_without_criterion(normalized_matrix)
                    st.subheader("Performance without Criterion (MEREC):")
                    st.dataframe(pd.DataFrame(performance_without_criterion).T)

                    Ej_for_criteria = compute_removal_effect_for_criteria(Si_df, performance_without_criterion)
                    st.subheader("Removal Effect of Each Criterion (MEREC):")
                    st.dataframe(pd.DataFrame(Ej_for_criteria).T)

                    criteria_weights = compute_criteria_weights(Ej_for_criteria)
                    st.subheader("Final Weights of the Criteria (MEREC):")
                    st.dataframe(pd.DataFrame(criteria_weights).T)

                except Exception as e:
                    st.error(f"Error calculating MEREC: {e}")

            elif weight_method == "MEREC-G":
                try:
                    normalized_matrix = normalize_matrix_excel(payoff_matrix, criterion_types) if data_input_method == "Upload Excel" else normalize_matrix_manual(payoff_matrix, criterion_types)
                    st.subheader("Normalized Matrix:")
                    st.dataframe(normalized_matrix)

                    GM_df = calculate_performance_G(normalized_matrix)
                    st.subheader("Calculated Performance (MEREC-G):")
                    st.dataframe(pd.DataFrame(GM_df).T)

                    performance_without_criterion_G = calculate_performance_without_criterion_geometric(normalized_matrix)
                    st.subheader("Performance without Criterion (MEREC-G):")
                    st.dataframe(pd.DataFrame(performance_without_criterion_G).T)

                    Ej_for_criteria_G = compute_removal_effect_for_criteria_G(GM_df, performance_without_criterion_G)
                    st.subheader("Removal Effect of Each Criterion (MEREC-G):")
                    st.dataframe(pd.DataFrame(Ej_for_criteria_G).T)

                    criteria_weights = compute_criteria_weights_G(Ej_for_criteria_G)
                    st.subheader("Final Weights of the Criteria (MEREC-G):")
                    st.dataframe(pd.DataFrame(criteria_weights).T)

                except Exception as e:
                    st.error(f"Error calculating MEREC-G: {e}")

            elif weight_method == "MEREC-H":
                try:
                    normalized_matrix = normalize_matrix_excel(payoff_matrix, criterion_types) if data_input_method == "Upload Excel" else normalize_matrix_manual(payoff_matrix, criterion_types)
                    st.subheader("Normalized Matrix:")
                    st.dataframe(normalized_matrix)

                    HI_df = calculate_performance_H(normalized_matrix)
                    st.subheader("Calculated Performance (MEREC-H):")
                    st.dataframe(pd.DataFrame(HI_df).T)

                    performance_without_criterion_H = calculate_performance_without_criterion_harmonic(normalized_matrix)
                    st.subheader("Performance without Criterion (MEREC-H):")
                    st.dataframe(pd.DataFrame(performance_without_criterion_H).T)

                    Ej_for_criteria_H = compute_removal_effect_for_criteria_H(HI_df, performance_without_criterion_H)
                    st.subheader("Removal Effect of Each Criterion (MEREC-H):")
                    st.dataframe(pd.DataFrame(Ej_for_criteria_H).T)

                    criteria_weights = compute_criteria_weights_H(Ej_for_criteria_H)
                    st.subheader("Final Weights of the Criteria (MEREC-H):")
                    st.dataframe(pd.DataFrame(criteria_weights).T)

                except Exception as e:
                    st.error(f"Error calculating MEREC-H: {e}")

            # Check if criteria_weights are calculated before using them
            if criteria_weights is None:
                st.error("Criteria weights have not been calculated. Please select a valid weighting method.")
            else:
                # Proceed with SPOTIS calculations using criteria_weights
                try:
                    normalized_spotis = normalize_spotis(payoff_matrix, ideal_solution, criteria_ranges, criterion_types)
                    st.subheader("Normalized SPOTIS Matrix:")
                    st.dataframe(normalized_spotis)

                    weighted_spotis = apply_weights(normalized_spotis, criteria_weights)
                    st.subheader("Final Spotis Matrix:")
                    st.dataframe(weighted_spotis)

                    sorted_spotis = sum_and_sort_rows(weighted_spotis)
                    st.subheader("Sorted Spotis Matrix:")
                    st.dataframe(sorted_spotis)
                except Exception as e:
                    st.error(f"Error calculating SPOTIS: {e}")
    else:
        st.subheader("About")
        st.write("The MEREC Method is a method created by Ghorabee et al. [2021]")
        st.write("The Method SPOTIS is a method created by Dezert et al. [2020]")
        st.write("Both Articles")
        st.write("https://www.mdpi.com/2073-8994/13/4/525")
        st.write('https://www.researchgate.net/publication/344069742_The_SPOTIS_Rank_Reversal_Free_Method_for_Multi-Criteria_Decision-Making_Support')
        st.write("To cite this work:")
        st.write("Araujo, Tullio Mozart Pires de Castro; Junior, Célio Manso de Azeveodo; Gomes, Carlos Francisco Simões.; Santos, Marcos dos. MEREC-SPOTIS For Decision Making (v1), Universidade Federal Fluminense, Niterói, Rio de Janeiro, 2023.")
    
    # Add logo to the sidebar
    logo_path = "https://i.imgur.com/g7fITf4.png"  # Replace with the actual path to your logo image file
    st.sidebar.image(logo_path, use_column_width=True)

if __name__ == "__main__":
    main()
