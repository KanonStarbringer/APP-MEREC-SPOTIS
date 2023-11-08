import streamlit as st
import pandas as pd
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
    #page_icon=":chart_with_upwards_trend:",  # You can customize the icon
    #layout="wide",  # You can set the layout (wide or center)
    initial_sidebar_state="auto"  # You can set the initial sidebar state
)
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

    return edited_matrix, criterion_types, num_alternatives, num_criteria

def normalize_matrix(matrix, criterion_types):
    
    normalized_matrix = matrix.copy()
    for j, criterion_type in enumerate(criterion_types):
        if criterion_type == "Benefit":
            col_min = matrix.iloc[:, j+1].min()
            normalized_matrix.iloc[:, j+1] = col_min / matrix.iloc[:, j+1]
        else:
            col_max = matrix.iloc[:, j+1].max()
            normalized_matrix.iloc[:, j+1] = matrix.iloc[:, j+1] / col_max

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
        min_val = st.number_input(f'Min value for Criterion {i+1}', 0, 10000, 0)  # You can adjust the slider ranges as needed
        max_val = st.number_input(f'Max value for Criterion {i+1}', 0, 10000, 1000)  # Initial value is just an example

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


def main():
    menu = ["Home","MEREC, MEREC-G, MEREC-H","MEREC-SPOTIS", "About"]

    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.header("Home")
        st.subheader("MEREC-SPOTIS Calculator")
        st.write("This is a MCDA Calculator for the MEREC-SPOTIS Method")
        st.write("To use this Calculator, is quite intuitive:")
        st.write("First, define how many alternatives and criteria you'll measure.")
        st.write("Then, define if the criteria are of benefit (more is better).")
        st.write("Or, if the criteria are of cost (if less is better).")

    elif choice == "MEREC, MEREC-G, MEREC-H":
        st.title("MEREC method complete")

        payoff_matrix, criterion_types, _, num_criteria = get_payoff_matrix()
        st.subheader("Payoff Matrix:")
        st.dataframe(payoff_matrix)

        normalized_matrix = normalize_matrix(payoff_matrix, criterion_types)
        st.subheader("Normalized Matrix:")
        st.dataframe(normalized_matrix)

        Si_df = calculate_performance(normalized_matrix)
        st.subheader("Calculated Performance:")
        st.dataframe(Si_df) 

        GM = calculate_performance_G(normalized_matrix)
        st.subheader("Calculated performance MEREC G")
        st.dataframe(GM)

        HI = calculate_performance_H(normalized_matrix)
        st.subheader("Calculate performance MEREC H")
        st.dataframe(HI)

        performance_without_criterion = calculate_performance_without_criterion(normalized_matrix)
        st.subheader("Performance without Criterion:")
        st.dataframe(performance_without_criterion)  

        performance_without_criterion_G = calculate_performance_without_criterion_geometric(normalized_matrix)
        st.subheader("Performance without criterion geometric")
        st.dataframe(performance_without_criterion_G)

        performance_without_criterion_H = calculate_performance_without_criterion_harmonic(normalized_matrix)
        st.subheader("Performance without criterion harmonic")
        st.dataframe(performance_without_criterion_H)

        Ej_for_criteria = compute_removal_effect_for_criteria(Si_df, performance_without_criterion)
        st.subheader("Removal Effect of Each Criterion (Ej):")
        st.dataframe(Ej_for_criteria)        

        Ej_for_criteria_G = compute_removal_effect_for_criteria_G(GM, performance_without_criterion_G)
        st.subheader("Removal Effect of Each Criterion (Ej) geometric:")
        st.dataframe(Ej_for_criteria_G)

        Ej_for_criteria_H = compute_removal_effect_for_criteria_H(HI, performance_without_criterion_H)
        st.subheader("Removel Effect of Each Criterion (Ej) harmonic:")
        st.dataframe(Ej_for_criteria_H)

        criteria_weights = compute_criteria_weights(Ej_for_criteria)
        st.subheader("Final Weights of the Criteria:")
        st.dataframe(criteria_weights) 

        criteria_weights_G = compute_criteria_weights_G(Ej_for_criteria_G)
        st.subheader("Final Weights of the Criteria G:")
        st.dataframe(criteria_weights_G)

        criteria_weights_H = compute_criteria_weights_H(Ej_for_criteria_H)
        st.subheader("Final Weights of the Criteria H:")
        st.dataframe(criteria_weights_H)

        plot_criteria_weights_all(criteria_weights, criteria_weights_G, criteria_weights_H)



    elif choice == "MEREC-SPOTIS":
        st.title("MEREC-SPOTIS Method MCDA Calculator")

        payoff_matrix, criterion_types, _, num_criteria = get_payoff_matrix()
        st.subheader("Payoff Matrix:")
        st.dataframe(payoff_matrix)

        normalized_matrix = normalize_matrix(payoff_matrix, criterion_types)
        st.subheader("Normalized Matrix:")
        st.dataframe(normalized_matrix)

        Si_df = calculate_performance(normalized_matrix)
        st.subheader("Calculated Performance:")
        st.dataframe(Si_df)

        performance_without_criterion = calculate_performance_without_criterion(normalized_matrix)
        st.subheader("Performance without Criterion:")
        st.dataframe(performance_without_criterion)

        Ej_for_criteria = compute_removal_effect_for_criteria(Si_df, performance_without_criterion)
        st.subheader("Removal Effect of Each Criterion (Ej):")
        st.dataframe(Ej_for_criteria)

        criteria_weights = compute_criteria_weights(Ej_for_criteria)
        st.subheader("Final Weights of the Criteria:")
        st.dataframe(criteria_weights)

        plot_criteria_weights(criteria_weights)

        criteria_ranges = get_criteria_ranges(num_criteria)
        st.subheader("Criteria Ranges:")
        st.dataframe(criteria_ranges) 

        ideal_solution = compute_ideal_solution(criterion_types, criteria_ranges)
        st.subheader("Ideal Solution:")
        st.dataframe(ideal_solution)

        normalized_spotis = normalize_spotis(payoff_matrix, ideal_solution, criteria_ranges, criterion_types)
        st.subheader("Normalized Matrix:")
        st.dataframe(normalized_spotis)

        weighted_spotis = apply_weights(normalized_spotis, criteria_weights)
        st.subheader("Final Spotis Matrix:")
        st.dataframe(weighted_spotis)

        sorted_spotis = sum_and_sort_rows(weighted_spotis)
        st.subheader("Sorted Spotis Matrix:")
        st.dataframe(sorted_spotis)

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
