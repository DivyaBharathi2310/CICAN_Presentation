import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import os
os.environ['GRADIENT_ACCESS_TOKEN'] = st.secrets["GRADIENT_ACCESS_TOKEN"]
os.environ['GRADIENT_WORKSPACE_ID'] = st.secrets["GRADIENT_WORKSPACE_ID"]

# Load the dataset with a specified encoding
data = pd.read_csv('combined_data.csv', encoding='latin1')


# Page 1: Dashboard
def dashboard():
    st.title("Edmonton Food Drive Dashboard")
    st.image("Untitled design.jpg")

    st.subheader("💡 Abstract:")
    inspiration = '''
    Food drives are crucial for addressing hunger and supporting vulnerable communities.
    They provide essential sustenance to those in need, fostering a sense of community and compassion.
    We Norquesters are super proud to be a part of the food drive.
    '''
    st.write(inspiration)

    st.subheader("👨🏻‍💻 What our Project Does?")
    what_it_does = '''
    This machine learning model mainly focuses on predicting the number of bags collected. With this, the volunteering experience and resource allocation can be optimized.
    Further, our team has also proposed an optimized version of the data collection form.
    By addressing logistical challenges, it will enhance efficiency and help vulnerable communities receive essential sustenance more effectively.
    '''
    st.write(what_it_does)

# Page 2: Exploratory Data Analysis (EDA)
def exploratory_data_analysis():
    st.title("Exploratory Data Analysis")
    st.markdown(
        f"**Disclaimer:** The data used in this app consists ONLY of data collected in collaboration with NorQuest College during the 2023 Food Drive Project and does not represent the entire Food Drive.")
    # Rename columns for clarity
    data_cleaned = data

    # Visualize the distribution of numerical features using Plotly

    fig = px.histogram(data_cleaned, x='Time to Complete (min)', nbins=20,
                       labels={'Time to Complete (min)': 'Time to Complete'}, title="Distribution of time to complete")
    st.plotly_chart(fig)

    ### Stake Bags EDA###
    # Average Bags/Route per Stake
    stake_mean_data = data_cleaned.groupby(by='Stake')['Bags/Route'].mean().sort_values()
    stake_mean = px.bar(orientation='h', y=stake_mean_data.index, x=stake_mean_data.values,
                        labels={'y': 'Stake', 'x': 'Average Donation Bags Collected per Route'},
                        title='Average Donation Bags Collected per Route in each Stake'
                        )
    st.plotly_chart(stake_mean)

    # Total Bags/Stake
    stake_total_data = data_cleaned.groupby(by='Stake')['Donation Bags Collected'].sum().sort_values()
    stake_total = px.bar(orientation='h', y=stake_total_data.index, x=stake_total_data.values,
                         labels={'y': 'Stake', 'x': 'Total Donation Bags Collected'},
                         title='Total Donation Bags Collected in each Stake'
                         )
    st.plotly_chart(stake_total)

    ### Ward Bags EDA ###
    # Average Bags/Route per Ward
    ward_mean_data = data_cleaned.groupby(by='Ward/Branch')['Bags/Route'].mean().sort_values()
    ward_mean = px.bar(orientation='h', y=ward_mean_data.index, x=ward_mean_data.values,
                       labels={'y': 'Ward/Branch', 'x': 'Average Donation Bags Collected per Route'},
                       title='Average Donation Bags Collected per Route in each Ward/Branch',
                       height=800
                       )
    st.plotly_chart(ward_mean)

    # Total Bags/Ward
    ward_total_data = data_cleaned.groupby(by='Ward/Branch')['Donation Bags Collected'].sum().sort_values()
    ward_total = px.bar(orientation='h', y=ward_total_data.index, x=ward_total_data.values,
                        labels={'y': 'Ward/Branch', 'x': 'Total Donation Bags Collected'},
                        title='Total Donation Bags Collected in each Ward/Branch',
                        height=800
                        )
    st.plotly_chart(ward_total)


# Page 3: Machine Learning Modeling
def machine_learning_modeling_classification():
    stake_encoding = {'Edmonton North Stake': 0, 'Gateway Stake': 1, 'Riverbend Stake': 2, 'Bonnie Doon Stake': 3,
                      'YSA Stake': 4}
    ward_encoding = {'Namao Ward': 0, 'Lee Ridge Ward': 1, 'Blackmud Creek Ward': 2, 'Rabbit Hill Ward': 3,
                     'Clareview Ward': 4, 'Crawford Plains Ward': 5, 'Silver Berry Ward': 6, 'Connors Hill Ward': 7,
                     'Stony Plain Ward': 8, 'Londonderry Ward': 9, 'Southgate Ward': 10, 'Greenfield Ward': 11,
                     'Rutherford Ward': 12, 'Griesbach Ward': 13, 'Ellerslie Ward': 14, 'Forest Heights Ward': 15,
                     'Coronation Park Ward': 16, 'Woodbend Ward': 17, 'Wainwright Branch': 18,
                     'Terwillegar Park Ward': 19, 'Rio Vista Ward': 20, 'Wild Rose Ward': 21,
                     'Windsor Park YSA Ward': 22, 'Strathcona Married Student Ward': 23, 'Drayton Valley Ward': 24,
                     'Beaumont Ward': 25, 'Belmead Ward': 26}

    st.title("Machine Learning Modeling")
    st.write("Enter the details to predict if it will take > 1h to complete:")

    # Input fields for user to enter data
    stake = st.selectbox("Select a Stake", data['Stake'].unique())
    ward_branch = st.selectbox("Select a Ward/Branch", data.loc[data['Stake'] == stake, 'Ward/Branch'].unique())
    adult_volunteers = st.slider("Number of Adult Volunteers", 1, 50, 2)
    youth_volunteers = st.slider("Number of Youth Volunteers", 1, 50, 2)
    doors_in_route = st.slider("Number of Doors to check", 10, 500, 200)
    routes_completed = st.slider("Routes Completed", 1, 10, 2)

    # Cols to calc
    donation_bags_collected = data.loc[data['Ward/Branch'] == ward_branch, 'Donation Bags Collected'].mean()
    bags_per_door = int(donation_bags_collected) / int(doors_in_route)
    bags_per_route = int(donation_bags_collected) / int(routes_completed)
    total_volunteers = int(adult_volunteers) + int(youth_volunteers)

    stake_num = int(stake_encoding[stake])
    ward_branch_num = int(ward_encoding[ward_branch])

    # Predict button
    if st.button("Predict"):
        # Load the trained model
        model = joblib.load('random_forest_classifier_model.pkl')
        st.write('Model Loaded')
        # Prepare input data for prediction
        input_data = [
            [stake_num, ward_branch_num, adult_volunteers, youth_volunteers, donation_bags_collected, routes_completed,
             doors_in_route, bags_per_door, bags_per_route, total_volunteers]]

        # Make prediction
        prediction = model.predict(input_data)
        prediction_label = ['More', 'Less']

        # Display the prediction
        st.success(
            f"It will take {prediction_label[prediction[0]]} than 1 hour to complete {'these routes.' if routes_completed > 1 else 'this route.'}")

        # You can add additional information or actions based on the prediction if needed

def machine_learning_modeling_regression():
    st.title("Machine Learning Modeling")
    st.write("Enter the details to predict donation bags:")

    # Input fields for user to enter data
    completed_routes = st.slider("Completed More Than One Route", 0, 1, 0)
    routes_completed = st.slider("Routes Completed", 1, 10, 5)
    time_spent = st.slider("Time Spent (minutes)", 10, 300, 60)
    adult_volunteers = st.slider("Number of Adult Volunteers", 1, 50, 10)
    doors_in_route = st.slider("Number of Doors in Route", 10, 500, 100)
    youth_volunteers = st.slider("Number of Youth Volunteers", 0, 50, 10)

    # Predict button
    if st.button("Predict"):
        from sklearn.model_selection import train_test_split

        X = data.drop(columns=['Donation Bags Collected','Location','Ward/Branch','Stake','Unnamed: 0'])
        y = data['Donation Bags Collected']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        from sklearn.neighbors import KNeighborsRegressor
        model = KNeighborsRegressor(n_neighbors=5)  # You can adjust the number of neighbors
        model.fit(X_train, y_train)
    
        # Prepare input data for prediction
        user_input = [[adult_volunteers, youth_volunteers, time_spent, completed_routes,routes_completed , doors_in_route]] 

        # Make prediction
        prediction = model.predict(user_input)

        # Display the prediction
        st.success(f"Predicted Donation Bags: {prediction[0]}")
# Page 4: Stake/Ward Map

def neighbourhood_mapping():
    st.title("Stake/Ward Map")
    st.write("This is an interactive map try selecting a route, ward or stake to get more information about it!")
    st.markdown("""
    <iframe src="https://www.google.com/maps/d/embed?mid=1kiXlZT8tTH_cDYqfP5lKoQ8Jt7A_FbM&ehbc=2E312F&noprof=1" width="800" height="640"></iframe>
    """, unsafe_allow_html=True)


# Page 5: Data Collection
def data_collection():
    st.title("Data Collection")
    st.write("Please fill out the Google form to contribute to our Food Drive!")
    google_form_url = "https://forms.gle/NQX4z9WwqhJaeUFV8"  # YOUR_GOOGLE_FORM_URL_HERE
    st.markdown(f"[Fill out the form]({google_form_url})")


def chatbot():
    st.title("Interactive Food Drive Assistant")
    st.write("Ask a question about the Food Drive!")

    with Gradient() as gradient:
        base_model = gradient.get_base_model(base_model_slug="nous-hermes2")
        new_model_adapter = base_model.create_model_adapter(name="interactive_food_drive_model")

        user_input = st.text_input("Ask your question:")
        if user_input and user_input.lower() not in ['quit', 'exit']:
            sample_query = f"### Instruction: {user_input} \n\n### Response:"
            st.markdown(f"Asking: {sample_query}")

            # before fine-tuning
            completion = new_model_adapter.complete(query=sample_query, max_generated_token_count=100).generated_output
            st.markdown(f"Generated: {completion}")

        # Delete the model adapter after generating the response
        new_model_adapter.delete()

# Main App Logic
def main():
    st.sidebar.title("Food Drive App")
    app_page = st.sidebar.radio("Select a Page",
                                ["Dashboard", "EDA", "ML Modeling", "Stake/Ward Map", "Data Collection", "Chatbot"])

    if app_page == "Dashboard":
        dashboard()
    elif app_page == "EDA":
        exploratory_data_analysis()
    elif app_page == "ML Modeling-Time Prediction":
        machine_learning_modeling_classification()
    elif app_page == "ML Modeling-Donation Bag Prediction":
        machine_learning_modeling_regression()
    elif app_page == "Stake/Ward Map":
        neighbourhood_mapping()
    elif app_page == "Data Collection":
        data_collection()
    elif app_page == "Chatbot":
        chatbot()


if __name__ == "__main__":
    main()
