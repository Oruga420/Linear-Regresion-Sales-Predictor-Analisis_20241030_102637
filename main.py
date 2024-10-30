import streamlit as st
import pandas as pd
import plotly.express as px
from utils import validate_data, preprocess_data
from models import train_linear_regression, get_model_metrics

st.set_page_config(
    page_title="Sales Prediction App",
    page_icon="📈",
    layout="wide"
)

def main():
    st.title("📊 Sales Prediction Analysis")
    st.write("""
    Upload your sales data to predict future sales based on Marketing Spend and Price.
    The data should be in CSV format with columns: Marketing_Spend, Price, and Sales.
    """)

    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            # Load and validate data
            df = pd.read_csv(uploaded_file)
            if not validate_data(df):
                st.error("Please ensure your CSV contains Marketing_Spend, Price, and Sales columns!")
                return

            # Preprocess data
            X, y = preprocess_data(df)

            # Train model
            model, X_train, X_test, y_train, y_test, y_pred = train_linear_regression(X, y)

            # Display results in columns
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Model Performance")
                metrics = get_model_metrics(model, X_test, y_test)
                st.write("R² Score:", f"{metrics['r2']:.3f}")
                st.write("Mean Squared Error:", f"{metrics['mse']:.3f}")
                
                st.subheader("Model Coefficients")
                coef_df = pd.DataFrame({
                    'Feature': ['Marketing Spend', 'Price'],
                    'Coefficient': model.coef_
                })
                st.table(coef_df)

            with col2:
                st.subheader("Actual vs Predicted Sales")
                fig = px.scatter(
                    x=y_test,
                    y=y_pred,
                    labels={'x': 'Actual Sales', 'y': 'Predicted Sales'},
                    title='Actual vs Predicted Sales'
                )
                fig.add_shape(
                    type='line',
                    x0=y_test.min(),
                    y0=y_test.min(),
                    x1=y_test.max(),
                    y1=y_test.max(),
                    line=dict(color='red', dash='dash')
                )
                st.plotly_chart(fig)

            # Feature importance visualization
            st.subheader("Feature Importance Analysis")
            importance_fig = px.bar(
                coef_df,
                x='Feature',
                y='Coefficient',
                title='Feature Importance'
            )
            st.plotly_chart(importance_fig)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.write("Please ensure your data is properly formatted and try again.")

if __name__ == "__main__":
    main()
