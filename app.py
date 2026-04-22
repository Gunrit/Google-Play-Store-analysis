import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# PAGE CONFIG
st.set_page_config(
    page_title="Google Play Store Reviews Analytics",
    page_icon="📱",
    layout="wide"
)

# LOAD DATA
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_apps_data.csv")
    return df

df = load_data()

# HEADER
st.title("📱 Google Play Store Reviews Analytics")
st.markdown(
    "An interactive dashboard to analyze app categories, ratings, installs, revenue, and user sentiment."
)

# SIDEBAR FILTERS
st.sidebar.header("Filter Data")

category_options = ["All"] + sorted(df["Category"].dropna().unique().tolist())
selected_category = st.sidebar.selectbox("Select Category", category_options)

type_options = ["All"] + sorted(df["Type"].dropna().unique().tolist())
selected_type = st.sidebar.selectbox("Select App Type", type_options)

rating_range = st.sidebar.slider(
    "Select Rating Range",
    float(df["Rating"].min()),
    float(df["Rating"].max()),
    (float(df["Rating"].min()), float(df["Rating"].max()))
)

# APPLY FILTERS
filtered_df = df.copy()

if selected_category != "All":
    filtered_df = filtered_df[filtered_df["Category"] == selected_category]

if selected_type != "All":
    filtered_df = filtered_df[filtered_df["Type"] == selected_type]

filtered_df = filtered_df[
    (filtered_df["Rating"] >= rating_range[0]) &
    (filtered_df["Rating"] <= rating_range[1])
]

# KPI METRICS
total_apps = filtered_df["App"].nunique()
avg_rating = round(filtered_df["Rating"].mean(), 2)
total_reviews = int(filtered_df.groupby("App")["Reviews"].first().sum())

avg_sentiment = round(filtered_df["Sentiment_Score"].mean(), 2)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Apps", total_apps)
col2.metric("Average Rating", avg_rating)
col3.metric("Total Reviews", f"{total_reviews:,}")
col4.metric("Average Sentiment", avg_sentiment)
st.markdown("---")
st.markdown("## 📊 Market Overview")

# FIRST 2 CHARTS
col1, col2 = st.columns(2)

with col1:
    top_categories = (
        filtered_df["Category"]
        .value_counts()
        .head(10)
        .reset_index()
    )
    top_categories.columns = ["Category", "Count"]

    fig1 = px.bar(
        top_categories,
        x="Category",
        y="Count",
        title="Top 10 App Categories",
        color="Count",
        color_continuous_scale="Blues"
    )
    fig1.update_layout(
        xaxis_title="Category",
        yaxis_title="Count",
        xaxis_tickangle=-45
    )
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    fig2 = px.histogram(
        filtered_df,
        x="Rating",
        nbins=20,
        title="Rating Distribution",
        color_discrete_sequence=["#4CC9F0"]
    )
    fig2.update_layout(
        xaxis_title="Rating",
        yaxis_title="Frequency"
    )
    st.plotly_chart(fig2, use_container_width=True)

st.markdown(
    """
**Insights:**
- Most apps are concentrated in a few dominant categories, especially Games and Family.
- The rating distribution shows that a large number of apps are rated between 4.0 and 4.5, indicating generally positive app quality.
"""
)

st.markdown("---")

# -------------------------------
# USER SENTIMENT ANALYSIS
# -------------------------------
st.markdown("## 😊 User Sentiment Analysis")

col3, col4 = st.columns(2)

with col3:
    fig3 = px.histogram(
        filtered_df,
        x="Sentiment_Score",
        nbins=30,
        title="Sentiment Score Distribution",
        color_discrete_sequence=["#F72585"]
    )
    fig3.update_layout(
        xaxis_title="Sentiment Score",
        yaxis_title="Frequency"
    )
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    avg_sentiment_by_category = (
        filtered_df.groupby("Category")["Sentiment_Score"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    fig4 = px.bar(
        avg_sentiment_by_category,
        x="Category",
        y="Sentiment_Score",
        title="Top 10 Categories by Average Sentiment",
        color="Sentiment_Score",
        color_continuous_scale="RdPu"
    )
    fig4.update_layout(
        xaxis_title="Category",
        yaxis_title="Average Sentiment Score",
        xaxis_tickangle=-45
    )
    st.plotly_chart(fig4, use_container_width=True)

st.markdown(
    """
**Insights:**
- Most sentiment scores lean toward the positive side, suggesting overall favorable user feedback.
- Some categories receive consistently stronger sentiment, which may reflect better user satisfaction and experience.
"""
)

st.markdown("---")

# FREE VS PAID APP COMPARISON
st.markdown("## 💰 Free vs Paid App Comparison")

col5, col6 = st.columns(2)

with col5:
    fig5 = px.box(
        filtered_df,
        x="Type",
        y="Rating",
        title="App Ratings: Free vs Paid",
        color="Type",
        color_discrete_sequence=["#4361EE", "#F4A261"]
    )
    fig5.update_layout(
        xaxis_title="App Type",
        yaxis_title="Rating"
    )
    st.plotly_chart(fig5, use_container_width=True)

with col6:
    type_distribution = (
        filtered_df["Type"]
        .value_counts()
        .reset_index()
    )
    type_distribution.columns = ["Type", "Count"]

    fig6 = px.pie(
        type_distribution,
        names="Type",
        values="Count",
        title="App Type Distribution",
        hole=0.4,
        color_discrete_sequence=["#4361EE", "#F4A261"]
    )
    st.plotly_chart(fig6, use_container_width=True)

st.markdown(
    """
**Insights:**
- Free apps dominate the Play Store ecosystem by a very large margin.
- Paid apps often show slightly stronger median ratings, though free apps are far more common.
"""
)
st.markdown("---")
st.markdown("## 🤖 Machine Learning Insights")
# PREPARE DATA FOR ML

ml_df = filtered_df.copy()

features = ['Installs', 'Reviews', 'Size', 'Price', 'Sentiment_Score']
target = 'Rating'

ml_df = ml_df.dropna(subset=features + [target])

# Keep one row per app to reduce duplicate-review leakage
ml_df = ml_df.drop_duplicates(subset=['App'])

X = ml_df[features]
y = ml_df[target]

# Only run ML if enough rows exist
if len(ml_df) > 10:
    # TRAIN-TEST SPLIT
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # TRAIN MODELS
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_preds = lr_model.predict(X_test)

    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)

    # EVALUATE MODELS
    lr_r2 = r2_score(y_test, lr_preds)
    rf_r2 = r2_score(y_test, rf_preds)

    lr_mse = mean_squared_error(y_test, lr_preds)
    rf_mse = mean_squared_error(y_test, rf_preds)

    # SHOW METRICS
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Linear Regression R²", f"{lr_r2:.3f}")
        st.metric("Linear Regression MSE", f"{lr_mse:.3f}")

    with col2:
        st.metric("Random Forest R²", f"{rf_r2:.3f}")
        st.metric("Random Forest MSE", f"{rf_mse:.3f}")

    # MODEL COMPARISON CHART

    comparison_df = pd.DataFrame({
        "Model": ["Linear Regression", "Random Forest"],
        "R2 Score": [lr_r2, rf_r2]
    })

    fig_model = px.bar(
        comparison_df,
        x="Model",
        y="R2 Score",
        title="Model Performance Comparison",
        color="Model",
        color_discrete_sequence=["#4CC9F0", "#4361EE"]
    )

    st.plotly_chart(fig_model, use_container_width=True)

    # FEATURE IMPORTANCE
    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": rf_model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    fig_importance = px.bar(
        importance_df,
        x="Feature",
        y="Importance",
        title="Feature Importance (Random Forest)",
        color="Importance",
        color_continuous_scale="Viridis"
    )

    st.plotly_chart(fig_importance, use_container_width=True)

    st.markdown("""
**Insights:**
- Random Forest outperforms Linear Regression, indicating non-linear relationships in the data.
- The number of reviews is the most influential factor in predicting app ratings.
- Sentiment score significantly impacts ratings, reflecting the importance of user feedback.
- Installs and size have moderate influence, while price has minimal effect.
- Overall, popularity and user experience matter more than pricing in determining ratings.
""")

    st.markdown("---")

    # PREDICTION SECTION
    st.markdown("🔮 App Rating Prediction Tool")
    st.markdown("Enter app details to estimate a predicted rating using the Random Forest model.")

    col3, col4 = st.columns(2)

    with col3:
        installs_input = st.number_input("Installs", min_value=0, value=10000)
        reviews_input = st.number_input("Reviews", min_value=0, value=500)
        size_input = st.number_input("Size (MB)", min_value=0.0, value=20.0)

    with col4:
        price_input = st.number_input("Price ($)", min_value=0.0, value=0.0)
        sentiment_input = st.slider("Sentiment Score", -1.0, 1.0, 0.0)

    input_data = pd.DataFrame({
        'Installs': [installs_input],
        'Reviews': [reviews_input],
        'Size': [size_input],
        'Price': [price_input],
        'Sentiment_Score': [sentiment_input]
    })

    if st.button("Predict Rating"):
        prediction = rf_model.predict(input_data)[0]
        st.success(f"Predicted App Rating: ⭐ {prediction:.2f}")

else:
    st.warning("Not enough filtered data available to train and display machine learning models.")
