class EDAProcessor:
    def __init__(self, data):
        self.data = data

    def perform_eda(self):
        # Logic for EDA or further preprocessing
        eda_data = self.data  # Placeholder for the real EDA logic
        return eda_data

def load_data():
    """Load the dataset and return as a DataFrame."""
    df = pd.read_csv("../data/train_merged.csv", low_memory=False)
    return df

def preprocess_data(df):
    """Preprocess the dataset including transformations and feature engineering."""
    columns_to_transform = ['family', 'city', 'state']
    df[columns_to_transform] = df[columns_to_transform].apply(lambda x: x.str.lower().str.replace(' ', '_'))
    
    df = df.drop('id', axis=1)
    
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['weekday'] = df['date'].dt.weekday
    
    df.set_index('date', inplace=True)
    df['oil_price'] = df['oil_price'].interpolate(method='time')
    df['transactions'] = df['transactions'].fillna(0)
    
    return df

def aggregate_data(df):
    """Perform various aggregations on the dataset."""
    df_aggregated_by_store = df.groupby('store_nbr')['sales'].sum().reset_index()
    df_aggregated_by_city = df.groupby('city')['sales'].sum().reset_index()
    df_aggregated_by_product = df.groupby('family')['sales'].sum().reset_index()
    df_aggregated_by_store_type = df.groupby('type')['sales'].sum().reset_index()
    df_aggregated_by_store_cluster = df.groupby('cluster')['sales'].sum().reset_index()
    
    quito_sales = df_aggregated_by_city[df_aggregated_by_city['city'] == 'quito']['sales'].sum()
    guayaquil_sales = df_aggregated_by_city[df_aggregated_by_city['city'] == 'guayaquil']['sales'].sum()
    
    return df_aggregated_by_store, df_aggregated_by_city, df_aggregated_by_product, df_aggregated_by_store_type, df_aggregated_by_store_cluster, quito_sales, guayaquil_sales

def detect_and_cap_outliers(df):
    """Detect and cap outliers in the dataset."""
    filtered_df = df.loc['2016-04-01':'2016-05-31']
    
    upper_threshold = 20000
    outlier_dates = filtered_df[filtered_df['sales'] > upper_threshold].index
    outlier_min = outlier_dates.min()
    outlier_max = outlier_dates.max()
    
    Q1 = filtered_df['sales'].quantile(0.25)
    Q3 = filtered_df['sales'].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    print(f"Upper bound is: {upper_bound:.2f}")
    print(f"Lower bound is: {lower_bound:.2f}")
    
    outliers_above_upper_bound = filtered_df[filtered_df['sales'] > upper_bound]
    if not outliers_above_upper_bound.empty:
        highest_non_outlier = filtered_df[filtered_df['sales'] <= upper_bound]['sales'].max()
        print(f"All outliers lie above: {highest_non_outlier}")
        print(f"Number of outliers above this threshold: {outliers_above_upper_bound.shape[0]}")
    else:
        print("No outliers above the upper bound.")
    
    percentage_of_outliers = len(outliers_above_upper_bound) / len(filtered_df['sales']) * 100
    
    threshold = 17500
    df.loc[(df.index >= outlier_min) & (df.index <= outlier_max) & (df['sales'] > threshold), 'sales'] = threshold
    
    return df

def adjust_sales_for_specific_date(df):
    """Adjust sales data for a specific date and product."""
    upper_threshold = 60000
    specific_date = '2016-10-07'
    
    df_filtered_part = df[(df['date'] == specific_date) & (df['sales'] > 20000)]
    
    store = 39
    product = 'meats'
    start_date = '2016-10-01'
    end_date = '2016-10-30'
    
    monthly_data = df[(df['store_nbr'] == store) & (df['family'] == product) & 
                      (df['date'] >= start_date) & (df['date'] <= end_date)]
    
    median_sales = monthly_data['sales'].median()
    print(median_sales)
    
    df.loc[(df['date'] == specific_date) & (df['sales'] > 20000), 'sales'] = median_sales
    
    df_filtered_part = df[(df['date'] == specific_date) & (df['sales'] > 20000)]
    
    return df, df_filtered_part

def process_oil_data():
    """Process the oil dataset."""
    oil = pd.read_csv("../data/oil.csv")
    oil['date'] = pd.to_datetime(oil['date'])
    oil.set_index('date', inplace=True)
    oil['dcoilwtico'] = oil['dcoilwtico'].interpolate(method='time')
    
    return oil

def finalize_data(df):
    """Finalize the data by renaming columns and dropping unnecessary ones."""
    df.rename(columns={'family': 'products', 'onpromotion': 'promo', 'national_holiday': 'holiday'}, inplace=True)
    
    columns_to_drop = ['state', 'type', 'cluster', 'transactions', 'state_holiday', 'city_holiday']
    df_final = df.drop(columns=columns_to_drop)
    
    df_final.to_csv('../data/train_post_eda.csv')
    
    return df_final

def split_and_save_data(df_final):
    """Split the data into training and test sets and save them."""
    df_final['date'] = pd.to_datetime(df_final['date'])
    
    split_date = '2017-08-16'
    
    df_train = df_final[df_final['date'] < split_date]
    df_test = df_final[df_final['date'] >= split_date]
    
    start_date = df_train['date'].min()
    end_date = df_train['date'].max()
    
    start_date_test = df_test['date'].min()
    end_date_test = df_test['date'].max()
    
    print("Start date:", start_date)
    print("End date:", end_date)
    print("Start date test set:", start_date_test)
    print("End date test set:", end_date_test)
    
    df_train.to_csv('../data/train_final.csv')
    df_test.to_csv('../data/test_final.csv')

def main():
    df = load_data()
    df = preprocess_data(df)
    aggregate_data(df)
    df = detect_and_cap_outliers(df)
    df, df_filtered_part = adjust_sales_for_specific_date(df)
    process_oil_data()
    df_final = finalize_data(df)
    split_and_save_data(df_final)

if __name__ == "__main__":
    main()
