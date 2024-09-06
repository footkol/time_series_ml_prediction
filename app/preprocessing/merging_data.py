class DataMerger:
    def __init__(self, data):
        self.data = data

    def merge(self):
        # Logic to merge data
        merged_data = self.data  # Placeholder for the real merging logic
        return merged_data


def load_data():
    """Load all datasets."""
    train = pd.read_csv("../data/train.csv")
    stores = pd.read_csv("../data/stores.csv")
    holidays = pd.read_csv("../data/holidays_events.csv")
    oil = pd.read_csv("../data/oil.csv")
    transactions = pd.read_csv("../data/transactions.csv")
    test = pd.read_csv("../data/test.csv")
    return train, stores, holidays, oil, transactions, test

def preprocess_oil(oil):
    """Preprocess the oil dataset."""
    oil['dcoilwtico'] = oil['dcoilwtico'].bfill()
    return oil

def preprocess_holidays(holidays):
    """Preprocess the holidays dataset."""
    holidays = holidays[holidays['type'] != "Work Day"]
    holidays['events'] = holidays['type'].apply(lambda x: x if x == 'Event' else None)
    holidays['type'].replace('Event', np.nan, inplace=True)
    holidays['holiday'] = holidays['type'].notna()
    
    national_holidays = holidays[(holidays['locale'] != "Local") & (holidays['locale'] != "Regional")]
    national_holidays = national_holidays.rename(columns={'holiday': 'national_holiday'})
    
    state_holidays = holidays[(holidays['locale'] != "Local") & (holidays['locale'] != "National")]
    state_holidays = state_holidays.rename(columns={'holiday': 'state_holiday'})
    
    city_holidays = holidays[(holidays['locale'] != "National") & (holidays['locale'] != "Regional")]
    city_holidays = city_holidays.rename(columns={'holiday': 'city_holiday'})
    
    return national_holidays, state_holidays, city_holidays

def merge_datasets(df, stores, oil, transactions, national_holidays, state_holidays, city_holidays):
    """Merge all datasets into a final DataFrame."""
    df_stores = df.merge(stores, on='store_nbr', how='left')
    df_stores_oil = df_stores.merge(oil, on='date', how='left')
    df_stores_oil_trans = df_stores_oil.merge(transactions, on=['date', 'store_nbr'], how='left')
    
    df_stores_oil_trans_national_holidays = df_stores_oil_trans.merge(
        national_holidays[['national_holiday', 'date', 'events']], on='date', how='left'
    )
    
    df_stores_oil_trans_national_state_holidays = df_stores_oil_trans_national_holidays.merge(
        state_holidays[['state_holiday', 'date', 'locale_name']], on=['date', 'state'], how='left'
    ).drop('locale_name', axis=1)
    
    df_final = df_stores_oil_trans_national_state_holidays.merge(
        city_holidays[['city_holiday', 'date', 'locale_name']], on=['date', 'city'], how='left'
    ).drop('locale_name', axis=1)
    
    return df_final

def clean_data(df_final):
    """Clean and prepare the final DataFrame."""
    df_final = df_final.rename(columns={'dcoilwtico': 'oil_price'})
    pd.set_option('future.no_silent_downcasting', True)
    df_final = df_final.fillna({'national_holiday': False, 'state_holiday': False, 'city_holiday': False})
    df_final['events'] = df_final['events'].notna()
    return df_final

def save_data(df_final):
    """Save the final DataFrame to a CSV file."""
    df_final.to_csv('../data/train_merged.csv', index=False)

def main():
    train, stores, holidays, oil, transactions, test = load_data()
    oil = preprocess_oil(oil)
    national_holidays, state_holidays, city_holidays = preprocess_holidays(holidays)
    df = pd.concat([train, test], axis=0, ignore_index=True)
    df_final = merge_datasets(df, stores, oil, transactions, national_holidays, state_holidays, city_holidays)
    df_final = clean_data(df_final)
    save_data(df_final)

if __name__ == "__main__":
    main()
