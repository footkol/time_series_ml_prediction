app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Model API is running"}

@app.post("/predict/lstm/")
def predict_lstm(data: list):
    try:
        input_data = np.array(data).reshape(1, TIME_STEPS, len(data[0]))
        prediction = model_lstm.predict(input_data)
        return {"lstm_prediction": float(prediction[0][0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in LSTM prediction: {str(e)}")

@app.post("/predict/gru/")
def predict_gru(data: list):
    try:
        input_data = np.array(data).reshape(1, TIME_STEPS, len(data[0]))
        prediction = model_gru.predict(input_data)
        return {"gru_prediction": float(prediction[0][0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in GRU prediction: {str(e)}")

# Load and preprocess data
def load_and_preprocess_data():
    df = pd.read_csv("../data/train_final.csv", low_memory=False)
    df_test = pd.read_csv("../data/test_final.csv", low_memory=False)

    df = df.drop(columns=['date', 'index'])
    df = pd.get_dummies(df, dtype=int, sparse=True)
    
    df_test = df_test.drop(['date', 'index', 'sales'], axis=1)
    df_test = pd.get_dummies(df_test, dtype=int, sparse=True)

    transformed_sales = np.log1p(df['sales'])
    transformed_promo = np.log1p(df['promo'])
    df['promo'] = transformed_promo

    transformed_promo_test = np.log1p(df_test['promo'])
    df_test['promo'] = transformed_promo_test

    return df, df_test, transformed_sales


# Generate cyclical features
def generate_cyclical_features(df, col_name, period, start_num=0):
    kwargs = {
        f'sin_{col_name}': lambda x: np.sin(2*np.pi*(df[col_name]-start_num)/period),
        f'cos_{col_name}': lambda x: np.cos(2*np.pi*(df[col_name]-start_num)/period)
    }
    return df.assign(**kwargs).drop(columns=[col_name])


# Preprocessing features and splitting data
def split_data(df):
    df_feat_weekday = generate_cyclical_features(df, 'weekday', 6, 0)
    df_final = generate_cyclical_features(df_feat_weekday, 'month', 12, 1)
    
    y = df_final[['sales']]
    X = df_final.drop(columns=['sales'])
    
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, shuffle=False, random_state=1)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False, random_state=1)

    f_values, p_values = f_regression(X_train, y_train)
    selected_features = X_train.columns[p_values < 0.05]

    return X_train[selected_features], X_val[selected_features], X_test[selected_features], y_train, y_val, y_test


# Data scaling
def scale_data(X_train, X_val, X_test):
    scaler = StandardScaler()
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled.astype(np.float32), X_val_scaled.astype(np.float32), X_test_scaled.astype(np.float32), scaler


# Data generator for training
def data_generator(X, y, time_steps, batch_size):
    while True:
        for i in range(0, len(X) - time_steps, batch_size):
            X_batch, y_batch = [], []
            for j in range(batch_size):
                if i + j + time_steps < len(X):
                    X_batch.append(X[i + j:i + j + time_steps])
                    if i + j + time_steps < len(y):
                        y_batch.append(y.iloc[i + j + time_steps])
                    else:
                        print(f"Index out of bounds for y: {i + j + time_steps}")
            yield np.array(X_batch), np.array(y_batch)


# Create model (GRU/LSTM)
def create_model(units, m, time_steps, num_features):
    model = Sequential()
    model.add(m(units=units, return_sequences=True, input_shape=(time_steps, num_features)))
    model.add(Dropout(0.2))
    model.add(m(units=units))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(loss='mse', optimizer='adam')
    return model


# Fit model
def fit_model(model, train_gen, val_gen, steps_per_epoch, validation_steps):
    early_stop = EarlyStopping(monitor='val_loss', patience=10)
    history = model.fit(
        train_gen,
        epochs=10,
        validation_data=val_gen,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=[early_stop],
        shuffle=False
    )
    return history


# Test data generator
def data_generator_test(X_data, timesteps, batch_size):
    while True:
        for i in range(0, len(X_data) - timesteps, batch_size):
            X_batch = np.array([X_data[j:j+timesteps] for j in range(i, min(i+batch_size, len(X_data) - timesteps))])
            yield X_batch


# Model prediction
def prediction(model, test_gen):
    return model.predict(test_gen)


# Evaluate predictions
def evaluate_prediction(predictions, model_name):
    print(f'{model_name} predictions stats:')
    print(f'Mean of Predictions: {np.mean(predictions):.4f}')
    print(f'Standard Deviation of Predictions: {np.std(predictions):.4f}')
    print(f'Minimum of Predictions: {np.min(predictions):.4f}')
    print(f'Maximum of Predictions: {np.max(predictions):.4f}\n')


# Run ARIMA model for benchmarking
def run_arima(df):
    train_size = int(len(df) * 0.7)
    train, test = df['sales'][:train_size], df['sales'][train_size:]
    
    model = ARIMA(train, order=(5, 1, 0))
    model_fit = model.fit(disp=0)
    forecast = model_fit.forecast(steps=len(test))[0]
    mse = mean_squared_error(test, forecast)
    print(f'ARIMA Model MSE: {mse}')


