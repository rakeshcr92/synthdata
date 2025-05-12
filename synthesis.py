
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer, TVAESynthesizer, GaussianCopulaSynthesizer

def recommend_model(df):
    numeric_ratio = len(df.select_dtypes(include=['int64', 'float64']).columns) / df.shape[1]
    cat_ratio = len(df.select_dtypes(include=['object', 'category']).columns) / df.shape[1]

    if cat_ratio > 0.6:
        return "CTGAN"
    elif 0.3 < cat_ratio <= 0.6:
        return "TVAE"
    elif numeric_ratio > 0.8:
        return "GaussianCopula"
    else:
        return "TVAE"

def generate_synthetic_data(df, model_name="Auto", privacy_mode=False):
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=df)

    if model_name == "Auto":
        model_name = recommend_model(df)

    epochs = 10 if not privacy_mode else 3
    batch_size = 500 if not privacy_mode else 1000

    if model_name == "CTGAN":
        model = CTGANSynthesizer(metadata, epochs=epochs, batch_size=batch_size)
    elif model_name == "TVAE":
        model = TVAESynthesizer(metadata, epochs=epochs, batch_size=batch_size)
    elif model_name == "GaussianCopula":
        model = GaussianCopulaSynthesizer(metadata)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    if len(df) > 2000:
        df = df.sample(2000, random_state=42)

    model.fit(df)
    synthetic_data = model.sample(num_rows=len(df))
    return synthetic_data
