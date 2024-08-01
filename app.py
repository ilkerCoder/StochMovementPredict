from tkinter import messagebox
import tkinter as tk
import warnings
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

import sklearn
sklearn._config.set_config(transform_output="pandas")
warnings.filterwarnings("ignore")


class StockSectorsExperiment:
    """
    Bu sınıfın amacı, otomatik preprocessing pipeline'ı oluşturmak içindir.
    """

    def __init__(self, scaler_name: str = None, imputer_name: str = None):
        self.target = "Sector"
        self.create_pipeline(scaler_name=scaler_name,
                             imputer_name=imputer_name)
        self.target_encoder = {
            "Finance": 0,
            "Technology": 1,
            "Healthcare": 2
        }

    def scaler(self, name: str = None):
        if name == "StandardScaler":
            from sklearn.preprocessing import StandardScaler
            return StandardScaler()
        elif name == "MinMaxScaler":
            from sklearn.preprocessing import MinMaxScaler
            return MinMaxScaler()
        elif name == "RobustScaler":
            from sklearn.preprocessing import RobustScaler
            return RobustScaler()
        elif name is None:
            return None
        else:
            raise ValueError(
                "Lütfen geçerli bir scaler adı girin. (StandardScaler, MinMaxScaler, RobustScaler)")

    def imputer(self, name: str = None):
        if name == "DropImputer":
            from feature_engine.imputation import DropMissingData
            return DropMissingData()
        elif name == "MeanImputer":
            from feature_engine.imputation import MeanMedianImputer
            return MeanMedianImputer(imputation_method="mean")
        elif name == "MedianImputer":
            from feature_engine.imputation import MeanMedianImputer
            return MeanMedianImputer(imputation_method="median")
        elif name is None:
            return None
        else:
            raise ValueError(
                "Lütfen geçerli bir imputer adı girin. (SimpleImputer)")

    def create_pipeline(self, scaler_name: str = None, imputer_name: str = None):
        from sklearn.pipeline import Pipeline
        try:
            scaler = self.scaler(name=scaler_name)
            imputer = self.imputer(name=imputer_name)
        except ValueError as e:
            raise ValueError("Scaler veya imputer adı geçersiz.") from e
        if scaler is None and imputer is None:
            self.pipeline = None
        elif scaler is None:
            self.pipeline = Pipeline([
                (f"{imputer_name}_Imputer", imputer)
            ])
        elif imputer is None:
            self.pipeline = Pipeline([
                (f"{scaler_name}_Scaler", scaler)
            ])
        else:
            self.pipeline = Pipeline([
                (f"{imputer_name}_Imputer", imputer),
                (f"{scaler_name}_Scaler", scaler)
            ])


def drop_missing_features(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    df = df.copy()
    missing_values = df.isnull().mean()
    return df.drop(missing_values[missing_values > threshold].index, axis=1)


df = pd.read_csv("data/stock_sectors/feature_data.csv")

df.rename(columns={"Unnamed: 0": "Symbol"}, inplace=True)
# Özel karakterleri alt çizgiyle değiştirme
df.columns = df.columns.str.replace('[^a-zA-Z0-9_]', '_')

df = drop_missing_features(df)

example_prep_exp = StockSectorsExperiment(
    scaler_name="RobustScaler", imputer_name="DropImputer")
example_prep_exp.pipeline
X = df.drop(columns=[example_prep_exp.target, "Symbol"])
y = df[example_prep_exp.target]
y = y.map(example_prep_exp.target_encoder)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
X_train_prepared = example_prep_exp.pipeline.fit_transform(X_train)
X_test_prepared = example_prep_exp.pipeline.transform(X_test)
y_train_prepared = y_train[y_train.index.isin(X_train_prepared.index)]
y_test_prepared = y_test[y_test.index.isin(X_test_prepared.index)]


xgb_model = XGBClassifier()

xgb_model.fit(X_train_prepared, y_train_prepared)

y_pred_xgb = xgb_model.predict(X_test_prepared)
accuracy = accuracy_score(y_test_prepared, y_pred_xgb)

print("XGBoost modeli ile elde edilen doğruluk:", accuracy)


def predict_and_compare():
    try:
        # Model tahmini
        y_pred_xgb = xgb_model.predict(X_test_prepared)

        # Kullanıcı tahminini al
        user_input = user_prediction.get()
        if not user_input.isdigit():
            raise ValueError("Tahmin sayı olmalıdır.")

        user_pred = int(user_input)

        # Sonuçları karşılaştır
        model_accuracy = accuracy_score(y_test_prepared, y_pred_xgb)
        messagebox.showinfo("Sonuçlar", f"Modelin doğruluğu: {model_accuracy:.2f}\n"
                            f"Kullanıcı tahmini: {user_pred}")
    except Exception as e:
        messagebox.showerror("Hata", str(e))


# Tkinter arayüzü
root = tk.Tk()
root.title("Stock Sectors Prediction")

frame = tk.Frame(root)
frame.pack(padx=10, pady=10)

tk.Label(frame, text="Kullanıcı Tahmini (0: Finance, 1: Technology, 2: Healthcare):").pack()
user_prediction = tk.Entry(frame)
user_prediction.pack()

predict_button = tk.Button(
    frame, text="Tahmin Yap ve Karşılaştır", command=predict_and_compare)
predict_button.pack()

root.mainloop()
