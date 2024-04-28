from datetime import datetime, time
import logging
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.logs_consola import setup_logging

# Setup logging
setup_logging()
logging = logging.getLogger(__name__)

try:
    import pandas as pd
    from typing import Tuple, Union, List
    import xgboost as xgb
except ImportError as e:
    logging.warning(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Some dependencies are missing. Please, install them to use the model. {e}")

PATH_MODEL = "model/challenge.json"
MORNING_MIN = time(5, 0)
MORNING_MAX = time(11, 59)
AFTERNOON_MIN = time(12, 0)
AFTERNOON_MAX = time(18, 59)
EVENING_MIN = time(19, 0)
EVENING_MAX = time(23, 59)
NIGHT_MIN = time(0, 0)
NIGHT_MAX = time(4, 59)

class DelayModel:

    def __init__(self):
        self._model = self.initialize_model()
        self.load_model_if_available()
        self.threshold = 0.7

    def initialize_model(self):
        """ Initialize a new XGBoost classifier model with predefined settings. """
        return xgb.XGBClassifier(random_state=1, learning_rate=0.01, scale_pos_weight=5.407)

    def load_model_if_available(self):
        """ Load the model from file if it exists, otherwise initialize a new model. """
        if os.path.exists(PATH_MODEL):
            self.load_existing_model(PATH_MODEL)
        else:
            logging.info(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} No trained model found. Initialized with a new model.")
            self.status_model = False

    def load_existing_model(self, filename):
        """ Load an existing model from a file. """
        try:
            self._model = xgb.XGBClassifier()
            self._model.load_model(filename)
            logging.info(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Loaded the trained model.")
            self.status_model = True
        except Exception as e:
            logging.error(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Error loading the model: {e}")
            self._model = None
    
    @staticmethod
    def get_period_day(date):
        date_time = datetime.strptime(date, '%Y-%m-%d %H:%M:%S').time()
        
        if MORNING_MIN < date_time < MORNING_MAX:
            return 'mañana'
        elif AFTERNOON_MIN < date_time < AFTERNOON_MAX:
            return 'tarde'
        elif (EVENING_MIN < date_time < EVENING_MAX) or (NIGHT_MIN < date_time < NIGHT_MAX):
            return 'noche'
        
    @staticmethod
    def is_high_season(fecha):
        fecha_año = int(fecha.split('-')[0])
        fecha = datetime.strptime(fecha, '%Y-%m-%d %H:%M:%S')
        range1_min = datetime.strptime('15-Dec', '%d-%b').replace(year = fecha_año)
        range1_max = datetime.strptime('31-Dec', '%d-%b').replace(year = fecha_año)
        range2_min = datetime.strptime('1-Jan', '%d-%b').replace(year = fecha_año)
        range2_max = datetime.strptime('3-Mar', '%d-%b').replace(year = fecha_año)
        range3_min = datetime.strptime('15-Jul', '%d-%b').replace(year = fecha_año)
        range3_max = datetime.strptime('31-Jul', '%d-%b').replace(year = fecha_año)
        range4_min = datetime.strptime('11-Sep', '%d-%b').replace(year = fecha_año)
        range4_max = datetime.strptime('30-Sep', '%d-%b').replace(year = fecha_año)
        
        if ((fecha >= range1_min and fecha <= range1_max) or 
            (fecha >= range2_min and fecha <= range2_max) or 
            (fecha >= range3_min and fecha <= range3_max) or
            (fecha >= range4_min and fecha <= range4_max)):
            return 1
        else:
            return 0
        
    @staticmethod
    def get_min_diff(data):
        fecha_o = datetime.strptime(data['Fecha-O'], '%Y-%m-%d %H:%M:%S')
        fecha_i = datetime.strptime(data['Fecha-I'], '%Y-%m-%d %H:%M:%S')
        min_diff = ((fecha_o - fecha_i).total_seconds())/60
        return min_diff

    def preprocess(self, data: pd.DataFrame, target_column: str = None) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """
        try:
            logging.info(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Preprocessing data")
            threshold_in_minutes = 15
            data['period_day'] = data['Fecha-I'].apply(self.get_period_day)
            data['high_season'] = data['Fecha-I'].apply(self.is_high_season)
            data['min_diff'] = data.apply(self.get_min_diff, axis=1)
            data['delay'] = data['min_diff'].apply(lambda x: 1 if x > threshold_in_minutes else 0)

            logging.info(f"Modelo nuevo: {self.status_model}")
            if self.status_model:
                importances_dict = self._model.get_booster().get_score(importance_type='weight')
                importances = {k: v for k, v in sorted(importances_dict.items(), key=lambda item: item[1], reverse=True)}
                top_10_features = list(importances.keys())[:10]
            else:
                top_10_features = ["OPERA_Latin American Wings", "MES_10", "MES_7", "OPERA_Grupo LATAM", "MES_6", "MES_4", "MES_8", "MES_12", "OPERA_Sky Airline", "TIPOVUELO_I"]
            
            logging.info(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Top 10 features: {top_10_features}")

            filtered_features = pd.DataFrame()
            for prefix in ['OPERA_', 'TIPOVUELO_', 'MES_']:
                dummies = pd.get_dummies(data[prefix.rstrip('_')], prefix=prefix)
                expected_cols = [col for col in top_10_features if col.startswith(prefix)]
                dummies = dummies.reindex(columns=expected_cols, fill_value=0)
                filtered_features = pd.concat([filtered_features, dummies], axis=1)

            if target_column:
                target = data[[target_column]]
                logging.info(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Data preprocessed with target")
                return filtered_features, target

            logging.info(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Data preprocessed without target")
            return filtered_features
        
        except Exception as e:
            logging.error(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Error preprocessing data: {e}")
            return None

    def fit(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        logging.info(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Fitting model")
        self._model.fit(features, target.values.ravel(), sample_weight=None)
        try:
            self._model.save_model(PATH_MODEL)
            logging.info(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Model saved")
        except Exception as e:
            logging.error(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Model not saved: {e}")

    def predict(self, features: pd.DataFrame) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.
        
        Returns:
            (List[int]): predicted targets.
        """
        logging.info(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Predicting")
        predictions = self._model.predict_proba(features)[:, 1]
        predictions = [1 if p > self.threshold else 0 for p in predictions]
        logging.info(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Predictions made")

        return predictions