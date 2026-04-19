import onecode
import rasterio
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error


class Prediction:
    def __init__(self,dataset, mnt, slope):
        self.scaler = None
        self.y_test = None
        self.y_train = None
        self.X_test = None
        self.X_train = None
        self.best_model = None
        self.features = None
        self.target = None
        self.mnt = mnt
        self.slope = slope
        self.input_data = None
        self.element_of_interest = None
        self.gdf = dataset

        self.user_input_data()


    def user_input_data(self):

        self.element_of_interest = onecode.text_input(
            key="Chemical_Element",
            value="V",
            label="Enter the chemical element of interest",
            max_chars=30,
            placeholder="Au"
        )

        input_X = onecode.number_input(
            key="Coordinate_X",
            value=123456,
            label="Enter the coordinates X",
            min=0,
            max=None,
            step=1
        )

        input_Y= onecode.number_input(
            key="Coordinate_Y",
            value=987654,
            label="Enter the coordinates Y",
            min=0,
            max=None,
            step=1
        )

        input_elevation = onecode.number_input(
            key="Elevation",
            value=450,
            label="Enter the elevation value",
            min=0,
            max=None,
            step=1
        )

        input_slope = onecode.number_input(
            key="slope",
            value=12.5,
            label="Enter the slope value",
            min=0,
            max=None,
            step=1
        )

        self.input_data = [input_slope, input_elevation, input_X, input_Y]

    def start_predicton(self):
        """Start the prediction process."""

        self.data_preparation()
        self.best_model_search()
        self.predict_report()

    def convert_columns_to_ppm(self, p_df, el):
        """Converts all values to ppm"""

        new_df = p_df.copy()
        if el.endswith("ppb"):
            new_df[f"{el.split('_')[0]}_ppm"] = p_df[el] / 1000

        elif el.endswith("pct"):
            new_df[f"{el.split('_')[0]}_ppm"] = p_df[el] * 10000

        return new_df

    def target_checking(self, p_target, p_features):
        """ Check if input target is valid"""

        cols_targets = [elem for elem in self.gdf.columns if elem not in p_features]

        if p_target.lower() in cols_targets:
            return p_target.lower()

        else:
            for tg in cols_targets:

                if p_target.lower() in tg.lower():
                    return tg

            raise ValueError(f"{p_target} Chemical element not found")

    def data_preparation(self):
        """Data preparation for prediction"""

        onecode.Logger.info("data_preparation...")
        # =============Data preparation============

        coords = [(geom.x, geom.y) for geom in self.gdf.geometry]

        elevations = []
        for val in self.mnt.sample(coords):
            elevations.append(val[0])

        rows, cols = rasterio.transform.rowcol(self.mnt.transform,
                            [pt[0] for pt in coords],
                            [pt[1] for pt in coords])
        slope_values = self.slope[rows, cols]

        self.gdf["elevation"] = elevations
        self.gdf["slope"] = slope_values

        self.features = ["slope", "elevation", "X", "Y"]

        element_of_interest = self.target_checking(self.element_of_interest, self.features)
        self.gdf = self.convert_columns_to_ppm(self.gdf, element_of_interest)
        self.target = f"{self.element_of_interest.split('_')[0]}_ppm"

        onecode.Logger.info("Finished data preparation")

    def best_model_search(self):
        """Search the best parameters for the model"""

        onecode.Logger.info("best model search...")
        # =========== Search of best model (parameters) ================

        X = self.gdf[self.features]
        y = self.gdf[self.target]

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        param_grid = {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2"]
        }

        rf = RandomForestRegressor(random_state=42)

        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            cv=5,
            scoring="r2",
            n_jobs=-1,
            verbose=2
        )
        grid_search.fit(self.X_train, self.y_train)

        self.best_model = grid_search.best_estimator_

        print("Best parameters:", grid_search.best_params_)

        onecode.Logger.info("Finished best model search")


    def predict_ppm(self):
        """ input_data = [slope, elevation, X, Y]
            Return prediction
        """
        if self.validate_input(self.input_data):
            X_scaled = self.scaler.transform([self.input_data])
            pred = self.best_model.predict(X_scaled)

            return pred[0]

        else:
            raise ValueError("Error in data")


    def validate_input(self,data):
        """Return True if input data is valid"""

        if len(data) != 4:
            raise ValueError("Needs 4 parameters")

        if any(pd.isna(data)):
            raise ValueError("Missing values")

        return True


    def model_quality_report(self,r2_train, r2_test):
        """ Return Report about the model quality """

        gap = r2_train - r2_test

        if r2_test < 0.5:
            return "[Weak model predictions are unreliable]"

        elif gap > 0.2:
            return "[Overfitting detected use predictions with caution]"

        elif r2_test >= 0.7:
            return "[Reliable model]"

        else:
            return "[Acceptable model but needs improvement]"

    def predict_report(self):
        """Save prediction results """

        onecode.Logger.info("predicting...")
        # ========== Prediction ===============

        y_train_pred = self.best_model.predict(self.X_train)
        y_test_pred = self.best_model.predict(self.X_test)

        r2_train = r2_score(self.y_train, y_train_pred)
        r2_test = r2_score(self.y_test, y_test_pred)

        rmse = np.sqrt(mean_squared_error(self.y_test, y_test_pred))

        print("R2 train:", r2_train)
        print("R2 test:", r2_test)
        print("RMSE:", rmse)
        print(self.model_quality_report(r2_train, r2_test), "\n")

        slope, elevation, X, Y = self.input_data
        prediction = self.predict_ppm()

        print(f"Slope: {slope}, Elevation: {elevation}, X: {X}, Y: {Y}")
        print(f"Concentration predicted ({self.target.split('_')[0]}) : {prediction:.2f} ppm")

        # =========== Saving result ==================

        row = {
            "slope": slope,
            "elevation": elevation,
            "X": X,
            "Y": Y,
            "prediction_ppm": prediction,
            "r2_train": r2_train,
            "r2_test": r2_test,
            "rmse": rmse,
            "model_quality": self.model_quality_report(r2_train, r2_test),
            "element": self.target.split('_')[0]
        }

        df = pd.DataFrame([row])
        output= onecode.file_output(
            key="output_prediction",
            value="output/prediction/predictions.csv",
            make_path=True
        )
        df.to_csv(output, index=False, sep=";", encoding="utf-8")

        onecode.Logger.info("Finish predicting")