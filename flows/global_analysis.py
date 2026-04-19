import rasterio
import geopandas as gpd
from matplotlib import pyplot as plt
from scipy.ndimage import binary_opening
import folium
from folium.plugins import Draw, MeasureControl
from folium.raster_layers import ImageOverlay
import numpy as np
from pykrige.ok import OrdinaryKriging
from sklearn.cluster import DBSCAN
import onecode


class GlobalAnalysis:

    def __init__(self,gdf_filename, mnt_filename,):

        self.mnt_path = None
        self.gdf_path = None
        self.kiging_results = None
        self.gdf_analyzed = None
        self.concentrated_elements = None
        self.hillshade_stretch = None
        self.mnt = None
        self.gdf = None
        self.distance = None
        self.number_of_points = None

        self.import_data(gdf_filename, mnt_filename)

    def analyze(self):

        self.clustering_input_parameter()
        self.mnt_procesing()
        self.geochemistry_anomalies()
        self.kriging()
        self.map_anomalie()
        self.interactive_map()

    def import_data(self,gdf_filename,mnt_filename):

        onecode.Logger.info("Importing data...")

        self.gdf =gpd.read_file(gdf_filename)

        self.mnt=rasterio.open(mnt_filename)

    def clustering_input_parameter(self):

        self.distance = onecode.number_input(
            key="Minimum_Distance",
            value=1000,
            label="Enter the minimum distance",
            min=0,
            max=None,
            step=1
        )

        self.number_of_points = onecode.number_input(
            key="Number_of_Points",
            value=10,
            label="Enter the number of points",
            min=0,
            max=None,
            step=1
        )

        onecode.Logger.info("Finish Importing data ")

    def mnt_procesing(self):

        onecode.Logger.info("Mnt Procesing...")

        # --------Hillshade and slope extraction-----
        azimuth = 150  # for the hillshade
        altitude = 45  # for the hillshade

        filtered_topo = self.mnt.read(1)
        res_x =  self.mnt.transform[0]
        res_y = - self.mnt.transform[4]

        dy, dx = np.gradient(filtered_topo, res_y, res_x)

        slope = np.sqrt(dx ** 2 + dy ** 2)  # slope calculation
        aspect = np.arctan2(dy, dx)

        azimuth_rad = np.radians(azimuth)
        altitude_rad = np.radians(altitude)

        hillshade = 255 * (  # hillshade calculation
                np.cos(altitude_rad) * np.cos(slope) +
                np.sin(altitude_rad) * np.sin(slope) *
                np.cos(azimuth_rad - aspect)
        )

        # Contrast improvement
        p2, p98 = np.percentile(hillshade, (2, 98))
        self.hillshade_stretch = np.clip((hillshade - p2) / (p98 - p2), 0, 1)
        p2, p98 = np.percentile(slope, (2, 98))
        slope_stretch = np.clip((slope - p2) / (p98 - p2), 0, 1)

        # Save slope as DTM
        output_slope_file = onecode.file_output(
            key="slope",
            value="output/geotiff/slope.tif",
            make_path=True
        )
        self.save_geotiff(
            slope_stretch,
            self.gdf.geometry.x.values,
            self.gdf.geometry.y.values,
            output_slope_file
        )

        onecode.Logger.info("Finish mnt procesing")

    def save_geotiff(self,e_zi, e_x, e_y, output_path, crs="EPSG:2154"):

        height, width = e_zi.shape

        # transformation spatiale
        transform = rasterio.transform.from_bounds(
            e_x.min(), e_y.min(),
            e_x.max(), e_y.max(),
            width, height
        )

        with rasterio.open(
                output_path,
                'w',
                driver='GTiff',
                height=height,
                width=width,
                count=1,
                dtype=e_zi.dtype,
                crs=crs,
                transform=transform
        ) as dst:
            dst.write(e_zi, 1)

    def geochemistry_anomalies(self):

        onecode.Logger.info("geochemistry anomalies...")

        # ===== Geochemistry anomalies calculation =======

        xmin, ymin, xmax, ymax = self.mnt.bounds
        cols_exclues = ["geometry", "INDC_B", "X", "Y"]
        elements = [col for col in self.gdf.columns if col not in cols_exclues]
        self.gdf_analyzed = self.gdf.copy()

        for element in elements:
            mean = self.gdf_analyzed[element].mean()
            std = self.gdf_analyzed[element].std()

            self.gdf_analyzed[f"{element}_anomaly"] = (
                    self.gdf_analyzed[element] > (mean + 2 * std)
            )

        # ============ Clustering =====================

        results = {}

        for el in elements:
            mask = self.gdf_analyzed[f"{el}_anomaly"]
            points = self.gdf_analyzed[mask].copy()

            if len(points) < 2:
                continue

            coords = np.array([(geom.x, geom.y) for geom in points.geometry])

            model = DBSCAN(
                eps=self.distance,
                min_samples=self.number_of_points
            )
            labels = model.fit_predict(coords)
            points["cluster"] = labels
            # Noise management
            points_clustered = points[points["cluster"] != -1].copy()

            results[el] = points_clustered

        # ============= Significant anomalies plotting ==============

        fig, ax = plt.subplots(figsize=(10, 8))

        # background hillshade
        ax.imshow(
            self.hillshade_stretch,
            cmap="gray",
            extent=[xmin, xmax, ymin, ymax],
            origin="upper"
        )

        # show clusterize points
        self.concentrated_elements = []
        for el, df in results.items():
            if not df.empty:
                self.concentrated_elements.append(el)
                df.plot(
                    ax=ax,
                    markersize=30,
                    label=el
                )
        print(f"significant elements : {self.concentrated_elements}")
        if not self.concentrated_elements:
            raise ValueError("No elements found for these input parameters try changing parameters")

        plt.legend(
            loc="center left",
            bbox_to_anchor=(1, 0.5)
        )
        plt.title("Significant Anomalies (clusters)")
        output_file = onecode.file_output(
            key="scatter_map_png",
            value="output/maps/scatter_map.png",
            make_path=True
        )
        plt.savefig(output_file)
        plt.close()

        onecode.Logger.info("Finish geochemistry anomalies")

    def kriging(self):
        """ """
        onecode.Logger.info("Kriging...")

        # =========== kriging of anomalies ==================

        self.kiging_results = {}

        x = self.gdf_analyzed.geometry.x.values
        y = self.gdf_analyzed.geometry.y.values

        gridx = np.linspace(x.min(), x.max(), 200)
        gridy = np.linspace(y.min(), y.max(), 200)

        for  el in self.concentrated_elements:
            z = self.gdf_analyzed[f"{el}_anomaly"].astype(float).values
            # Kriging
            OK = OrdinaryKriging(
                x, y, z,
                variogram_model="spherical"
            )

            zi, ss = OK.execute("grid", gridx, gridy)
            self.kiging_results[el] = zi

            # Saving

            fig, ax = plt.subplots(figsize=(8, 6))

            # Background hillshade
            ax.imshow(
                self.hillshade_stretch,
                cmap="gray",
                extent=[x.min(), x.max(), y.min(), y.max()],
                origin="upper"
            )

            im = ax.imshow(
                zi,
                extent=[x.min(), x.max(), y.min(), y.max()],
                origin="lower",
                cmap="viridis",
                alpha=0.6
            )

            ax.set_title(f"Krigeage indicatrice - {el}")
            plt.colorbar(im, ax=ax)

            output_file = onecode.file_output(
                key=f"maps_{el}",
                value=f"output/maps/{el}.png",
                make_path=True
            )
            plt.savefig(output_file,dpi=300, bbox_inches='tight')
            plt.close(fig)

            # Save as DTM
            output_dtm_file = onecode.file_output(
                key=f"maps_dtm_{el}",
                value=f"output/geotiff/{el}.tif",
                make_path=True
            )
            self.save_geotiff(
                zi,
                x,
                y,
                output_dtm_file
            )

        onecode.Logger.info("Finish kriging")


    def map_anomalie(self):
        """"""

        onecode.Logger.info("saving map_anomalie...")
        plt.figure(figsize=(10, 8))
        dtm_anomaly = np.zeros_like(list(self.kiging_results.values())[0])

        x = self.gdf_analyzed.geometry.x.values
        y = self.gdf_analyzed.geometry.y.values

        for el, zi in self.kiging_results.items():
            mask = zi >= np.percentile(zi, 98)
            mask_clean = binary_opening(mask, structure=np.ones((3, 3)))
            dtm_anomaly += mask_clean.astype(int)

            plt.imshow(
                mask_clean,
                extent=(x.min(), x.max(), y.min(), y.max()),
                origin='lower',
                cmap='hot'
            )
            # contours
            plt.contour(
                mask.astype(int),
                levels=[0.5],
                extent=[x.min(), x.max(), y.min(), y.max()],
                colors='yellow',
                linewidths=1.5
            )

        plt.axis('off')

        ouput_map_file = onecode.file_output(
            key=f"map_anomalie_zone",
            value="output/maps/map_anomalie_zone.png",
            make_path=True
        )
        plt.savefig(ouput_map_file, bbox_inches='tight', pad_inches=0)
        plt.close()

        # Save as DTM
        ouput_dtm_file = onecode.file_output(
            key=f"map_dtm_anomalie_zone",
            value="output/geotiff/map_anomalie.tif",
            make_path=True
        )
        self.save_geotiff(
            np.flipud(dtm_anomaly.astype("float32")),
            x,
            y,
            ouput_dtm_file
        )

        onecode.Logger.info("Finish map_anomalie")

    def interactive_map(self):
        """"""

        onecode.Logger.info("Starting interactive map...")

        # ============= Interactive Maps ===========================

        # REPROJECTION
        gdf_proj = self.gdf_analyzed.to_crs(epsg=2154)
        gdf_wgs84 = self.gdf_analyzed.to_crs(epsg=4326)
        # coords
        x = gdf_proj.geometry.x.values
        y = gdf_proj.geometry.y.values

        # Create maps
        gridx = np.linspace(x.min(), x.max(), 200)
        gridy = np.linspace(y.min(), y.max(), 200)

        center = [gdf_wgs84.geometry.y.mean(), gdf_wgs84.geometry.x.mean()]

        m = folium.Map(location=center, zoom_start=10, control_scale=True)

        folium.TileLayer("OpenStreetMap").add_to(m)
        folium.TileLayer("CartoDB positron").add_to(m)

        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri",
            name="Satellite"
        ).add_to(m)

        for el in self.concentrated_elements:

            fig, ax = plt.subplots(figsize=(6, 6))

            ax.imshow(
                self.kiging_results[el],
                cmap="viridis",
                origin="lower",
                vmin=0, vmax=1
            )

            ax.axis("off")
            img_path = f"data/outputs/temp/{el}.png"
            output_file = onecode.file_output(
                key=f"temp_{el}",
                value=f"temp/{el}.png",
                make_path=True
            )
            plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
            plt.close()

            # ----- add raster -----
            folium.raster_layers.ImageOverlay(
                image=img_path,
                bounds=[
                    [gdf_wgs84.geometry.y.min(), gdf_wgs84.geometry.x.min()],
                    [gdf_wgs84.geometry.y.max(), gdf_wgs84.geometry.x.max()]
                ],
                opacity=0.8,
                name=f"Krigeage {el}",
                show=False
            ).add_to(m)

            # ----- interactive points -----
            fg = folium.FeatureGroup(name=f"Points {el}", show=False)

            for _, row in gdf_wgs84.iterrows():
                val = row[f"{el}_anomaly"]

                popup_text = f"""
                <b>Élément :</b> {el}<br>
                <b>Valeur brute :</b> {row[el]:.2f}<br>
                <b>Anomalie :</b> {val}
                """

                folium.CircleMarker(
                    location=[row.geometry.y, row.geometry.x],
                    radius=4,
                    color="red" if val else "blue",
                    fill=True,
                    fill_opacity=0.7,
                    tooltip=f"{el}: {val}",
                    popup=popup_text,
                ).add_to(fg)

            fg.add_to(m)

        folium.raster_layers.ImageOverlay(
            image=self.hillshade_stretch,
            bounds=[
                [gdf_wgs84.geometry.y.min(), gdf_wgs84.geometry.x.min()],
                [gdf_wgs84.geometry.y.max(), gdf_wgs84.geometry.x.max()]],
            opacity=0.6,
            name=f"Hillshade",
            show=False
        ).add_to(m)

        ImageOverlay(
            image="data/outputs/output/maps/map_anomalie_zone.png",
            bounds=[
                [gdf_wgs84.geometry.y.min(), gdf_wgs84.geometry.x.min()],
                [gdf_wgs84.geometry.y.max(), gdf_wgs84.geometry.x.max()]
            ],
            opacity=0.6,
            name=f"Map Anomalie",
        ).add_to(m)

        # Interactive tools
        Draw().add_to(m)
        MeasureControl().add_to(m)

        # layer control
        folium.LayerControl(collapsed=False).add_to(m)

        # save

        output_map_of_chemical_elements_file = onecode.file_output(
            key="map_of_chemical_elements",
            value="output/global_interactive_map/map_of_chemical_elements.html",
            make_path=True
        )
        m.save(output_map_of_chemical_elements_file)

        onecode.Logger.info("Finish interactive map")


