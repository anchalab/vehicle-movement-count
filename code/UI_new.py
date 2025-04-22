import sys
from shiny import App, reactive, render, ui
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from PIL import Image
import io
import base64
import tempfile
import os
import TrajectoryGatesCombined  # Import the gates module

app_ui = ui.page_fluid(
    ui.h1("Intersection Traffic Analysis Dashboard", class_="text-center"),
    ui.p("Analyze vehicle movement patterns at intersections using k-means clustering", class_="text-center"),
    
    ui.layout_sidebar(
        ui.sidebar(
            ui.h4("Inputs"),
            ui.input_radio_buttons(
                "intersection_type",
                "Intersection Type:",
                {"t_intersection": "T-Intersection (k=3)", 
                 "four_way": "4-Way Intersection (k=4)"},
                selected="four_way"
            ),
            
            ui.input_file("image_file", "Upload Intersection Image:", 
                         accept=[".jpg", ".jpeg", ".png"]),
            
            ui.input_file("data_file", "Upload Vehicle Movement Data (CSV):", 
                         accept=[".csv",".txt"]),
            
            ui.input_checkbox("use_gates_algorithm", "Use Gates Algorithm", value=True),
            
            ui.input_action_button("analyze_btn", "Run Analysis", 
                                  class_="btn-primary w-100"),
            
            ui.card(
                ui.card_header("Help"),
                "CSV should contain vehicle movement data with x1, y1, x2, y2 columns for start and end coordinates."
            ),
            width=300
        ),
        
        ui.card(
            ui.card_header("Analysis Results"),
            ui.layout_column_wrap(
                ui.card(
                    ui.card_header("Cluster Visualization"),
                    ui.output_ui("cluster_image"),
                ),
                ui.card(
                    ui.card_header("Movement Summary"),
                    ui.output_table("movement_table"),
                ),
                width=1/2
            ),
            
            ui.card(
                ui.card_header("Detailed Results"),
                ui.navset_tab(
                    ui.nav_panel("Cluster Statistics", 
                           ui.output_plot("cluster_stats")),
                    ui.nav_panel("Raw Data", 
                           ui.output_table("raw_data_table")),
                    ui.nav_panel("Gates Transitions", 
                           ui.output_table("gates_transitions"))
                )
            )
        )
    )
)

def server(input, output, session):
    # Reactive values to store data
    processed_data = reactive.value(None)
    cluster_results = reactive.value(None)
    movement_counts = reactive.value(None)
    processed_image = reactive.value(None)
    gates_transitions = reactive.value(None)
    
    # Process data when analyze button is clicked
    @reactive.effect
    @reactive.event(input.analyze_btn)
    def _():
        if not input.data_file() or not input.image_file():
            return
        
        # Save uploaded files to temporary locations for gates.py to process
        temp_csv = None
        temp_img = None
        
        try:
            # Save CSV to temp file
            temp_csv = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
            temp_csv_path = temp_csv.name
            with open(input.data_file()[0]["datapath"], 'rb') as src_file:
                temp_csv.write(src_file.read())
            temp_csv.close()
            
            # Save image to temp file
            temp_img = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            temp_img_path = temp_img.name
            with open(input.image_file()[0]["datapath"], 'rb') as src_file:
                temp_img.write(src_file.read())
            temp_img.close()
            
            # Run gates analysis if checkbox is checked
            if input.use_gates_algorithm():
                try:
                    
                    # Determine k based on intersection type
                    cluster_count = 3 if input.intersection_type() == "t_intersection" else 4
                    # Call gates.analyze with the temp file paths
                    # TrajectoryGatesCombined.analyze(temp_csv_path, temp_img_path, cluster_count)
                    # Call analyze and retrieve image + data
                    image_base64, movement_df = TrajectoryGatesCombined.analyze(
                        temp_csv_path, temp_img_path, cluster_count
                    )
                    # Save results to reactive values
                    processed_image.set(image_base64)
                    movement_counts.set(movement_df)

                except Exception as e:
                    print(f"Error in gates algorithm: {e}")
                    gates_transitions.set(pd.DataFrame({
                        "Error": [f"Failed to run gates analysis: {str(e)}"]
                    }))
        
        finally:
            # Clean up temporary files
            if temp_csv and os.path.exists(temp_csv_path):
                os.unlink(temp_csv_path)
            if temp_img and os.path.exists(temp_img_path):
                os.unlink(temp_img_path)


    # Output: Cluster visualization
    @output
    @render.ui
    def cluster_image():
        if processed_image() is None:
            return ui.p("Run analysis to see cluster visualization")
        
        return ui.tags.img(
            src=f"data:image/jpg;base64,{processed_image()}",
            style="width: 100%; max-height: 400px; object-fit: contain;"
        )
    # Output: Cluster visualization
    @output
    @render.ui
    def movement_table():
        if movement_counts() is None:
            return None
        
        return ui.tags.table(
            ui.tags.thead(
                ui.tags.tr(
                    ui.tags.th("InGate"),
                    ui.tags.th("OutGate"),
                    ui.tags.th("Vehicle Count")
                )
            ),
            ui.tags.tbody(
                [
                    ui.tags.tr(
                        ui.tags.td(row['InGate']),
                        ui.tags.td(row['OutGate']),
                        ui.tags.td(row['Vehicle Count'])
                    )
                    for _, row in movement_counts().iterrows()
                ]
            )
        )
    
    # Output: Cluster statistics
    @output
    @render.plot
    def cluster_stats():
        if processed_data() is None:
            return None
        
        data = processed_data()
        
        # Create a summary of vehicles per cluster
        cluster_summary = data.groupby('start_cluster').size().reset_index(name='count')
        cluster_summary['cluster'] = 'Cluster ' + (cluster_summary['start_cluster'] + 1).astype(str)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(cluster_summary['cluster'], cluster_summary['count'], color='skyblue')
        ax.set_title('Vehicles per Cluster')
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Count')
        
        return fig
    
app = App(app_ui, server)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8002)