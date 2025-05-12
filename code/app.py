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
                {"two_way": "2-Way Mid-Block (k=2)",
                "t_intersection": "T-Intersection (k=3)", 
                "four_way": "4-Way Intersection (k=4)"},
                selected="four_way"
            ),
            
            
            ui.input_file("image_file", "Upload Intersection Image:", 
                         accept=[".jpg", ".jpeg", ".png"]),
            
            ui.input_file("data_file", "Upload Vehicle Movement Data (csv/txt):", 
                         accept=[".csv",".txt"]),
            
            # ui.input_checkbox("use_gates_algorithm", "Use Gates Algorithm", value=True),
            
            ui.input_action_button("analyze_btn", "Run Analysis", 
                                  class_="btn-primary w-100"),
            
            ui.card(
                ui.card_header("Help"),
                "CSV should contain vehicle movement data with columns 'frame','vehicle_id','vehicle_type','a','b','c','X1','Y1','X2','Y2' in order for trajectories start and end points (Here a, b, c can be anything)."
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
        ),  

        ui.card(
            ui.card_header("Detailed Results"),
            ui.layout_column_wrap(
                ui.card(
                    ui.card_header("Gate Direction Mapping"),
                    ui.div(
                        ui.output_ui("dropdown_sets"),
                        # Recreated the button layout to align with dropdowns
                        ui.row(
                            ui.column(6, 
                                ui.input_action_button("reset_mapping", "Reset Mapping", 
                                                        class_="btn-sm btn-outline-danger")
                            ),
                            ui.column(6, 
                                ui.input_action_button("submit", "Submit Mapping", 
                                                        class_="btn-sm btn-outline-primary")
                            ),
                            class_="mt-1"
                        )
                    ),        
                ),
                
                ui.card(
                    ui.card_header("Mapping Results"),
                    ui.output_table("mapping_result_table"),
                    ui.div(
                        ui.output_ui("export_csv_div"),
                        class_="p-30"
                    )
                ),
                width=1/2
            ),    
        )    
    )
)

def server(input, output, session):
    # Reactive values to store data
    processed_data = reactive.value(None)
    cluster_results = reactive.value(None)
    movement_counts = reactive.value(None)
    csv_output = reactive.value(None)
    processed_image = reactive.value(None)
    gates_transitions = reactive.value(None)
    mapping_result = reactive.value(None)
    dir_csv_output = reactive.value(None)
    # Store selected gates
    selected_gates = reactive.value({})
    
    # Store selected directions
    selected_directions = reactive.value({})  

    # Reactive values to track states
    content_generated = reactive.value(False)

    # Process data when analyze button is clicked
    @reactive.effect
    @reactive.event(input.analyze_btn)
    def _():
        if not input.data_file() or not input.image_file():
            return
        
        # Save uploaded files to temporary locations for gates.py to process
        temp_csv = None
        temp_img = None

        # Reset selected gates when number of gates changes
        selected_gates.set({})
        selected_directions.set({})
        # Clear the mapping results when reset is clicked
        mapping_result.set(None)
        content_generated.set(False)
        
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
            # if input.use_gates_algorithm():
            try:
                
                # Determine k based on intersection type
                cluster_count = 3 if input.intersection_type() == "t_intersection" else 2 if input.intersection_type() == "two_way" else 4
    
                # Call analyze and retrieve image + data
                image_base64, movement_df, output = TrajectoryGatesCombined.analyze(
                    temp_csv_path, temp_img_path, cluster_count
                )
                # Save results to reactive values
                processed_image.set(image_base64)
                movement_counts.set(movement_df)
                csv_output.set(output)

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
    
    @reactive.effect
    @reactive.event(input.reset_mapping)
    def update_dropdown_sets():
        # Reset selected gates when number of gates changes
        selected_gates.set({})
        selected_directions.set({})
        # Clear the mapping results when reset is clicked
        mapping_result.set(None)
        content_generated.set(False)

    # Update selected gates whenever any gate dropdown changes
    @reactive.effect
    def update_selected_gates():
        num_gates = 3 if input.intersection_type() == "t_intersection" else 2 if input.intersection_type() == "two_way" else 4
        gate_selection_map = {}
        direction_selection_map = {}
        
        for i in range(1, num_gates + 1):
            gate_id = f"gate_{i}"
            direction_id = f"direction_{i}"
            
            # Only add to selection map if the gate has a non-empty value
            if hasattr(input, gate_id) and input[gate_id]() and input[gate_id]() != "":
                gate_selection_map[i] = input[gate_id]()
            
            # Store direction selections regardless of gate selection
            if hasattr(input, direction_id) and input[direction_id]():
                direction_selection_map[i] = input[direction_id]()
        
        selected_gates.set(gate_selection_map)
        selected_directions.set(direction_selection_map)

    # Output: Gate direction Mapping Dropdown
    @output
    @render.ui
    def dropdown_sets():
        # Create a set of dropdowns for each gate
        num_gates = 3 if input.intersection_type() == "t_intersection" else 2 if input.intersection_type() == "two_way" else 4
        gate_selection_map = selected_gates.get()
        direction_selection_map = selected_directions.get()
        dropdown_sets = []
        
        for i in range(1, num_gates + 1):
            gate_id = f"gate_{i}"
            direction_id = f"direction_{i}"
            

            # Get all currently selected gate values
            selected_values = list(gate_selection_map.values())
            
            # Create gate options
            gate_options = {"": "Select a gate"}  # Add empty option as default

            for j in range(1, num_gates + 1):
                gate_option = f"Gate {j}"
                # Include this gate if it's either:
                # 1. Currently selected by this dropdown set
                # 2. Not selected by any other dropdown set
                if (i in gate_selection_map and gate_selection_map[i] == gate_option) or gate_option not in selected_values:
                    gate_options[gate_option] = gate_option

                    
            # Create a row for each gate-direction pair
            dropdown_sets.append(
                ui.row(
                    ui.column(6, ui.input_select(
                        gate_id,
                        "Gate",
                        gate_options,
                        # Set the selected value if it exists in the selection map
                        selected=gate_selection_map.get(i, "")
                    )),
                    ui.column(6, ui.input_select(
                        direction_id,
                        "Direction",
                        {"East": "East", "West": "West", "North": "North", "South": "South"},
                        # Set the selected direction value if it exists
                        selected=direction_selection_map.get(i, "east")
                    )),
                    class_="mb-3"  # Add margin-bottom for spacing
                )
            )
        
        return ui.div(*dropdown_sets)
    
    # Process data when analyze button is clicked
    @reactive.effect
    @reactive.event(input.submit)
    def mapping():
        
        gate_dir_map = {}
        # if input.submit() or input_changed.get():
        
        # Skip processing if no data exists
        if movement_counts.get() is None or movement_counts.get().empty:
            ui.notification_show("No movement data available to map", type="warning")
            return
        df = movement_counts.get().copy()
        current_gates = 3 if input.intersection_type() == "t_intersection" else 2 if input.intersection_type() == "two_way" else 4
    
        for i in range(1, current_gates + 1):
            gate_value = input[f"gate_{i}"]()
            direction_value = input[f"direction_{i}"]()
            if gate_value and direction_value:
                gate_dir_map[gate_value] = direction_value
        
        if gate_dir_map:
            df['InDir'] = df['InGate'].map(gate_dir_map)
            df['OutDir'] = df['OutGate'].map(gate_dir_map)
            content_generated.set(True)
            
            mapping_result.set(df)

            gate_csv_output = csv_output.get()
            gate_csv_output['InGate'] = gate_csv_output['InGate'].map(gate_dir_map)
            gate_csv_output['OutGate'] = gate_csv_output['OutGate'].map(gate_dir_map)
            dir_csv_output.set(gate_csv_output)

            ui.notification_show("Mapping applied successfully", type="message")
        else:
            ui.notification_show("Please select gates and directions before mapping", type="warning")

    # Output: Cluster visualization
    @output
    @render.ui
    def mapping_result_table():
        if mapping_result() is None:
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
                        ui.tags.td(row['InDir']),
                        ui.tags.td(row['OutDir']),
                        ui.tags.td(row['Vehicle Count'])
                    )
                    for _, row in mapping_result().iterrows()
                ]
            )
        )

    # Dynamic Export CSV button creation 
    @output
    @render.ui
    def export_csv_div():
        if not content_generated.get():
            return
        return ui.div(
            ui.download_button(
                "export_csv",
                "Export CSV",
                class_="btn-sm btn-success mb-2"
            )
        )if content_generated.get() else ui.div()

    @render.download(filename="output_result.csv")
    def export_csv():
        result = dir_csv_output.get()
        yield result.to_csv(index=False)

app = App(app_ui, server)


# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=8003)