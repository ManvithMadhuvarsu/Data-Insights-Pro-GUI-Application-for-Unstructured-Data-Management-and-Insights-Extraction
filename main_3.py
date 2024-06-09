import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import os
import seaborn as sns

class DataInsightPro(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Data Insight Pro")
        self.state("zoomed")  # Open the app in full screen

        # Define visualization options
        self.visualization_options = ["All", "Histogram", "Bar Plot", "Pie Chart", "Scatter Plot", "Line Plot", "Heat Map", "Box Plot"]

        # Create main frame and scrollbar
        self.main_frame = tk.Frame(self)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.scrollbar = ttk.Scrollbar(self.main_frame, orient=tk.VERTICAL)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.canvas = tk.Canvas(self.main_frame, yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.scrollbar.config(command=self.canvas.yview)

        # Create a frame to hold the GUI elements
        self.gui_frame = tk.Frame(self.canvas)
        self.canvas.create_window((self.winfo_width() // 2, self.winfo_height() // 2), window=self.gui_frame, anchor=tk.CENTER)

        # Create application name label
        self.app_name_label = tk.Label(self.gui_frame, text="Data Insight Pro: Unstructured Data Management and Insights Extraction", font=("Arial", 24))
        self.app_name_label.grid(row=0, column=0, columnspan=6, pady=(20, 0))

        # Create file upload button
        self.upload_button = tk.Button(self.gui_frame, text="Upload File", command=self.upload_file)
        self.upload_button.grid(row=1, column=0, padx=10, pady=20)

        # Create preprocess data button
        self.preprocess_button = tk.Button(self.gui_frame, text="Preprocess Data", command=self.preprocess_data)
        self.preprocess_button.grid(row=1, column=1, padx=10, pady=20)

        # Create visualization dropdown
        self.visualization_var = tk.StringVar(self)
        self.visualization_var.set("Select Visualization")
        self.visualization_dropdown = tk.OptionMenu(self.gui_frame, self.visualization_var, *self.visualization_options)
        self.visualization_dropdown.grid(row=1, column=2, padx=10, pady=20)

        # Create generate button
        self.generate_button = tk.Button(self.gui_frame, text="Generate Visualization", command=self.generate_visualization)
        self.generate_button.grid(row=1, column=3, padx=10, pady=20)

        # Create download cleaned dataset button
        self.download_cleaned_dataset_button = tk.Button(self.gui_frame, text="Download Cleaned Dataset", command=self.download_cleaned_dataset)
        self.download_cleaned_dataset_button.grid(row=1, column=4, padx=10, pady=20)

        # Create download visualization button
        self.download_visualization_button = tk.Button(self.gui_frame, text="Download Visualization", command=self.download_visualization)
        self.download_visualization_button.grid(row=1, column=5, padx=10, pady=20)

        # Create canvas for visualization
        self.figure = plt.figure(figsize=(8, 50))
        self.gs = GridSpec(2, 2, figure=self.figure)
        self.visualization_canvas = FigureCanvasTkAgg(self.figure, self.gui_frame)
        self.visualization_canvas.get_tk_widget().grid(row=2, column=0, columnspan=4, padx=20, pady=20, sticky="nsew")
        self.pd = tk.Text(self.gui_frame, height=15, width=50)

        # Adjust margins using tight_layout
        self.figure.tight_layout()

        # Create log text box
        self.log_text = tk.Text(self.gui_frame, height=15, width=50)
        self.log_text.grid(row=2, column=4, columnspan=2, padx=20, pady=20, sticky="nsew")

        self.gui_frame.grid_rowconfigure(1, weight=1)
        self.gui_frame.grid_columnconfigure(0, weight=1)
        self.gui_frame.grid_columnconfigure(4, weight=1)

        self.data = None
        self.cleaned_data = None

        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        self.center_window()

    def center_window(self):
        # Center the main grid to the window
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x_offset = (width - self.gui_frame.winfo_reqwidth()) // 2
        y_offset = (height - self.gui_frame.winfo_reqheight()) // 2
        self.canvas.create_window((x_offset, y_offset), window=self.gui_frame, anchor=tk.CENTER)

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(-1 * (event.delta // 50), "units")

    def _on_canvas_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def upload_file(self):
        self.log_text.delete('1.0', tk.END)
        file_path = filedialog.askopenfilename(filetypes=[("Excel files", ".xlsx"), ("CSV files", ".csv")])
        if file_path:
            try:
                if file_path.endswith(".xlsx"):
                    self.data = pd.read_excel(file_path, engine='openpyxl')
                else:
                    self.data = pd.read_csv(file_path)
                self.log_text.insert(tk.END, f"File uploaded: {file_path}\n")
            except Exception as e:
                self.log_text.insert(tk.END, f"Error reading file: {e}\n")
                messagebox.showerror("Error", f"Error reading file: {e}")

    def preprocess_data(self):
        if self.data is not None:
            self.log_text.insert(tk.END, "Head of the dataset:\n")
            self.log_text.insert(tk.END, str(self.data.head()) + "\n\n")

            # Capture data information in a string buffer
            import io
            buffer = io.StringIO()
            self.data.info(buf=buffer)
            data_info_str = buffer.getvalue()

            self.log_text.insert(tk.END, "Data info:\n")
            self.log_text.insert(tk.END, data_info_str + "\n\n")

            self.log_text.insert(tk.END, "Null values:\n")
            self.log_text.insert(tk.END, str(self.data.isnull().sum()) + "\n\n")

            self.log_text.insert(tk.END, "Statistical analysis:\n")
            self.log_text.insert(tk.END, str(self.data.describe()) + "\n\n")

            numerical_cols = self.data.select_dtypes(include=['number']).columns
            if len(numerical_cols) >= 2:
                X = self.data[numerical_cols[:2]]
                y = self.data[numerical_cols[2]]

                # Handle NaN or infinity values
                X = X.fillna(X.mean())
                y = y.fillna(y.mean())

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Calculate accuracy before data cleaning
                regr = LinearRegression()
                regr.fit(X_train, y_train)
                y_pred = regr.predict(X_test)
                accuracy_before = regr.score(X_test, y_test)
                self.log_text.insert(tk.END, f"Accuracy before data cleaning: {accuracy_before:.2f}\n\n")

                # Data cleaning
                self.cleaned_data = self.data.dropna()  # Remove rows with missing values
                self.cleaned_data = self.cleaned_data.drop_duplicates()  # Remove duplicate rows

                # Calculate accuracy after data cleaning
                X_cleaned = self.cleaned_data[numerical_cols[:2]]
                y_cleaned = self.cleaned_data[numerical_cols[2]]
                X_train_cleaned, X_test_cleaned, y_train_cleaned, y_test_cleaned = train_test_split(X_cleaned, y_cleaned, test_size=0.2, random_state=42)

                regr_cleaned = LinearRegression()
                regr_cleaned.fit(X_train_cleaned, y_train_cleaned)
                y_pred_cleaned = regr_cleaned.predict(X_test_cleaned)
                accuracy_after = regr_cleaned.score(X_test_cleaned, y_test_cleaned)
                self.log_text.insert(tk.END, f"Accuracy after data cleaning: {accuracy_after:.2f}\n\n")
            else:
                messagebox.showerror("Error", "Dataset does not have enough numerical columns for accuracy calculation.")
        else:
            messagebox.showerror("Error", "No data available to preprocess.")

    def generate_visualization(self):
        if self.cleaned_data is not None:
            self.perform_visualization()
        else:
            messagebox.showerror("Error", "No preprocessed data available to generate visualization.")

    def perform_visualization(self):
        # Generate selected visualization
        visualization = self.visualization_var.get()
        self.figure.clear()
        if visualization == "All":
            self.generate_all_visualizations()
        else:
            self.generate_single_visualization(visualization)
        self.visualization_canvas.draw()
        self.log_text.insert(tk.END, "Visualization generated.\n")

    def generate_all_visualizations(self):
        visualizations = self.visualization_options[1:]
        num_rows = 8
        num_cols = 1
        gs = GridSpec(num_rows, num_cols)

        for i, visualization in enumerate(visualizations):
            row = i // num_cols
            col = i % num_cols
            ax = self.figure.add_subplot(gs[row, col])
            self.generate_single_visualization(visualization, ax)

    def generate_single_visualization(self, visualization, ax=None):
        if ax is None:
            ax = self.figure.add_subplot(self.gs[:, :1])

        # Add visualization title
        ax.set_title(visualization)

        if visualization == "Histogram":
            numerical_cols = self.data.select_dtypes(include=['number']).columns
            if len(numerical_cols) >= 3:
                for col in numerical_cols[:8]:
                    data = self.data[col]
                    data.plot(kind="hist", ax=ax, legend=True)
                    ax.set_xlabel(col)
                    ax.set_ylabel("Frequency")
            else:
                messagebox.showerror("Error", "Histogram requires at least three numerical columns in the dataset.")
                return
        elif visualization == "Bar Plot":
            numerical_cols = self.data.select_dtypes(include=['number']).columns
            if len(numerical_cols) >= 2:
                for col in numerical_cols[:8]:
                    data = self.data[col].mode()  # Use median instead of individual values
                    ax.bar(col, data)
                    ax.set_xlabel(col)
                    ax.set_ylabel("Mode")
            else:
                messagebox.showerror("Error", "Bar plot requires at least two numerical columns in the dataset.")
                return
        elif visualization == "Pie Chart":
            categorical_cols = self.data.select_dtypes(include=['object']).columns
            if len(categorical_cols) >= 1:
                col = categorical_cols[0]  # Select first categorical column
                data = self.data[col].value_counts().mean()  # Use mean of value counts
                ax.pie([data], labels=[col], autopct='%1.1f%%')
                ax.set_title(f"Pie Chart for {col}")
            else:
                messagebox.showerror("Error", "Pie chart requires at least one categorical column in the dataset.")
                return
        elif visualization == "Scatter Plot":
            numerical_cols = self.data.select_dtypes(include=['number']).columns
            if len(numerical_cols) >= 2:
                data = self.data[numerical_cols[:2]]
                data.plot(kind="scatter", x=numerical_cols[0], y=numerical_cols[1], ax=ax)
                ax.set_xlabel(numerical_cols[0])
                ax.set_ylabel(numerical_cols[1])
            else:
                messagebox.showerror("Error", "Scatter plot requires at least two numerical columns in the dataset.")
                return
        elif visualization == "Line Plot":
            numerical_cols = self.data.select_dtypes(include=['number']).columns
            if len(numerical_cols) >= 2:
                data = self.data[numerical_cols[:8]].copy()  # Create a copy to avoid modifying the original data
                data = data.melt(var_name='Columns', value_name='Value')  # Reshape the data into a long format
                ax = sns.lineplot(data=data, x='Columns', y='Value', ax=ax)
                ax.set_xlabel("Columns")
                ax.set_ylabel("Value")
            else:
                messagebox.showerror("Error", "Line plot requires at least two numerical columns in the dataset.")
                return
        elif visualization == "Heat Map":
            numerical_cols = self.data.select_dtypes(include=['number']).columns
            if len(numerical_cols) >= 2:
                col = numerical_cols[0]  # Select the first numerical column
                data = self.data[[col]].median().reset_index()  # Calculate median for a single column
                data.columns = ["Columns", col]  # Assign column names
                sns.heatmap(data=data.set_index("Columns").T, annot=True, ax=ax)
                ax.set_xlabel("Columns")
                ax.set_ylabel("Rows")
            else:
                messagebox.showerror("Error", "Heat Map requires at least two numerical columns in the dataset.")
                return
        elif visualization == "Box Plot":
            numerical_cols = self.data.select_dtypes(include=['number']).columns
            if len(numerical_cols) >= 2:
                data = self.data[numerical_cols[:2]]
                data.boxplot(ax=ax)
                ax.set_xlabel(numerical_cols[0])
                ax.set_ylabel(numerical_cols[1])

                # Drop outliers after creating the box plot
                Q1 = data[numerical_cols[1]].quantile(0.25)
                Q3 = data[numerical_cols[1]].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                self.data = self.data[(self.data[numerical_cols[1]] >= lower_bound) & (self.data[numerical_cols[1]] <= upper_bound)]
                self.log_text.insert(tk.END, "Outliers removed from the dataset.\n")
            else:
                messagebox.showerror("Error", "Box plot requires at least two numerical columns in the dataset.")
                return

    def download_cleaned_dataset(self):
        if self.data is not None:
            file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV File", "*.csv")])
            if file_path:
                try:
                    self.data.to_csv(file_path, index=False)
                    self.log_text.insert(tk.END, f"Cleaned dataset saved to {file_path}\n")
                except Exception as e:
                    self.log_text.insert(tk.END, f"Error saving cleaned dataset: {e}\n")
                    messagebox.showerror("Error", f"Error saving cleaned dataset: {e}")
        else:
            messagebox.showerror("Error", "No data available to save.")

    def download_visualization(self):
        if self.figure is not None:
            file_path = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF File", "*.pdf"), ("PNG File", "*.png")])
            if file_path:
                try:
                    if file_path.endswith(".pdf"):
                        with PdfPages(file_path, 'a4') as pdf:
                            pdf.savefig(self.figure)
                            plt.clf()
                            log_text = self.log_text.get("1.0", tk.END)
                            fig = plt.figure(figsize=(8.27, 11.69))  # A4 size in inches
                            plt.figtext(0.05, 0.95, log_text, fontsize=8, ha='left', va='top', wrap=True)
                            pdf.savefig(fig)
                        self.log_text.insert(tk.END, f"Visualization and log saved to {file_path}\n")
                    elif file_path.endswith(".png"):
                        self.figure.savefig(file_path, format='png')
                        plt.clf()
                        log_text = self.app_name_label.cget("text") + "\n\n" + self.log_text.get("1.0", tk.END)
                        fig = plt.figure(figsize=(8.27, 11.69))  # A4 size in inches
                        plt.figtext(0.05, 0.95, log_text, fontsize=8, ha='left', va='top', wrap=True)
                        plt.savefig(f"{os.path.splitext(file_path)[0]}_log.png", bbox_inches='tight')
                        self.log_text.insert(tk.END, f"Visualization saved to {file_path} and log saved to {os.path.splitext(file_path)[0]}_log.png\n")
                    else:
                        self.log_text.insert(tk.END, "Invalid file extension. Please choose either .pdf or .png.\n")
                except Exception as e:
                    self.log_text.insert(tk.END, f"Error saving visualization: {e}\n")
                    messagebox.showerror("Error", f"Error saving visualization: {e}")
        else:
            messagebox.showerror("Error", "No visualization available to save.")

if __name__ == "__main__":
    app = DataInsightPro()
    app.mainloop()
