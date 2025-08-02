from enum import Enum
import os
import cv2
import numpy as np
import csv
from pathlib import Path
import ast
import traceback


class Operator(Enum):
    ADD = 1
    SUB = 2
    MUL = 3
    DIV = 4
    EQL = 5


class LabellingTool:
    def __init__(self):
        # Get the project root directory by going two levels up from the script location
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent.parent
        
        # Create absolute paths
        self.csv_path = project_root / "dataset" / "segmented.csv"
        self.labels_path = project_root / "dataset" / "labels.csv"
        
        print(f"CSV path: {self.csv_path}")
        print(f"Labels path: {self.labels_path}")

        self.current_index = 0

        #empty lists to be loaded into
        self.images = []
        self.labels = {}

        #helps map the operator symbols to its label
        self.operator_map = {
            '1': 'ADD',
            '2': 'SUB',
            '3': 'MUL',
            '4': 'DIV',
            '5': 'EQL'
        }
        self.load_existing_labels()

    def load_existing_labels(self):
        #check if the label path exists just in case
        if self.labels_path.exists():
            with open(self.labels_path, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # skip header
                for row in reader:
                    if len(row) >= 2: #safety check to avoid incomplete rows
                        self.labels[int(row[0])] = row[1] #store in csv

    def save_labels(self):
        with open(self.labels_path, 'w', newline='') as f: #open label file to write
            writer = csv.writer(f)
            writer.writerow(['symbol_id', 'label'])
            for symbol_id, label in self.labels.items():
                writer.writerow([symbol_id, label]) # write label to the file in a consistent format
        print(f"Labels saved to {self.labels_path}")


    def load_images(self):
        try:
            if not self.csv_path.exists():
                print(f"Error: File not found at {self.csv_path}")
                return False
            
            # Read the entire file content to inspect it
            with open(self.csv_path, 'r') as f:
                content = f.read()
                print(f"File exists and contains {len(content)} characters")
                
                # Check if the file is empty
                if not content.strip():
                    print("Error: The segmented.csv file is empty")
                    return False
                
                # Print the first few lines to inspect the format
                print("First 200 characters:")
                print(content[:200])
                
            # Now try to parse the CSV
            with open(self.csv_path, 'r') as f:
                reader = csv.reader(f)
                try:
                    header = next(reader)  # skip header
                    print(f"CSV header: {header}")
                except StopIteration:
                    print("Error: CSV file has no rows")
                    return False

                for i, row in enumerate(reader):
                    print(f"Processing row {i+1}: length={len(row)}")
                    
                    if len(row) < 3:
                        print(f"Warning: Row {i+1} has fewer than 3 columns: {row}")
                        continue
                        
                    try:
                        symbol_id = int(row[0])
                        print(f"  symbol_id = {symbol_id}")
                        
                        image_data = ast.literal_eval(row[1])
                        print(f"  image_data length = {len(image_data)}")
                        
                        shape = ast.literal_eval(row[2])
                        print(f"  shape = {shape}")
                        
                        image = np.array(image_data).reshape(shape)
                        print(f"  image shape = {image.shape}")
                        
                        self.images.append((symbol_id, image))
                    except Exception as row_error:
                        print(f"Error processing row {i+1}: {row_error}")
                        traceback.print_exc()
                        continue

            print(f"Loaded {len(self.images)} symbols for labelling")

            if len(self.images) == 0:
                print("Warning: No images were loaded. Check the CSV format.")
                return False
                
            return True

        except Exception as e:
            print(f"Error loading images: {e}")
            traceback.print_exc()
            return False

    def display_menu(self):
        print(
        """
        Operator Labels:
        {1} ADD (+)
        {2} SUB (-)
        {3} MUL (×)
        {4} DIV (÷)
        {5} EQL (=)
        
        Navigation:
        {n} Next symbol
        {p} Previous symbol
        {s} Save progress
        {q} Save and quit
        {d} Delete label
        """
        )


    def run(self):
        if not self.load_images():
            print("Failed to load images. Please check the error messages above.")
            return

        while self.current_index < len(self.images):
            symbol_id, image = self.images[self.current_index]

            # Display current image
            window_name = 'Operator Symbol'
            cv2.imshow(window_name, image)
            cv2.waitKey(1)

            # Show current status
            current_label = self.labels.get(symbol_id, "unlabelled")
            print(f"\nSymbol {symbol_id + 1} of {len(self.images)} (Label: {current_label})")

            self.display_menu()
            choice = input("Choice >> ").lower()

            if choice == 'q':
                break
            elif choice == 's':
                self.save_labels()
                continue
            elif choice == 'n':
                self.current_index = min(self.current_index + 1, len(self.images) - 1)
                continue
            elif choice == 'p':
                self.current_index = max(0, self.current_index - 1)
                continue
            elif choice == 'd':
                if symbol_id in self.labels:
                    del self.labels[symbol_id]
                    print(f"Deleted label for symbol {symbol_id}")
                continue
            elif choice in self.operator_map:
                self.labels[symbol_id] = self.operator_map[choice]
                print(f"Labeled as {self.operator_map[choice]}")
                self.current_index += 1
            else:
                print("Invalid choice")
                continue

        # Save labels before exiting
        self.save_labels()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    tool = LabellingTool()
    tool.run()