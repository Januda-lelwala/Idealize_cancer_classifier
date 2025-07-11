import pandas as pd
import numpy as np
import os

class CSVOversampler:
    def __init__(self, original_csv_path, target_column):
        """
        Initialize the CSV oversampler
        
        Args:
            original_csv_path: Path to the original CSV file
            target_column: Name of the target column for classification
        """
        self.original_csv_path = original_csv_path
        self.target_column = target_column
        self.original_df = None
        self.minority_df = None
        self.majority_df = None
        
    def load_and_analyze_data(self):
        """Load the original data and analyze class distribution"""
        self.original_df = pd.read_csv(self.original_csv_path)
        print(f"Original dataset shape: {self.original_df.shape}")
        
        # Analyze class distribution
        class_counts = self.original_df[self.target_column].value_counts()
        print("\nClass distribution:")
        print(class_counts)
        
        # Identify majority and minority classes
        self.majority_class = class_counts.idxmax()
        self.minority_class = class_counts.idxmin()
        self.majority_count = class_counts.max()
        self.minority_count = class_counts.min()
        
        print(f"\nMajority class: {self.majority_class} ({self.majority_count} samples)")
        print(f"Minority class: {self.minority_class} ({self.minority_count} samples)")
        print(f"Imbalance ratio: {self.minority_count/self.majority_count:.3f}")
        
        return class_counts
    
    def extract_minority_data(self, output_path='minority_class_data.csv'):
        """Extract minority class data to a separate CSV file"""
        if self.original_df is None:
            self.load_and_analyze_data()
        
        # Extract minority class samples
        self.minority_df = self.original_df[self.original_df[self.target_column] == self.minority_class].copy()
        
        # Save minority data to CSV
        self.minority_df.to_csv(output_path, index=False)
        print(f"\nMinority class data saved to: {output_path}")
        print(f"Minority class samples: {len(self.minority_df)}")
        
        return output_path
    
    def create_balanced_dataset(self, balance_ratio=1.0, output_path='balanced_train.csv'):
        """
        Create a balanced dataset by duplicating minority class data
        
        Args:
            balance_ratio: 1.0 = perfectly balanced, 0.5 = half balanced, etc.
            output_path: Path for the balanced dataset
        """
        if self.original_df is None:
            self.load_and_analyze_data()
        
        if self.minority_df is None:
            self.extract_minority_data()
        
        # Calculate how many times to duplicate minority data
        target_minority_count = int(self.minority_count + (self.majority_count - self.minority_count) * balance_ratio)
        times_to_duplicate = (target_minority_count - self.minority_count) // self.minority_count
        remainder_samples = (target_minority_count - self.minority_count) % self.minority_count
        
        print(f"\nCreating balanced dataset:")
        print(f"Target minority count: {target_minority_count}")
        print(f"Times to duplicate entire minority set: {times_to_duplicate}")
        print(f"Additional samples needed: {remainder_samples}")
        
        # Start with original data
        balanced_df = self.original_df.copy()
        
        # Add complete duplications of minority data
        for i in range(times_to_duplicate):
            balanced_df = pd.concat([balanced_df, self.minority_df], ignore_index=True)
            print(f"Added complete minority set copy {i+1}")
        
        # Add partial duplication if needed
        if remainder_samples > 0:
            # Randomly sample additional minority samples
            np.random.seed(42)  # For reproducibility
            additional_samples = self.minority_df.sample(n=remainder_samples, replace=True)
            balanced_df = pd.concat([balanced_df, additional_samples], ignore_index=True)
            print(f"Added {remainder_samples} additional minority samples")
        
        # Shuffle the balanced dataset
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Save balanced dataset
        balanced_df.to_csv(output_path, index=False)
        
        # Print final statistics
        final_class_counts = balanced_df[self.target_column].value_counts()
        final_ratio = final_class_counts.min() / final_class_counts.max()
        
        print(f"\nBalanced dataset created: {output_path}")
        print(f"Final dataset shape: {balanced_df.shape}")
        print(f"Final class distribution:")
        print(final_class_counts)
        print(f"Final balance ratio: {final_ratio:.3f}")
        
        return output_path, balanced_df
    
    def create_multiple_minority_copies(self, num_copies=3, output_dir='minority_copies'):
        """
        Create multiple copies of minority data as separate CSV files
        
        Args:
            num_copies: Number of minority data copies to create
            output_dir: Directory to store the copies
        """
        if self.minority_df is None:
            self.extract_minority_data()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        minority_files = []
        
        for i in range(num_copies):
            # Add some noise/variation to each copy (optional)
            copy_df = self.minority_df.copy()
            
            # Save each copy
            filename = f"minority_copy_{i+1}.csv"
            filepath = os.path.join(output_dir, filename)
            copy_df.to_csv(filepath, index=False)
            minority_files.append(filepath)
            print(f"Created minority copy {i+1}: {filepath}")
        
        return minority_files
    
    def combine_csvs_for_balance(self, minority_files, output_path='combined_balanced.csv'):
        """
        Combine original CSV with multiple minority CSV files
        
        Args:
            minority_files: List of paths to minority data CSV files
            output_path: Path for the final combined CSV
        """
        if self.original_df is None:
            self.load_and_analyze_data()
        
        # Start with original data
        combined_df = self.original_df.copy()
        
        # Add each minority file
        for file_path in minority_files:
            minority_data = pd.read_csv(file_path)
            combined_df = pd.concat([combined_df, minority_data], ignore_index=True)
            print(f"Added data from: {file_path}")
        
        # Shuffle the combined dataset
        combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Save combined dataset
        combined_df.to_csv(output_path, index=False)
        
        # Print statistics
        final_class_counts = combined_df[self.target_column].value_counts()
        final_ratio = final_class_counts.min() / final_class_counts.max()
        
        print(f"\nCombined dataset created: {output_path}")
        print(f"Final dataset shape: {combined_df.shape}")
        print(f"Final class distribution:")
        print(final_class_counts)
        print(f"Final balance ratio: {final_ratio:.3f}")
        
        return output_path, combined_df

def create_balanced_csv(input_csv='train.csv', target_col='survival_status', output_csv='balanced_train.csv', balance_ratio=1.0):
    """
    Simple function to create a balanced CSV file
    
    Args:
        input_csv: Path to input CSV file
        target_col: Name of target column
        output_csv: Path for output balanced CSV
        balance_ratio: Balance ratio (1.0 = perfectly balanced)
    
    Returns:
        Path to created balanced CSV file
    """
    print(f"Creating balanced dataset from {input_csv}...")
    
    # Initialize oversampler
    oversampler = CSVOversampler(input_csv, target_col)
    
    # Create balanced dataset
    balanced_path, _ = oversampler.create_balanced_dataset(
        balance_ratio=balance_ratio,
        output_path=output_csv
    )
    
    print(f"\n✅ Balanced dataset created: {balanced_path}")
    print("You can now use this balanced CSV in your classifier!")
    
    return balanced_path

def main():
    """Example usage of the CSVOversampler"""
    # Initialize the oversampler
    oversampler = CSVOversampler('train.csv', 'survival_status')
    
    # Method 1: Direct balanced dataset creation
    print("=" * 50)
    print("METHOD 1: Direct balanced dataset creation")
    print("=" * 50)
    oversampler.load_and_analyze_data()
    balanced_path, balanced_df = oversampler.create_balanced_dataset(
        balance_ratio=1.0, 
        output_path='balanced_train_direct.csv'
    )
    
    # Method 2: Create multiple minority copies and combine
    print("\n" + "=" * 50)
    print("METHOD 2: Multiple minority copies approach")
    print("=" * 50)
    
    # Extract minority data
    minority_path = oversampler.extract_minority_data('minority_class_data.csv')
    
    # Calculate how many copies we need for balance
    copies_needed = (oversampler.majority_count // oversampler.minority_count) - 1
    if oversampler.majority_count % oversampler.minority_count > 0:
        copies_needed += 1
    
    print(f"Creating {copies_needed} copies of minority data for balance")
    
    # Create multiple copies
    minority_files = oversampler.create_multiple_minority_copies(
        num_copies=copies_needed,
        output_dir='minority_copies'
    )
    
    # Combine all files
    combined_path, combined_df = oversampler.combine_csvs_for_balance(
        minority_files,
        output_path='balanced_train_combined.csv'
    )
    
    print(f"\nBalanced datasets created:")
    print(f"1. Direct method: balanced_train_direct.csv")
    print(f"2. Combined method: balanced_train_combined.csv")

if __name__ == "__main__":
    # Simple execution - just run this file to create balanced dataset
    print("🔄 CSV Oversampler - Creating Balanced Dataset")
    print("=" * 60)
    
    # Create balanced dataset with default settings
    balanced_file = create_balanced_csv(
        input_csv='train.csv',
        target_col='survival_status', 
        output_csv='balanced_train.csv',
        balance_ratio=1.0
    )
    
    print(f"\n📊 Balanced dataset ready: {balanced_file}")
    print("Now you can use 'balanced_train.csv' in your patient_survival_classifier.py")
    print("Just change the CSV path in your classifier to use the balanced data!")
