"""Main module for project-tailwind."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table

console = Console()


def main():
    """Main entry point for the application."""
    console.print("Welcome to Project Tailwind!", style="bold green")
    
    # Example data science workflow
    data = np.random.randn(100, 3)
    df = pd.DataFrame(data, columns=['A', 'B', 'C'])
    
    # Display summary statistics
    table = Table(title="Data Summary")
    table.add_column("Column", style="cyan", no_wrap=True)
    table.add_column("Mean", style="magenta")
    table.add_column("Std", style="green")
    
    for col in df.columns:
        table.add_row(col, f"{df[col].mean():.3f}", f"{df[col].std():.3f}")
    
    console.print(table)
    
    # Simple plot
    plt.figure(figsize=(10, 6))
    df.plot(kind='hist', bins=20, alpha=0.7)
    plt.title("Sample Data Distribution")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()


if __name__ == "__main__":
    main()