"""
FLOWRA Framework - Example Usage

This script demonstrates how to use the FLOWRA framework for:
1. Flow-based adaptation of a pre-trained model
2. Training with dynamic subspace refinement
3. Multi-task adapter composition

The example uses a simple model for demonstration.
For real use, replace with your actual model and data.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import sys
sys.path.insert(0, '/.')

from flowra import FLOWRA
from flowra.utils import summarize_adaptation, plot_rank_allocation, plot_training_curves


# ============================================================================
# Example 1: Basic Usage - Adapt a Model for a Single Task
# ============================================================================

def example_basic_usage():
    """
    Demonstrates the basic FLOWRA workflow:
    1. Create/load a model
    2. Initialize FLOWRA
    3. Analyze flow and apply adaptation
    4. Train the adapted model
    """
    print("\n" + "=" * 70)
    print("Example 1: Basic FLOWRA Usage")
    print("=" * 70)
    
    # Create a simple model for demonstration
    # In practice, this would be your pre-trained model
    model = nn.Sequential(
        nn.Linear(256, 512),
        nn.ReLU(),
        nn.LayerNorm(512),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.LayerNorm(512),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 10)  # 10-class output
    )
    
    # Create dummy calibration data
    # In practice, use a small subset of your training data
    calibration_inputs = torch.randn(100, 256)
    calibration_targets = torch.randint(0, 10, (100,))
    calibration_dataset = TensorDataset(calibration_inputs, calibration_targets)
    calibration_loader = DataLoader(calibration_dataset, batch_size=32)
    
    # Initialize FLOWRA
    # total_budget: fraction of base parameters for adaptation (1% here)
    # interference_threshold: η for multi-task composition
    framework = FLOWRA(
        model=model,
        total_budget=0.01,  # 1% of base parameters
        interference_threshold=0.3
    )
    
    # Apply adaptation - this runs:
    # Phase 1: Flow Analysis
    # Phase 2: Architecture Configuration
    # And creates the adapted model
    adapted_model = framework.apply_adaptation(
        calibration_data=calibration_loader,
        use_flow_init=True  # Use Algorithm 5: Flow-Aware Initialization
    )
    
    # Print summary
    print("\n" + framework.summary())
    
    # Create training data
    train_inputs = torch.randn(1000, 256)
    train_targets = torch.randint(0, 10, (1000,))
    train_dataset = TensorDataset(train_inputs, train_targets)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Train the adapted model
    # use_dynamic_refinement enables Algorithm 3: Progressive Subspace Refinement
    framework.train(
        train_loader=train_loader,
        epochs=2,
        lr=1e-3,
        use_dynamic_refinement=True,
        refinement_interval=50
    )
    
    # Visualize results
    print("\nTraining completed!")
    plot_training_curves(framework.training_history)
    
    return framework


# ============================================================================
# Example 2: Custom Flow Analysis
# ============================================================================

def example_custom_analysis():
    """
    Demonstrates how to perform flow analysis separately
    and inspect the results before applying adaptation.
    """
    print("\n" + "=" * 70)
    print("Example 2: Custom Flow Analysis")
    print("=" * 70)
    
    # Create model
    model = nn.Sequential(
        nn.Linear(128, 256),
        nn.LayerNorm(256),
        nn.Linear(256, 256),
        nn.LayerNorm(256),
        nn.Linear(256, 64)
    )
    
    # Create calibration data
    cal_inputs = torch.randn(64, 128)
    cal_targets = torch.randint(0, 64, (64,))
    cal_loader = DataLoader(TensorDataset(cal_inputs, cal_targets), batch_size=16)
    
    # Initialize framework
    framework = FLOWRA(model, total_budget=0.05)
    
    # Step 1: Analyze flow (Phase 1 only)
    flow_info = framework.analyze_flow(cal_loader)
    
    # Inspect flow profiles
    print("\nFlow Profiles:")
    print("-" * 60)
    print(f"{'Layer':<30} {'ψ (Sensitivity)':<15} {'γ (Conductance)':<15} {'ρ (Redundancy)':<15}")
    print("-" * 60)
    
    for name, info in flow_info.items():
        fp = info.flow_profile
        print(f"{name:<30} {fp.psi:<15.4f} {fp.gamma:<15.4f} {fp.rho:<15.4f}")
    
    # Now you can make informed decisions about adaptation
    # For example, skip layers with very low sensitivity:
    important_layers = {
        name: info for name, info in flow_info.items()
        if info.flow_profile.psi > 0.05
    }
    print(f"\nLayers with ψ > 0.05: {len(important_layers)} of {len(flow_info)}")
    
    # Continue with adaptation
    framework.configure_architecture()
    
    return framework


# ============================================================================
# Example 3: Multi-Task Composition
# ============================================================================

def example_multi_task():
    """
    Demonstrates composing adapters from multiple tasks using
    Algorithm 4: Orthogonal Adapter Composition.
    """
    print("\n" + "=" * 70)
    print("Example 3: Multi-Task Adapter Composition")
    print("=" * 70)
    
    # Create base model
    model = nn.Sequential(
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )
    
    # Create calibration data
    cal_inputs = torch.randn(64, 128)
    cal_targets = torch.randint(0, 10, (64,))
    cal_loader = DataLoader(TensorDataset(cal_inputs, cal_targets), batch_size=16)
    
    # Train adapters for Task 1
    print("\nTraining Task 1 adapter...")
    framework1 = FLOWRA(model, total_budget=0.02)
    framework1.apply_adaptation(cal_loader)
    
    task1_inputs = torch.randn(500, 128)
    task1_targets = torch.randint(0, 10, (500,))
    task1_loader = DataLoader(TensorDataset(task1_inputs, task1_targets), batch_size=32)
    framework1.train(task1_loader, epochs=1, use_dynamic_refinement=False)
    
    # Train adapters for Task 2 (different task distribution)
    print("\nTraining Task 2 adapter...")
    import copy
    framework2 = FLOWRA(copy.deepcopy(model), total_budget=0.02)
    framework2.apply_adaptation(cal_loader)
    
    task2_inputs = torch.randn(500, 128) * 2 + 1  # Different distribution
    task2_targets = torch.randint(0, 10, (500,))
    task2_loader = DataLoader(TensorDataset(task2_inputs, task2_targets), batch_size=32)
    framework2.train(task2_loader, epochs=1, use_dynamic_refinement=False)
    
    # Compose adapters
    print("\nComposing adapters...")
    merged_adapters = framework1.merge_adapters(
        adapter_dicts=[framework1.adapters, framework2.adapters],
        task_weights=[0.6, 0.4]  # Weighted towards Task 1
    )
    
    print(f"\nMerged {len(merged_adapters)} adapter layers")
    
    # Analyze compatibility
    from flowra.algorithms.orthogonal_composition import analyze_task_compatibility
    compatibility = analyze_task_compatibility(
        [framework1.adapters, framework2.adapters],
        task_names=["Task 1", "Task 2"]
    )
    print(compatibility)
    
    return merged_adapters


# ============================================================================
# Example 4: Working with Transformers
# ============================================================================

def example_transformer():
    """
    Demonstrates FLOWRA with a Transformer-style model.
    Shows how attention layers get specialized adapters.
    """
    print("\n" + "=" * 70)
    print("Example 4: Transformer Adaptation")
    print("=" * 70)
    
    # Simple Transformer-like model
    class SimpleTransformer(nn.Module):
        def __init__(self, d_model=256, nhead=4, num_layers=2):
            super().__init__()
            self.embedding = nn.Linear(64, d_model)
            self.layers = nn.ModuleList([
                nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
                for _ in range(num_layers)
            ])
            self.classifier = nn.Linear(d_model, 10)
        
        def forward(self, x):
            x = self.embedding(x)
            for layer in self.layers:
                x = layer(x)
            return self.classifier(x.mean(dim=1))
    
    model = SimpleTransformer()
    
    # Calibration data (sequence data)
    cal_inputs = torch.randn(32, 16, 64)  # [batch, seq_len, features]
    cal_targets = torch.randint(0, 10, (32,))
    cal_loader = DataLoader(TensorDataset(cal_inputs, cal_targets), batch_size=8)
    
    # Apply FLOWRA
    framework = FLOWRA(model, total_budget=0.03)
    framework.apply_adaptation(cal_loader)
    
    # Check adapter types assigned
    print("\nAdapter Assignment:")
    for name, adapter in framework.adapters.items():
        print(f"  {name}: {type(adapter).__name__}")
    
    # Train briefly
    train_inputs = torch.randn(200, 16, 64)
    train_targets = torch.randint(0, 10, (200,))
    train_loader = DataLoader(TensorDataset(train_inputs, train_targets), batch_size=16)
    
    framework.train(train_loader, epochs=1)
    
    return framework


# ============================================================================
# Example 5: Efficient Inference with Merged Weights
# ============================================================================

def example_merged_inference():
    """
    Demonstrates merging adapters back into base weights
    for efficient inference without adapter overhead.
    """
    print("\n" + "=" * 70)
    print("Example 5: Merged Inference")
    print("=" * 70)
    
    # Setup model and train
    model = nn.Sequential(
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.Linear(64, 5)
    )
    
    cal_inputs = torch.randn(32, 64)
    cal_targets = torch.randint(0, 5, (32,))
    cal_loader = DataLoader(TensorDataset(cal_inputs, cal_targets), batch_size=8)
    
    framework = FLOWRA(model, total_budget=0.05)
    adapted_model = framework.apply_adaptation(cal_loader)
    
    train_inputs = torch.randn(200, 64)
    train_targets = torch.randint(0, 5, (200,))
    train_loader = DataLoader(TensorDataset(train_inputs, train_targets), batch_size=16)
    framework.train(train_loader, epochs=1, use_dynamic_refinement=False)
    
    # Merge adapters for inference
    merged_model = framework.merge_adapters_into_base()
    
    # Compare inference
    test_input = torch.randn(1, 64)
    
    with torch.no_grad():
        adapted_output = adapted_model(test_input)
        merged_output = merged_model(test_input)
    
    # Outputs should be identical
    diff = (adapted_output - merged_output).abs().max().item()
    print(f"\nMax output difference: {diff:.2e}")
    print(f"Outputs match: {diff < 1e-5}")
    
    # Check parameter count - merged should have same as original
    from flowra.utils import count_parameters
    print(f"\nAdapted model parameters: {count_parameters(adapted_model):,}")
    print(f"Merged model parameters: {count_parameters(merged_model):,}")
    print(f"Base model parameters: {count_parameters(framework.base_model):,}")
    
    return merged_model


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("FLOWRA Framework Examples")
    print("=" * 70)
    
    # Run examples
    example_basic_usage()
    example_custom_analysis()
    example_multi_task()
    example_transformer()
    example_merged_inference()
    
    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)
