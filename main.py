import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error

from src.mlp_regressor.mlp import ContextualAtomScalarMLP
from src.mlp_regressor.training import train_epoch, evaluate
from src.mlp_regressor.data import get_dataloaders

# --- CONFIGURATION ---
HIDDEN_LAYERS = [40, 40, 32]
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 30
PATIENCE = 10
MW_CUTOFF = 500  # Threshold to define "large" molecules

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():    
    train_loader, val_loader, test_loader = get_dataloaders(BATCH_SIZE)
    
    sample_batch = next(iter(train_loader))
    input_dim = sample_batch[0]['atom_features'].shape[1]
    
    mol_data = train_loader.dataset.mol_data

    model = ContextualAtomScalarMLP(input_dim=input_dim, hidden_dims=HIDDEN_LAYERS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    print(f"Training Contextual MLP with architecture:")
    print(f"   Atom Encoder: {input_dim} -> {' -> '.join(map(str, HIDDEN_LAYERS[:-1]))}")
    print(f"   Molecular Context: mean pooling -> {HIDDEN_LAYERS[-1]}")
    print(f"   Scalar Predictor: (atom_enc + mol_context) -> {HIDDEN_LAYERS[-1]} -> 1")
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 6. Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []

    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)
        val_preds, val_targets, _, _, _ = evaluate(model, val_loader, device)
        val_loss = mean_squared_error(val_targets, val_preds)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val MSE: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model_contextual.pt')
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # 7. Final Evaluation
    print("\nLoading best model and evaluating on test set...")
    model.load_state_dict(torch.load('best_model_contextual.pt'))
    test_preds, test_targets, test_baselines, test_mws, test_mol_index_returned = evaluate(model, test_loader, device)

    print("\nData Alignment Sanity Check:")
    for i in range(min(3, len(test_mol_index_returned))):
        mol_index = test_mol_index_returned[i]
        expected_mw = mol_data[mol_index]['mw']
        expected_exp = mol_data[mol_index]['exp_logp']
        expected_rdkit = mol_data[mol_index]['rdkit_logp']
        print(f"   Mol {mol_index}: MW={expected_mw:.1f} (got {test_mws[i]:.1f}), Exp={expected_exp:.2f} (got {test_targets[i]:.2f}), RDKit={expected_rdkit:.2f} (got {test_baselines[i]:.2f})")
        assert abs(expected_mw - test_mws[i]) < 0.1, f"MW mismatch for molecule {mol_index}!"
        assert abs(expected_exp - test_targets[i]) < 0.01, f"Exp mismatch for molecule {mol_index}!"
        assert abs(expected_rdkit - test_baselines[i]) < 0.01, f"RDKit mismatch for molecule {mol_index}!"
    print("   Data is correctly aligned!")

    print("\n" + "="*60)
    print("RESULTS ANALYSIS (Contextual Per-Atom MLP)")
    print("="*60)

    def get_rmse(true, pred): return np.sqrt(mean_squared_error(true, pred))
    mask_large = test_mws > MW_CUTOFF
    mask_small = ~mask_large

    print(f"\nDataset Statistics:")
    print(f"   Test Set Size: {len(test_targets)} molecules")
    print(f"   Small Molecules (<= 500 Da): {sum(mask_small)}")
    print(f"   Large Molecules (> 500 Da): {sum(mask_large)}")

    rmse_base_all = get_rmse(test_targets, test_baselines)
    rmse_our_all = get_rmse(test_targets, test_preds)
    
    print("\nBaseline Error Analysis:")
    print(f"   Overall MAE: {np.mean(np.abs(test_targets - test_baselines)):.4f}")

    print("\nSample Data (first 5 test molecules):")
    for i in range(min(5, len(test_targets))):
        size = "large" if test_mws[i] > MW_CUTOFF else "small"
        print(f"   [{size:5s}] MW={test_mws[i]:6.1f} | Exp={test_targets[i]:6.2f} | RDKit={test_baselines[i]:6.2f} | Error={abs(test_targets[i]-test_baselines[i]):5.2f}")

    print("-" * 60)
    print(f"{ 'METRIC':<20} | { 'BASELINE (RDKit)':<20} | { 'CONTEXTUAL MLP':<20}")
    print("-" * 60)
    print(f"{ 'RMSE (All)':<20} | {rmse_base_all:.4f}{' '*14} | {rmse_our_all:.4f}")

    if sum(mask_large) > 0:
        rmse_base_large = get_rmse(test_targets[mask_large], test_baselines[mask_large])
        rmse_our_large = get_rmse(test_targets[mask_large], test_preds[mask_large])
        rmse_base_small = get_rmse(test_targets[mask_small], test_baselines[mask_small])
        rmse_our_small = get_rmse(test_targets[mask_small], test_preds[mask_small])
        print(f"{ 'RMSE (Small)':<20} | {rmse_base_small:.4f}{' '*14} | {rmse_our_small:.4f}")
        print(f"{ 'RMSE (Large)':<20} | {rmse_base_large:.4f}{' '*14} | {rmse_our_large:.4f}")
        improvement = (rmse_base_large - rmse_our_large) / rmse_base_large * 100
        print("-" * 60)
        print(f"\nCONCLUSION: On large molecules, you reduced the error by {improvement:.1f}%")
    else:
        print("\nNo large molecules in test set for comparison.")

    print(f"\nBest validation MSE: {best_val_loss:.6f}")
    print("\nKEY INSIGHT: Each atom's scalar is now conditioned on the entire molecule's context!")

if __name__ == "__main__":
    main()