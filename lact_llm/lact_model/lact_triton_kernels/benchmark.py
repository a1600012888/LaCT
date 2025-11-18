import torch

def report_error(ref, ours, name=""):
    abs_error = torch.abs(ref - ours)
    abs_ref = torch.abs(ref)
    abs_ours = torch.abs(ours)

    error_rel = abs_error / (1e-4 + torch.maximum(abs_ref, abs_ours))

    mean_error_rel = error_rel.mean()
    max_error_rel = error_rel.max()

    top_k_error_rel = torch.topk(error_rel, k=10).values
    top_k_error_mean = top_k_error_rel.mean()

    mean_abs_error = abs_error.mean()
    max_abs_error = abs_error.max()
    mean_abs_ref = abs_ref.mean()
    max_abs_ref = abs_ref.max()
    mean_abs_ours = abs_ours.mean()
    max_abs_ours = abs_ours.max()

    print(
        f"{name} - Shape: {ref.shape} Ref dtype: {ref.dtype} Ours dtype: {ours.dtype}"
    )
    print(
        f"{name} - Mean Relative Error: {mean_error_rel}, Max Relative Error: {max_error_rel} Top 10 Relative Error: {top_k_error_mean}"
    )
    print(
        f"{name} - Mean Absolute Error: {mean_abs_error}, Max Absolute Error: {max_abs_error}"
    )
    print(
        f"{name} - Mean Reference Signal Magnitude: {mean_abs_ref}, Max Reference Signal Magnitude: {max_abs_ref}"
    )
    print(
        f"{name} - Mean Ours Signal Magnitude: {mean_abs_ours}, Max Ours Signal Magnitude: {max_abs_ours}"
    )
    print("--------------------------------")
