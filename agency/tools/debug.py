import torch


def print_tensor_stats(name, tt):
    print(
        f"{name} [{torch.min(tt[:]):.4f},{tt.mean():.4f},{torch.max(tt[:]):.4f}], NANS: {torch.sum(torch.isnan(tt))}"
    )


def debug_grads(named_parameters, N=5, lr=0.0001, show_pred=False):
    print("\n")
    print("=================================")
    print("DEBUG GRADS")
    print("=================================")
    for n, p in named_parameters:
        print("=================================")
        print(f"Name: {n}, shape: {p.shape}")
        p_short = torch.flatten(p)[0:N].detach().cpu().numpy()
        g_short = torch.flatten(p.grad)[0:N].detach().cpu().numpy()
        print(f" Vals: {p_short}, Grads: {g_short}")
        if show_pred:
            print(f" ------ PRED: {p_short - lr* g_short}")
    print("=================================")
    print("=================================")
    print("=================================")
    print("\n")


def manual_update(named_parameters, learning_rate=0.0001):
    for n, p in named_parameters:
        p.data.sub_(p.grad.data * learning_rate)


def print_named_params(output_name, named_parameters):
    print("\n")
    print(output_name + " params")
    for n, p in named_parameters:
        if p.requires_grad:
            print(f"----GRAD----- Name: {n}, shape: {p.shape}")
        else:
            print(f"----NO GRAD-- Name: {n}, shape: {p.shape}")
    print("\n")
