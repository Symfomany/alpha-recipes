from rich import print as rprint

try:
    import torch
except ImportError:
    torch = None
    
# --- Check CUDA / GPU ---

def _log_cuda_status() -> None:
    """Affiche l'état CUDA/GPU dans la console (rich)."""
    if torch is None:
        rprint("[bold yellow][CUDA][/bold yellow] [yellow]PyTorch non installé dans cette venv, impossible de tester le GPU.[/yellow]")
        return

    cuda_available = torch.cuda.is_available()
    if cuda_available:
        device_name = torch.cuda.get_device_name(0)
        rprint(
            f"[bold green][CUDA][/bold green] GPU détecté : [green]{device_name}[/green] "
            f"(torch.cuda.is_available() = True)"
        )
    else:
        rprint(
            "[bold red][CUDA][/bold red] [red]Aucun GPU CUDA détecté dans cette venv "
            "(torch.cuda.is_available() = False).[/red]"
        )


if __name__ == "__main__":
    _log_cuda_status()