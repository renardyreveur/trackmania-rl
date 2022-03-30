
# Training progress
def progress(data_loader, b_idx):
    base = '[{}/{} ({:.0f}%)]'
    total = len(data_loader)
    current = b_idx
    return base.format(current, total, 100.0 * current / total)
