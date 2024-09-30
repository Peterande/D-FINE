import torch

def save_only_ema_weights(checkpoint_file, output_file):
    """Extract and save only the EMA weights."""
    checkpoint = torch.load(checkpoint_file, map_location='cpu')
    
    weights = {}
    if 'ema' in checkpoint:
        weights['model'] = checkpoint['ema']['module']
    else:
        raise ValueError("The checkpoint does not contain 'ema'.")

    torch.save(weights, output_file)
    print(f"EMA weights saved to {output_file}")

if __name__ == '__main__':
    checkpoint_file = '/home/pengys/code/rtdetrv2_pytorch/tb0902/b2_obj365_fix/checkpoint0012.pth'
    output_file = '/home/pengys/code/rtdetrv2_pytorch/tb0902/b2_obj365_fix/b2_e12_obj365.pth'
    save_only_ema_weights(checkpoint_file, output_file)
