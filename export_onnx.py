import sys
import os
import torch
import torch.nn as nn

# Add cloned repo to python path
sys.path.insert(0, '/dinov3_deepstream/dinov3_repo')

from dinov3.hub.backbones import dinov3_vits16plus

class DINOv3BackboneONNXWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # x is [B, 3, H, W]
        # get_intermediate_layers returns a tuple of length 1 when n=1 and reshape=True.
        # The returned tensor shape is [B, 384, H/16, W/16]
        features = self.model.get_intermediate_layers(x, n=1, reshape=True)[0]
        return features

def main():
    print("Loading DINOv3 ViT-S/16+ model structure...")
    model = dinov3_vits16plus(pretrained=False)
    
    weights_path = '/dinov3_deepstream/dinov3_models/backbone/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth'
    print(f"Loading local weights from {weights_path}...")
    state_dict = torch.load(weights_path, map_location='cpu')
    
    if 'model' in state_dict:
        state_dict = state_dict['model']
        
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    
    wrapper = DINOv3BackboneONNXWrapper(model)
    wrapper.eval()
    
    # Create dummy input: 1x3x640x640
    dummy_input = torch.randn(1, 3, 640, 640, dtype=torch.float32)
    
    with torch.no_grad():
        out = wrapper(dummy_input)
        print(f"Test forward pass output shape: {out.shape}")
        assert out.shape == (1, 384, 40, 40), f"Expected shape (1, 384, 40, 40), but got {out.shape}"
        
    onnx_path = '/dinov3_deepstream/dinov3_models/backbone/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.onnx'
    print(f"Exporting model to ONNX at {onnx_path}...")
    
    torch.onnx.export(
        wrapper,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['features'],
        dynamo=False,
        dynamic_axes={
            'input': {0: 'batch_size'},
            'features': {0: 'batch_size'}
        }
    )
    print("ONNX export completed successfully!")

if __name__ == '__main__':
    main()
