#!/usr/bin/env python3
"""
Extract layer scores from SAFE-R computation for use in cascade analysis
"""

import argparse
import pandas as pd
import pathlib

LAYER_ORDER = ['Physical', 'Sensor', 'Data', 'Middleware',
               'Decision', 'Application', 'Social_Interface']

def extract_scores(matrix_path, weights_path, impl_path, output_path):
    """Extract layer scores using SAFE-R computation logic"""
    
    # Import the compute logic from score_safe_r
    import sys
    sys.path.append(str(pathlib.Path(__file__).parent))
    from score_safe_r import compute_scores, layer_of, LAYER_MAP
    
    # Load data
    A = pd.read_csv(matrix_path, index_col='Attack Vector')
    W = pd.read_csv(weights_path, index_col='Attack Vector')['Weight']
    I = pd.read_csv(impl_path, index_col='Defence')['Implementation']
    
    # Compute scores
    overall, layer_scores, E = compute_scores(A, W, I)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Layer': LAYER_ORDER,
        'Score': [layer_scores.get(layer, 0) * 5 for layer in LAYER_ORDER]
    })
    
    df.to_csv(output_path, index=False)
    print(f"✓ Layer scores saved to: {output_path}")
    print(f"  Overall SAFE-R: {overall*100:.1f}%")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix', required=True)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--impl', required=True, nargs='+')
    parser.add_argument('--outdir', default='data/layer_scores')
    
    args = parser.parse_args()
    
    out_root = pathlib.Path(args.outdir)
    out_root.mkdir(exist_ok=True, parents=True)
    
    for impl_path in args.impl:
        impl_path = pathlib.Path(impl_path)
        platform = impl_path.stem.replace('_implementation_status', '')
        output_path = out_root / f'{platform}_layer_scores.csv'
        
        extract_scores(args.matrix, args.weights, impl_path, output_path)

if __name__ == '__main__':
    main()