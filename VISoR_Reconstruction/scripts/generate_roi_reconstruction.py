import argparse, json
from VISoR_Reconstruction.reconstruction_executor.roi_reconstruction_generator import gen_roi_reconstruction_pipeline
from VISoR_Brain.format.visor_data import VISoRData

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reconstruction task generator')
    parser.add_argument('--dataset', type=str, help='VISoR dataset file (*.visor).')
    parser.add_argument('--parameters', type=str, help='Parameters file.')
    parser.add_argument('--output_path', type=str, help='ROI Reconstruction output path.')
    parser.add_argument('file', type=str, help='Output task file.')
    args = parser.parse_args()
    with open(args.parameters) as f:
        param = json.load(f)
    param['output_path'] = args.output_path
    dataset = VISoRData(args.dataset)
    s = gen_roi_reconstruction_pipeline(dataset, **param)
    with open(args.file, 'w') as f:
        f.write(s)
