import torch

import onnxruntime
import nnet
import utils
args = utils.get_args()
model = nnet.get_models(args)
ckpt_path = "./ckpt/kala_lmdb_fix_aug_2410/checkpoints/SVTR.ckpt"
# ckpt_path = "./ckpt/SVTR_kalapa_2710/checkpoints/SVTR.ckpt"
checkpoint = torch.load(ckpt_path, map_location=args.device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
model = model.to(args.device)
dummy_input = torch.rand([1,3,48,720]).to(args.device)
torch.onnx.export(
        model,
        dummy_input,
        'test_onnx.onnx',
        input_names = ['image'],
        output_names = ['logit'],
        # dynamic_axes = {'batch_images': {0: 'batch'},
        #                 'batch_logits': {0: 'batch'}},
        opset_version=15,
        verbose=True,
        # operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK
    )