import torch
import os
from model.model import ParsingNet


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True

    assert cfg.backbone in ['18', '34', '50', '101', '152', '50next', '101next', '50wide', '101wide']
    
    cls_num_per_lane = 18

    net = ParsingNet(pretrained=False,
                     backbone=cfg.backbone, 
                     cls_dim=(cfg.griding_num+1, cls_num_per_lane, cfg.num_lanes),
                     use_aux=False).cuda()
    # Don't need auxiliary segmentation during testing

    state_dict = torch.load(cfg.test_model, map_location='cpu')['model']
    compatible_state_dict = {}
    for k, v in state_dict.items():
        if 'module.' in k:
            compatible_state_dict[k[7:]] = v
        else:
            compatible_state_dict[k] = v

    net.load_state_dict(compatible_state_dict, strict=False)

    if not os.path.exists(cfg.test_work_dir):
        os.mkdir(cfg.test_work_dir)

    eval_lane(net, cfg.dataset, cfg.data_root, cfg.test_work_dir, cfg.griding_num, False)


